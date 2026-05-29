`timescale 1ns / 1ps

module linear_qkv_multihead (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    
    // Giao tiếp với RAM chứa ma trận X (Kích thước 64x64)
    output logic [11:0] addr_X,
    input  logic [15:0] data_X,
    
    // Giao tiếp với 3 ROM Trọng số Wq, Wk, Wv (Kích thước 64x64)
    // Tối ưu: Dùng chung 1 bus địa chỉ vì 3 ROM được quét đồng bộ
    output logic [11:0] addr_W,
    input  logic [15:0] data_Wq,
    input  logic [15:0] data_Wk,
    input  logic [15:0] data_Wv,
    
    // =========================================================
    // GIAO TIẾP VỚI 12 RAM CHỨA Q, K, V CHO 4 HEADS (Kích thước 64x16)
    // Sử dụng mảng SystemVerilog để khai báo 4 port cho gọn
    // =========================================================
    output logic        we_Q [4],
    output logic        we_K [4],
    output logic        we_V [4],
    output logic [9:0]  addr_out [4], // 10-bit địa chỉ cho 1024 words
    output logic [15:0] data_Q_out [4],
    output logic [15:0] data_K_out [4],
    output logic [15:0] data_V_out [4]
);
	typedef enum logic [1:0] {
        S_IDLE     = 2'd0,
        S_CALC_MAC = 2'd1,
        S_WRITE    = 2'd2,
        S_DONE     = 2'd3
    } state_t;

    state_t state;
    // ==========================================
    // 1. KHAI BÁO BỘ ĐẾM FSM
    // ==========================================
    logic [5:0] row_i; // Hàng đang tính (0 -> 63)
    logic [5:0] col_j; // Cột đang tính  (0 -> 63)
    logic [6:0] dot_k; // Biến chạy Dot-Product & Pipeline (0 -> 65)

    // Tạo địa chỉ đọc
    assign addr_X = {row_i, dot_k[5:0]}; // X[i, k]
    assign addr_W = {dot_k[5:0], col_j[5:0]}; // W[k, j]

    // ==========================================
    // 2. KHỞI TẠO 3 LÕI MAC SONG SONG (Q, K, V)
    // ==========================================
    logic mac_en, mac_clr;
    logic [15:0] q_mac_out, k_mac_out, v_mac_out;

    mac u_mac_q (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(data_X), .b_in(data_Wq), .mac_out(q_mac_out));
    mac u_mac_k (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(data_X), .b_in(data_Wk), .mac_out(k_mac_out));
    mac u_mac_v (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(data_X), .b_in(data_Wv), .mac_out(v_mac_out));

    // ==========================================
    // 3. LOGIC ĐỊNH TUYẾN KẾT QUẢ VÀO 4 HEADS
    // ==========================================
    logic [1:0] head_idx;
    logic [3:0] local_col;
    
    // Thuật toán bóc tách Head:
    // col_j = 0..63. 
    // Chia cho 16 lấy phần nguyên -> 2 bit cao [5:4] là số thứ tự Head (0, 1, 2, 3)
    // Chia cho 16 lấy dư -> 4 bit thấp [3:0] là số thứ tự cột cục bộ trong RAM Head (0..15)
    assign head_idx  = col_j[5:4];
    assign local_col = col_j[3:0];

    // Tạo khối ghép MUX định tuyến tự động vào mảng RAM
    always_comb begin
        for (int h = 0; h < 4; h++) begin
            we_Q[h] = 1'b0;
            we_K[h] = 1'b0;
            we_V[h] = 1'b0;
            
            // Địa chỉ ghi (Gồm 6 bit hàng và 4 bit cột cục bộ -> Tổng 10 bit)
            addr_out[h] = {row_i, local_col}; 
            
            // Dữ liệu luôn đẩy sẵn vào cả 4 port, nhưng chỉ RAM nào được bật `we` mới ăn data
            data_Q_out[h] = q_mac_out;
            data_K_out[h] = k_mac_out;
            data_V_out[h] = v_mac_out;
        end
        
        // Chỉ bật cờ Ghi (Write Enable) cho đúng Head đang xử lý tại chu kỳ S_WRITE
        if (state == S_WRITE) begin
            we_Q[head_idx] = 1'b1;
            we_K[head_idx] = 1'b1;
            we_V[head_idx] = 1'b1;
        end
    end

    // ==========================================
    // 4. MÁY TRẠNG THÁI TỔNG (MASTER FSM)
    // ==========================================
    

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            row_i   <= '0;
            col_j   <= '0;
            dot_k   <= '0;
            mac_clr <= 1'b0;
            mac_en  <= 1'b0;
        end 
        else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        row_i   <= '0;
                        col_j   <= '0;
                        dot_k   <= '0;
                        mac_clr <= 1'b1; // Cờ xóa sớm
                        state   <= S_CALC_MAC;
                    end
                end

                S_CALC_MAC: begin
                    // Pipeline giống attention_top của bạn
                    if (dot_k == 0) begin
                        mac_clr <= 1'b0; 
                        mac_en  <= 1'b1; 
                    end 
                    else if (dot_k >= 1 && dot_k <= 63) begin
                        mac_clr <= 1'b0;
                        mac_en  <= 1'b1; 
                    end
                    else if (dot_k == 64) begin
                        mac_en <= 1'b0; // RAM trả data 63 ra, MAC chốt nhịp cuối
                    end

                    if (dot_k < 65) begin
                        dot_k <= dot_k + 1'b1;
                    end else begin
                        state <= S_WRITE; 
                    end
                end

                S_WRITE: begin
                    // Ở chu kỳ này, mạch Tổ hợp (always_comb) phía trên sẽ tự kích we=1 cho đúng head_idx
                    dot_k <= '0;
                    
                    if (col_j < 63) begin
                        col_j   <= col_j + 1'b1;
                        mac_clr <= 1'b1; // Xóa MAC cho phần tử cột tiếp theo
                        state   <= S_CALC_MAC;
                    end else begin
                        col_j <= '0;
                        if (row_i < 63) begin
                            row_i   <= row_i + 1'b1;
                            mac_clr <= 1'b1; // Xóa MAC cho hàng tiếp theo
                            state   <= S_CALC_MAC;
                        end else begin
                            state   <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
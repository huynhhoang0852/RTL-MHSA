`timescale 1ns / 1ps

module attention_head (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    
    input  logic        linear_we,
    input  logic [9:0]  linear_addr, 
    input  logic [15:0] linear_data_Q,
    input  logic [15:0] linear_data_K,
    input  logic [15:0] linear_data_V,
     
    input  logic [9:0]  addr_Z_read,
    output logic [15:0] data_Z_out
);
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,
        S_MAC_PIPELINE  = 3'd1, 
        S_WRITE_RAM     = 3'd2,
        S_NEXT_ELEM     = 3'd3,
        S_START_SOFTMAX = 3'd4,
        S_WAIT_SOFTMAX  = 3'd5,
        S_DONE          = 3'd6
    } state_t;
    state_t state;

    // ==========================================
    // 1. KHAI BÁO TÍN HIỆU ĐỊNH TUYẾN FSM
    // ==========================================
    logic [1:0]  step;
    logic [6:0]  i, j, k;
    
    logic [6:0] dot_limit;
    assign dot_limit = (step == 0) ? 7'd16 : 7'd64;

    logic [15:0] mac1_result, mac2_result;
    logic mac_en, mac_clr;
    logic mac_settle;
    logic we_Score, we_Z;

    logic signed [15:0] max_array [64]; 

    // ==========================================
    // 2. BỘ NHỚ RAM Q, K, V
    // ==========================================
    logic [9:0]  addr_Q_A, addr_K_A, addr_K_B, addr_V_A, addr_V_B;
    logic [15:0] data_Q_A, data_K_A, data_K_B, data_V_A, data_V_B;

    assign addr_Q_A = linear_we ? linear_addr : {i[5:0], k[3:0]}; 
    assign addr_K_A = linear_we ? linear_addr : {j[5:0], k[3:0]}; 
    assign addr_K_B =                           {j[5:0] + 6'd1, k[3:0]}; 
    assign addr_V_A = linear_we ? linear_addr : {k[5:0], j[3:0]}; 
    assign addr_V_B =                           {k[5:0], j[3:0] + 4'd1}; 

    RAM_1024x16 u_ram_Q (.clock(clk), .address_a(addr_Q_A), .address_b(10'd0),    .data_a(linear_data_Q), .data_b(16'd0), .wren_a(linear_we), .wren_b(1'b0), .q_a(data_Q_A), .q_b());
    RAM_1024x16 u_ram_K (.clock(clk), .address_a(addr_K_A), .address_b(addr_K_B), .data_a(linear_data_K), .data_b(16'd0), .wren_a(linear_we), .wren_b(1'b0), .q_a(data_K_A), .q_b(data_K_B));
    RAM_1024x16 u_ram_V (.clock(clk), .address_a(addr_V_A), .address_b(addr_V_B), .data_a(linear_data_V), .data_b(16'd0), .wren_a(linear_we), .wren_b(1'b0), .q_a(data_V_A), .q_b(data_V_B));

    logic [9:0] addr_Z_A, addr_Z_B;
    assign addr_Z_A = (state == S_IDLE) ? addr_Z_read : {i[5:0], j[3:0]};
    assign addr_Z_B = {i[5:0], j[3:0] + 4'd1};
    RAM_1024x16 u_ram_Z (
        .clock(clk), 
        .address_a(addr_Z_A), .address_b(addr_Z_B), 
        .data_a(mac1_result),  .data_b(mac2_result), 
        .wren_a(we_Z),         .wren_b(we_Z), 
        .q_a(data_Z_out),      .q_b()
    );

    // ==========================================
    // 3. RAM SCORE
    //    - addr_Score_A/B chỉ được assign MỘT LẦN DUY NHẤT tại đây
    //    - Khi step==1: softmax đọc -> MUX sang softmax_addr_Score
    //    - Khi step==0/2: FSM ghi/đọc -> MUX sang {i,j}
    // ==========================================
    logic [11:0] softmax_addr_Score;
    logic [11:0] addr_Score_A, addr_Score_B; // Chỉ khai báo + assign 1 lần
    logic [15:0] data_Score_A_wire;          // Dây ra từ RAM Score port A
    logic        we_Score_B;

    assign addr_Score_A = (step == 1) ? softmax_addr_Score      : {i[5:0], j[5:0]};
    assign addr_Score_B = (step == 1) ? 12'd0                   : {i[5:0], j[5:0] + 6'd1};
    assign we_Score_B   = (step == 1) ? 1'b0                    : we_Score;

    // ==========================================
    // FIX #1: SCORE SCALING BẰNG BIT-SLICING VÀ LÀM TRÒN
    // Kỹ thuật: 
    // - Lấy 14 bit cao nhất `[15:2]` và copy bit dấu [15] 2 lần để giữ nguyên tính signed.
    // - Cộng thêm bit `[1]` (bit thập phân lớn nhất bị cắt đi) để làm tròn (Round half up).
    // - Bao bọc tất cả bằng $signed() để trình biên dịch hiểu đúng toán học có dấu.
    // ==========================================
    logic signed [15:0] score_even_scaled;
    logic signed [15:0] score_odd_scaled;

    assign score_even_scaled = $signed({ {2{mac1_result[15]}}, mac1_result[15:2] }) + $signed({15'd0, mac1_result[1]});
    assign score_odd_scaled  = $signed({ {2{mac2_result[15]}}, mac2_result[15:2] }) + $signed({15'd0, mac2_result[1]});

    RAM_4096x16 u_ram_Score (
        .clock     (clk),
        .address_a (addr_Score_A),    .address_b (addr_Score_B),
        .data_a    (score_even_scaled),.data_b   (score_odd_scaled),
        .wren_a    (we_Score),         .wren_b   (we_Score_B),
        .q_a       (data_Score_A_wire),.q_b      ()
    );

    // ==========================================
    // 4. RAM SOFTMAX (1-port)
    // ==========================================
    logic        softmax_we;
    logic [11:0] softmax_addr_Softmax;
    logic [15:0] softmax_data_out;   // Dữ liệu softmax_top GHI vào RAM
    logic [15:0] data_Softmax_out;   // Dữ liệu FSM ĐỌC ra từ RAM
    logic [11:0] addr_Softmax;
    logic        we_Softmax_ram;
 
    assign addr_Softmax   = (step == 1) ? softmax_addr_Softmax : {i[5:0], k[5:0]};
    assign we_Softmax_ram = (step == 1) ? softmax_we           : 1'b0;
 
    RAM_4096x16_1P u_ram_Softmax (
        .clock   (clk),
        .address (addr_Softmax),
        .data    (softmax_data_out),
        .wren    (we_Softmax_ram),
        .q       (data_Softmax_out)
    );
 
    // ==========================================
    // 5. ROM EXP + SOFTMAX TOP
    // ==========================================
    logic [12:0] softmax_addr_Exp;   // Output từ softmax_top, 13-bit unsigned
    logic [15:0] data_Exp_out;
    logic [11:0] safe_addr_Exp;      // Sau khi clamp, nối vào ROM
    logic [5:0]  softmax_current_row;
    logic signed [15:0] current_max_val;
 
    assign softmax_current_row = softmax_addr_Score[11:6];
    assign current_max_val     = max_array[softmax_current_row];
 
    // FIX: Chỉ clamp output của softmax_top, KHÔNG tính lại diff ở đây.
    // softmax_top tự tính diff = (score - max) + 4096 bên trong và xuất ra
    // softmax_addr_Exp (13-bit unsigned). Top-level chỉ bảo vệ ROM khỏi
    // địa chỉ >= 4095 (cận trên). Cận dưới (âm) đã được xử lý bên trong module.
    assign safe_addr_Exp = (softmax_addr_Exp >= 13'd4095) ? 12'd4095
                                                          : softmax_addr_Exp[11:0];
 
    ROM_EXP u_rom_exp (
        .clock   (clk),
        .address (safe_addr_Exp),
        .q       (data_Exp_out)
    );
 
    logic start_softmax, softmax_done;
 
    softmax_top u_softmax (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start_softmax),
        .done         (softmax_done),
        .addr_Score   (softmax_addr_Score),
        .data_Score   (data_Score_A_wire),  // signed [15:0] từ RAM Score
        .max_val_in   (current_max_val),    // signed [15:0], sign-extend đúng
        .addr_Exp     (softmax_addr_Exp),   // OUTPUT 13-bit từ softmax_top
        .data_Exp     (data_Exp_out),
        .we_Softmax   (softmax_we),
        .addr_Softmax (softmax_addr_Softmax),
        .data_Softmax (softmax_data_out)
    );

    // ==========================================
    // 6. MUX DỮ LIỆU CHO 2 LÕI MAC
    // ==========================================
    logic [15:0] mac_a, mac_1_b, mac_2_b;

    assign mac_a   = (step == 0) ? data_Q_A        : (step == 2) ? data_Softmax_out : 16'd0;
    assign mac_1_b = (step == 0) ? data_K_A        : (step == 2) ? data_V_A         : 16'd0;
    assign mac_2_b = (step == 0) ? data_K_B        : (step == 2) ? data_V_B         : 16'd0;

    mac u_mac_1 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(mac_a), .b_in(mac_1_b), .mac_out(mac1_result));
    mac u_mac_2 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(mac_a), .b_in(mac_2_b), .mac_out(mac2_result));

    // ==========================================
    // 7. MASTER FSM
    // ==========================================
    logic [6:0] j_limit;
    assign j_limit = (step == 0) ? 7'd62 : 7'd14;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            step          <= '0;
            we_Score      <= 1'b0;
            we_Z          <= 1'b0;
            mac_en        <= 1'b0;
            mac_clr       <= 1'b0;
            mac_settle    <= 1'b0;
            start_softmax <= 1'b0;
            i <= '0; j <= '0; k <= '0;
        end
        else begin
            case (state)
                // --------------------------------------------------
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        mac_clr    <= 1'b1;
                        mac_settle <= 1'b0;
                        step       <= '0;
                        i <= '0; j <= '0; k <= '0;
                        state      <= S_MAC_PIPELINE;
                    end
                end

                // --------------------------------------------------
                // FIX #2: Pipeline MAC với settle cycle
                //
                //   k=0          : mac_clr=0, mac_en=1 (bắt đầu tích lũy)
                //   k=1..N-1     : tích lũy bình thường
                //   k=N          : mac_en=0 (data[N-1] đã vào MAC nhịp này)
                //   k=N+1        : settle — accumulator chốt xong, chuyển S_WRITE_RAM
                // --------------------------------------------------
                S_MAC_PIPELINE: begin
                    if (k == 7'd0) begin
                        mac_clr    <= 1'b0;
                        mac_en     <= 1'b1;
                        mac_settle <= 1'b0;
                    end
                    else if (k == dot_limit) begin
                        mac_en <= 1'b0;
                    end
                    else if (k == dot_limit + 7'd1) begin
                        // Settle cycle: không tăng k, chuyển sang ghi
                        mac_settle <= 1'b1;
                        state      <= S_WRITE_RAM;
                    end

                    if (k <= dot_limit) begin
                        k <= k + 1'b1;
                    end
                end

                // --------------------------------------------------
                S_WRITE_RAM: begin
                    mac_settle <= 1'b0;

                    if (step == 2'd0) begin
                        we_Score <= 1'b1;
                        // Tìm Max online
                        if (j == 7'd0) begin
                            max_array[i] <= (score_even_scaled > score_odd_scaled)
                                             ? score_even_scaled : score_odd_scaled;
                        end else begin
                            if (score_even_scaled > max_array[i] && score_even_scaled >= score_odd_scaled)
                                max_array[i] <= score_even_scaled;
                            else if (score_odd_scaled > max_array[i])
                                max_array[i] <= score_odd_scaled;
                        end
                    end
                    else if (step == 2'd2) begin
                        we_Z <= 1'b1;
                    end

                    state <= S_NEXT_ELEM;
                end

                // --------------------------------------------------
                S_NEXT_ELEM: begin
                    we_Score <= 1'b0;
                    we_Z     <= 1'b0;

                    if (j < j_limit) begin
                        j       <= j + 7'd2;
                        k       <= '0;
                        mac_clr <= 1'b1;
                        state   <= S_MAC_PIPELINE;
                    end else begin
                        j <= '0;
                        if (i < 7'd63) begin
                            i       <= i + 1'b1;
                            k       <= '0;
                            mac_clr <= 1'b1;
                            state   <= S_MAC_PIPELINE;
                        end else begin
                            i <= '0;
                            if (step == 2'd0) begin
                                step  <= 2'd1;
                                state <= S_START_SOFTMAX;
                            end else if (step == 2'd2) begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                // --------------------------------------------------
                S_START_SOFTMAX: begin
                    start_softmax <= 1'b1;
                    state         <= S_WAIT_SOFTMAX;
                end

                S_WAIT_SOFTMAX: begin
                    start_softmax <= 1'b0;
                    if (softmax_done) begin
                        step    <= 2'd2;
                        i <= '0; j <= '0; k <= '0;
                        mac_clr <= 1'b1;
                        state   <= S_MAC_PIPELINE;
                    end
                end

                // --------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
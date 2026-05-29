`timescale 1ns / 1ps

module mhsa_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    // Giao tiếp với RAM chứa ma trận X đầu vào (Kích thước 64x64)
    output logic [11:0] addr_X,
    input  logic [15:0] data_X,

    // Giao tiếp với RAM Z_Final đầu ra cuối cùng (Kích thước 64x64 - RAM_2X)
    output logic        we_Z_final,
    output logic [11:0] addr_Z_final_A,
    output logic [11:0] addr_Z_final_B,
    output logic [15:0] data_Z_final_A,
    output logic [15:0] data_Z_final_B
);

    // ==========================================
    // 1. KHAI BÁO CÁC MẢNG TÍN HIỆU KẾT NỐI (ROUTING ARRAYS)
    // ==========================================
    // Tín hiệu điều hướng dữ liệu từ khối Linear QKV sang 4 Attention Heads
    logic        we_Q_arr [4], we_K_arr [4], we_V_arr [4];
    logic [9:0]  addr_QKV_arr [4];
    logic [15:0] data_Q_arr [4], data_K_arr [4], data_V_arr [4];

    // Tín hiệu điều khiển FSM chạy song song cho 4 Attention Heads
    logic start_heads;
    logic [3:0] done_heads; 
    
    // Tín hiệu kết nối từ 4 Attention Heads sang khối Linear Output để đọc ma trận Z
    logic [9:0]  addr_Z_read_arr [4];
    logic [15:0] data_Z_out_arr [4];

    // ==========================================
    // 2. KHỞI TẠO CÁC MODULE ROM TRỌNG SỐ CHUẨN (WEIGHT ROMs)
    // ==========================================
    // ROM_Wq, ROM_Wk, ROM_Wv sử dụng cấu hình Single-Port (4096 words x 16-bit)
    logic [11:0] addr_W_qkv;
    logic [15:0] data_Wq, data_Wk, data_Wv;
    
    ROM_Wq u_rom_Wq (.clock(clk), .address(addr_W_qkv), .q(data_Wq));
    ROM_Wk u_rom_Wk (.clock(clk), .address(addr_W_qkv), .q(data_Wk));
    ROM_Wv u_rom_Wv (.clock(clk), .address(addr_W_qkv), .q(data_Wv));

    // ROM_Wo sử dụng cấu hình Dual-Port (4096 words x 16-bit) để quét cặp cột song song
    logic [11:0] addr_Wo_A, addr_Wo_B;
    logic [15:0] data_Wo_A, data_Wo_B;
    
    ROM_Wo u_rom_Wo (
        .clock     (clk), 
        .address_a (addr_Wo_A), 
        .address_b (addr_Wo_B), 
        .q_a       (data_Wo_A), 
        .q_b       (data_Wo_B)
    );

    // ==========================================
    // 3. KHỞI TẠO MODULE LINEAR QKV
    // ==========================================
    logic start_linear_qkv, done_linear_qkv;

    linear_qkv_multihead u_linear_qkv (
        .clk(clk), .rst_n(rst_n), .start(start_linear_qkv), .done(done_linear_qkv),
        .addr_X(addr_X),         .data_X(data_X),
        .addr_W(addr_W_qkv),     .data_Wq(data_Wq), .data_Wk(data_Wk), .data_Wv(data_Wv),
        .we_Q(we_Q_arr),         .we_K(we_K_arr),   .we_V(we_V_arr),
        .addr_out(addr_QKV_arr),
        .data_Q_out(data_Q_arr), .data_K_out(data_K_arr), .data_V_out(data_V_arr)
    );

    // ==========================================
    // 4. KHỞI TẠO SONG SONG 4 KHỐI ATTENTION HEADS (Generate Block)
    // ==========================================
    genvar h;
    generate
        for (h = 0; h < 4; h++) begin : gen_heads
            attention_head u_head (
                .clk            (clk),
                .rst_n          (rst_n),
                .start          (start_heads),
                .done           (done_heads[h]),
                // Nạp dữ liệu đầu vào từ khối Linear QKV
                .linear_we      (we_Q_arr[h]), // Đồng bộ cờ ghi theo mảng we_Q
                .linear_addr    (addr_QKV_arr[h]),
                .linear_data_Q  (data_Q_arr[h]),
                .linear_data_K  (data_K_arr[h]),
                .linear_data_V  (data_V_arr[h]),
                // Kết nối đường ray cho phép khối Linear Output đọc kết quả Z
                .addr_Z_read    (addr_Z_read_arr[h]),
                .data_Z_out     (data_Z_out_arr[h])
            );
        end
    endgenerate

    // ==========================================
    // 5. KHỞI TẠO MODULE LINEAR OUTPUT
    // ==========================================
    logic start_linear_out, done_linear_out;

    linear_output u_linear_out (
        .clk        (clk), .rst_n(rst_n), .start(start_linear_out), .done(done_linear_out),
        // Thu thập cổng đọc từ 4 RAM Z của 4 Heads (Thực hiện Concat động)
        .addr_Z     (addr_Z_read_arr),
        .data_Z     (data_Z_out_arr),
        // Đọc ma trận trọng số ngõ ra W_O từ ROM
        .addr_Wo_A  (addr_Wo_A), .addr_Wo_B(addr_Wo_B),
        .data_Wo_A  (data_Wo_A), .data_Wo_B(data_Wo_B),
        // Chốt cờ ghi kết quả cuối cùng ra ngoài khối Attention
        .we_Out     (we_Z_final),
        .addr_Out_A (addr_Z_final_A), .addr_Out_B(addr_Z_final_B),
        .data_Out_A (data_Z_final_A), .data_Out_B(data_Z_final_B)
    );

    // ==========================================
    // 6. MASTER FSM (BỘ ĐIỀU KHIỂN CHỦ)
    // ==========================================
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,
        S_RUN_QKV       = 3'd1,
        S_RUN_HEADS     = 3'd2,
        S_RUN_OUT       = 3'd3,
        S_DONE          = 3'd4
    } state_t;
    
    state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            done             <= 1'b0;
            start_linear_qkv <= 1'b0;
            start_heads      <= 1'b0;
            start_linear_out <= 1'b0;
        end 
        else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        start_linear_qkv <= 1'b1; // Bật lệnh chạy Linear QKV
                        state            <= S_RUN_QKV;
                    end
                end

                S_RUN_QKV: begin
                    start_linear_qkv <= 1'b0; // Tắt xung kích hoạt ngay nhịp sau
                    if (done_linear_qkv) begin
                        start_heads <= 1'b1; // Khối QKV nạp xong RAM -> Bật cờ kích hoạt 4 Heads
                        state       <= S_RUN_HEADS;
                    end
                end

                S_RUN_HEADS: begin
                    start_heads <= 1'b0;
                    // Chờ toán tử AND bit logic (`&`) kiểm tra cả 4 đầu báo done đều bằng 1
                    if (&done_heads) begin 
                        start_linear_out <= 1'b1; // 4 Heads tính xong -> Bật lệnh chạy khối kết quả ra
                        state            <= S_RUN_OUT;
                    end
                end

                S_RUN_OUT: begin
                    start_linear_out <= 1'b0;
                    if (done_linear_out) begin
                        state <= S_DONE;
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
`timescale 1ns / 1ps

module linear_output (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    
    // Giao tiếp với 4 RAM Z của 4 Heads
    output logic [9:0]  addr_Z [4],
    input  logic [15:0] data_Z [4],
    
    // Giao tiếp với ROM Trọng số ngõ ra W_O
    output logic [11:0] addr_Wo_A,
    output logic [11:0] addr_Wo_B,
    input  logic [15:0] data_Wo_A,
    input  logic [15:0] data_Wo_B,
    
    // Giao tiếp với RAM Output cuối cùng
    output logic        we_Out,
    output logic [11:0] addr_Out_A,
    output logic [11:0] addr_Out_B,
    output logic [15:0] data_Out_A,
    output logic [15:0] data_Out_B
);

    // ==========================================
    // 1. KHAI BÁO CÁC BIẾN ĐẾM VÀ TRẠNG THÁI FSM
    // ==========================================
    logic [6:0] i, j, k; 
    
    typedef enum logic [2:0] {
        S_IDLE         = 3'd0,
        S_MAC_PIPELINE = 3'd1,
        S_WRITE_RAM    = 3'd2,
        S_NEXT_ELEM    = 3'd3,
        S_DONE         = 3'd4
    } state_t;
    
    state_t state;
    
    logic mac_en, mac_clr;
    logic [15:0] mac1_result, mac2_result;

    // ==========================================
    // 2. BỘ MUX GHÉP NỐI Z (ĐÃ FIX ĐỒNG BỘ LATENCY)
    // ==========================================
    logic [1:0] head_idx;
    logic [3:0] local_k;
    logic [15:0] z_concat_data;

    assign head_idx = k[5:4]; 
    assign local_k  = k[3:0]; 

    // BẢN VÁ LỖI: Tạo thanh ghi trễ 1 chu kỳ cho head_idx để khớp với thời gian đọc RAM
    logic [1:0] head_idx_delayed;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) 
            head_idx_delayed <= 2'd0;
        else        
            head_idx_delayed <= head_idx;
    end

    always_comb begin
        for (int h = 0; h < 4; h++) begin
            addr_Z[h] = {i[5:0], local_k};
        end
        
        // Dùng head_idx ĐÃ DELAY để đọc data_Z (Khắc phục hoàn toàn lỗi rác dữ liệu ở biên)
        case (head_idx_delayed)
            2'd0: z_concat_data = data_Z[0];
            2'd1: z_concat_data = data_Z[1];
            2'd2: z_concat_data = data_Z[2];
            2'd3: z_concat_data = data_Z[3];
            default: z_concat_data = 16'd0;
        endcase
    end

    // ==========================================
    // 3. ĐỊA CHỈ ROM TRỌNG SỐ W_O VÀ RAM OUTPUT
    // ==========================================
    assign addr_Wo_A = {k[5:0], j[5:0]};
    assign addr_Wo_B = {k[5:0], j[5:0] + 6'd1};

    assign addr_Out_A = {i[5:0], j[5:0]};
    assign addr_Out_B = {i[5:0], j[5:0] + 6'd1};
    
    assign data_Out_A = mac1_result;
    assign data_Out_B = mac2_result;

    // ==========================================
    // 4. KHỞI TẠO DUAL-CORE MAC PIPELINE
    // ==========================================
    mac u_mac_out1 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(z_concat_data), .b_in(data_Wo_A), .mac_out(mac1_result));
    mac u_mac_out2 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr_acc(mac_clr), .a_in(z_concat_data), .b_in(data_Wo_B), .mac_out(mac2_result));

    // ==========================================
    // 5. MÁY TRẠNG THÁI ĐIỀU KHIỂN (MASTER FSM)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            we_Out   <= 1'b0;
            mac_en   <= 1'b0;
            mac_clr  <= 1'b0;
            i        <= '0;
            j        <= '0;
            k        <= '0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        mac_clr <= 1'b1; 
                        state   <= S_MAC_PIPELINE;
                        i       <= '0;
                        j       <= '0;
                        k       <= '0;
                    end
                end

                S_MAC_PIPELINE: begin
                    if (k == 0) begin
                        mac_clr <= 1'b0;
                        mac_en  <= 1'b1; 
                    end
                    else if (k >= 1 && k <= 63) begin
                        mac_clr <= 1'b0;
                        mac_en  <= 1'b1; 
                    end
                    else if (k == 64) begin
                        mac_en  <= 1'b0; 
                    end

                    if (k < 65) begin
                        k <= k + 1'b1;
                    end else begin
                        k     <= '0;
                        state <= S_WRITE_RAM; 
                    end
                end

                S_WRITE_RAM: begin
                    we_Out <= 1'b1; 
                    state  <= S_NEXT_ELEM;
                end

                S_NEXT_ELEM: begin
                    we_Out <= 1'b0; 
                    
                    if (j < 62) begin
                        j       <= j + 2; 
                        mac_clr <= 1'b1;  
                        state   <= S_MAC_PIPELINE;
                    end else begin
                        j <= '0;
                        if (i < 63) begin
                            i       <= i + 1'b1; 
                            mac_clr <= 1'b1;
                            state   <= S_MAC_PIPELINE;
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
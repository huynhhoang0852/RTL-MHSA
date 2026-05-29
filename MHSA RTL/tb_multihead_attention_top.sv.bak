`timescale 1ns / 1ps

module tb_multihead_attention_top();

    // ==========================================
    // 1. KHAI BÁO TÍN HI?U
    // ==========================================
    logic        clk;
    logic        rst_n;
    logic        start;
    logic        done;

    // Bus RAM X
    logic [11:0] addr_X;
    logic [15:0] data_X;

    // Bus RAM Z Final
    logic        we_Z_final;
    logic [11:0] addr_Z_final_A;
    logic [11:0] addr_Z_final_B;
    logic [15:0] data_Z_final_A;
    logic [15:0] data_Z_final_B;

    // ==========================================
    // 2. KH?I T?O MODULE TOP (DUT)
    // ==========================================
    mhsa_top uut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .done           (done),
        .addr_X         (addr_X),
        .data_X         (data_X),
        .we_Z_final     (we_Z_final),
        .addr_Z_final_A (addr_Z_final_A),
        .addr_Z_final_B (addr_Z_final_B),
        .data_Z_final_A (data_Z_final_A),
        .data_Z_final_B (data_Z_final_B)
    );

    // ==========================================
    // 3. T?O CLOCK (T?n s? 100MHz -> Chu k? 10ns)
    // ==========================================
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ==========================================
    // 4. MÔ PH?NG RAM X & RAM Z (Behavioral RAM)
    // ==========================================
    // M?ng nh? mô ph?ng 4096 words x 16 bit
    logic [15:0] ram_x_mem [0:4095];
    logic [15:0] ram_z_mem [0:4095];

    // RAM X (Read-Only) có ?? tr? 1 chu k? (Kh?p v?i M10K block)
    always_ff @(posedge clk) begin
        data_X <= ram_x_mem[addr_X];
    end

    // RAM Z (Write-Only Dual Port) ?? h?ng k?t qu?
    always_ff @(posedge clk) begin
        if (we_Z_final) begin
            ram_z_mem[addr_Z_final_A] <= data_Z_final_A;
            ram_z_mem[addr_Z_final_B] <= data_Z_final_B;
        end
    end

    // ==========================================
    // 5. K?CH B?N TEST CHÍNH
    // ==========================================
    initial begin
        int fd;
        
        $display("==================================================");
        $display(" B?T ??U MÔ PH?NG MULTI-HEAD ATTENTION");
        $display("==================================================");

        // N?p ma tr?n X vào RAM gi? l?p t? file hex
        $readmemh("C:/Users/Dell/Downloads/VSCode/MHSA/matrix_X.hex", ram_x_mem);
        
        // Kh?i t?o các tín hi?u reset
        rst_n = 1'b0;
        start = 1'b0;
        
        // Ch? 20ns r?i nh? reset
        #20;
        rst_n = 1'b1;
        $display("[%0t] H? th?ng ?ã s?n sàng. Kích ho?t tín hi?u START...", $time);
        
        // B?n xung start 1 chu k? clock
        #10;
        start = 1'b1;
        #10;
        start = 1'b0;

        // ??i c? done báo hoàn thành
        wait(done == 1'b1);
        $display("==================================================");
        $display("[%0t] THU?T TOÁN HOÀN T?T!", $time);
        
        // Trích xu?t k?t qu? RAM Z ra file text ?? ki?m tra v?i Python
        fd = $fopen("matrix_Z_out.txt", "w");
        for (int i = 0; i < 4096; i++) begin
            $fdisplay(fd, "%04x", ram_z_mem[i]); // Xu?t format %04x (ch? th??ng) cho d? diff
        end
        $fclose(fd);
        $display("-> ?ã l?u k?t qu? ma tr?n Z vào file 'matrix_Z_out.txt'");
        $display("==================================================");

        #100;
        $stop; // D?ng mô ph?ng (ModelSim/Questa)
    end 

    // ==========================================
    // 6. TI?N TRÌNH THEO DÕI FSM
    // ==========================================
    initial begin
        wait(uut.state == 3'd1); $display("[%0t] ?ang ch?y Phase 1: Linear QKV...", $time);
        wait(uut.state == 3'd2); $display("[%0t] ?ang ch?y Phase 2: B?n Attention Heads...", $time);
        wait(uut.state == 3'd3); $display("[%0t] ?ang ch?y Phase 3: Linear Output...", $time);
    end

    // ==========================================================
    // 7. KH?I DEBUG CHUYÊN SÂU: GHI DATA T?T C? CÁC CH?NG (HEAD 0)
    // ==========================================================
    integer fd_q, fd_k, fd_v, fd_score, fd_softmax;

    initial begin
        fd_q       = $fopen("rtl_debug_Q_head0.txt", "w");
        fd_k       = $fopen("rtl_debug_K_head0.txt", "w");
        fd_v       = $fopen("rtl_debug_V_head0.txt", "w");
        fd_score   = $fopen("rtl_debug_Score_head0.txt", "w");
        fd_softmax = $fopen("rtl_debug_Softmax_head0.txt", "w");
    end

    // CH?NG 1: L?y Q, K, V t? Linear Output
    always_ff @(posedge clk) begin
        if (uut.we_Q_arr[0]) begin
            $fdisplay(fd_q, "%04x", uut.data_Q_arr[0]);
            $fdisplay(fd_k, "%04x", uut.data_K_arr[0]);
            $fdisplay(fd_v, "%04x", uut.data_V_arr[0]);
        end
    end

    // CH?NG 2: L?y Score t? bên trong Attention Head 0
    always_ff @(posedge clk) begin
        if (uut.gen_heads[0].u_head.we_Score) begin
            $fdisplay(fd_score, "%04x", uut.gen_heads[0].u_head.score_even_scaled);
            $fdisplay(fd_score, "%04x", uut.gen_heads[0].u_head.score_odd_scaled);
        end
    end

    // CH?NG 3: L?y Softmax t? bên trong Attention Head 0
    always_ff @(posedge clk) begin
        if (uut.gen_heads[0].u_head.we_Softmax_ram) begin
            $fdisplay(fd_softmax, "%04x", uut.gen_heads[0].u_head.softmax_data_out);
        end
    end
endmodule
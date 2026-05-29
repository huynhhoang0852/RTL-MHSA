`timescale 1ns / 1ps

// =============================================================
// Module  : mac_unit
// Chức năng: Multiply-Accumulate (MAC) cho phép nhân ma trận
//            Fixed-point Q8.8, tích lũy 64 phần tử dot-product
//
// Timing với attention_top FSM:
//   S_READ_MEM : clr_acc=1, en=0  -> acc_40 bị XÓA VỀ 0 ngay cycle này
//   S_WAIT_MEM : clr_acc=0, en=0  -> giữ nguyên (data RAM đang ra)
//   S_MAC_CALC : clr_acc=0, en=1  -> bắt đầu tích lũy full_product
//   ...x64...
//   S_WRITE_RAM: clr_acc=0, en=0  -> acc_40 giữ nguyên, mac_out hợp lệ
// =============================================================

module mac (
    input  logic               clk,
    input  logic               rst_n,    // Reset tích cực mức thấp
    input  logic               en,       // Cho phép tích lũy MAC
    input  logic               clr_acc,  // Xóa accumulator (độc lập với en)
    input  logic signed [15:0] a_in,     // Toán hạng A (Q8.8)
    input  logic signed [15:0] b_in,     // Toán hạng B (Q8.8)
    output logic signed [15:0] mac_out   // Kết quả (Q8.8), combinational
);

    // ----------------------------------------------------------
    // BƯỚC 1: PHÉP NHÂN TỔ HỢP
    //   a_in * b_in -> full_product (Q16.16, 32-bit signed)
    // ----------------------------------------------------------
    logic signed [31:0] full_product;
    assign full_product = a_in * b_in;

    // ----------------------------------------------------------
    // BƯỚC 2: BỘ CỘNG DỒN 40-BIT
    // ----------------------------------------------------------
    logic signed [39:0] acc_40;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_40 <= '0;
        end
        else if (clr_acc) begin
            // Xóa accumulator, sẵn sàng cho dot-product mới
            acc_40 <= '0;
        end
        else if (en) begin
            // Cộng dồn: SystemVerilog tự động sign-extend với cú pháp ép kiểu (cast)
            acc_40 <= acc_40 + 40'(full_product);
        end
    end

    // ----------------------------------------------------------
    // BƯỚC 3: CẮT BIT NGÕ RA VỚI ROUND-TO-NEAREST
    // ----------------------------------------------------------
    logic signed [39:0] acc_rounded;
    
    // Cộng 0x80 (bit[7] = 0.5 LSB) trước khi cắt để làm tròn
    assign acc_rounded = acc_40 + 40'sh80; 

    assign mac_out = acc_rounded[23:8];

endmodule
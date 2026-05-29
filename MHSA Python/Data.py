import numpy as np

# ==========================================
# CẤU HÌNH THÔNG SỐ (64x64)
# ==========================================
ROWS = 64
COLS = 64
TOTAL_WORDS = ROWS * COLS  # 4096 words
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS  # 256 (Chuẩn Q8.8)

# ==========================================
# HÀM CHUYỂN ĐỔI SỐ THỰC SANG HEX Q8.8
# ==========================================
def float_to_q88_hex(val_real):
    # Lượng tử hóa sang chuẩn Q8.8 (Nhân 256 và làm tròn)
    val_q88 = int(round(val_real * SCALE))
    # Kẹp giá trị (Clipping) để đảm bảo không tràn 16-bit có dấu (-32768 đến 32767)
    val_q88 = max(min(val_q88, 32767), -32768)
    # Chuyển đổi bù 2 (Two's complement) thành chuỗi Hex 16-bit
    return f"{val_q88 & 0xFFFF:04X}"

# ==========================================
# HÀM XUẤT FILE .MIF (Cho Quartus ROM IP)
# ==========================================
def write_mif(filename, data_matrix):
    flat_data = data_matrix.flatten()
    with open(filename, "w") as f:
        f.write(f"DEPTH = {TOTAL_WORDS};\n")
        f.write("WIDTH = 16;\n")
        f.write("ADDRESS_RADIX = UNS;\n")
        f.write("DATA_RADIX = HEX;\n\n")
        f.write("CONTENT BEGIN\n")
        for addr in range(TOTAL_WORDS):
            hex_str = float_to_q88_hex(flat_data[addr])
            f.write(f"\t{addr:4d} : {hex_str};\n")
        f.write("END;\n")
    print(f"[Thành công] Đã tạo file '{filename}' (MIF - {TOTAL_WORDS} words)")

# ==========================================
# HÀM XUẤT FILE .HEX (Cho $readmemh trong Testbench)
# ==========================================
def write_hex(filename, data_matrix):
    flat_data = data_matrix.flatten()
    with open(filename, "w") as f:
        for addr in range(TOTAL_WORDS):
            hex_str = float_to_q88_hex(flat_data[addr])
            # File hex thuần túy chỉ chứa dữ liệu, mỗi dòng 1 giá trị
            f.write(f"{hex_str}\n")
    print(f"[Thành công] Đã tạo file '{filename}' (HEX - {TOTAL_WORDS} words)")

# ==========================================
# HÀM MAIN: TẠO VÀ LƯU CÁC MA TRẬN
# ==========================================
if __name__ == "__main__":
    # Khởi tạo seed để kết quả random cố định (giúp dễ debug RTL Waveform)
    np.random.seed(42)

    # Sinh các ma trận ngẫu nhiên (Giá trị từ -0.2 đến 0.2 để chống tràn bộ MAC)
    min_val, max_val = -0.2, 0.2
    
    matrix_X  = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wq = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wk = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wv = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wo = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS)) # Ma trận trọng số đầu ra

    # Xuất ma trận đầu vào X ra định dạng .hex (Để mô phỏng bằng $readmemh)
    write_hex("matrix_X.hex", matrix_X)

    # Xuất các ma trận trọng số ra định dạng .mif (Để nạp vào ROM IP trong Quartus)
    write_mif("weights_Wq.mif", matrix_Wq)
    write_mif("weights_Wk.mif", matrix_Wk)
    write_mif("weights_Wv.mif", matrix_Wv)
    write_mif("weights_Wo.mif", matrix_Wo)
    
    print("\n[Hoàn tất] Tất cả các file đã sẵn sàng để nạp vào FPGA/ModelSim!")
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
# HÀM XUẤT FILE .MIF (Cho Quartus ROM/RAM IP)
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
    print(f"[+] Đã tạo file MIF: {filename}")

# ==========================================
# HÀM XUẤT FILE .HEX (Cho $readmemh trong Testbench)
# ==========================================
def write_hex(filename, data_matrix):
    flat_data = data_matrix.flatten()
    with open(filename, "w") as f:
        for addr in range(TOTAL_WORDS):
            hex_str = float_to_q88_hex(flat_data[addr])
            f.write(f"{hex_str}\n")
    print(f"[+] Đã tạo file HEX: {filename}")

# ==========================================
# HÀM XUẤT FILE .TXT FLOAT (Để dễ debug/kiểm tra)
# ==========================================
def write_float_txt(filename, data_matrix):
    with open(filename, "w") as f:
        for i in range(ROWS):
            row_str = " ".join([f"{val:8.4f}" for val in data_matrix[i]])
            f.write(row_str + "\n")
    print(f"[+] Đã tạo file FLOAT TXT: {filename}")

# ==========================================
# HÀM XUẤT ĐỒNG LOẠT 3 ĐỊNH DẠNG CHO 1 MA TRẬN
# ==========================================
def export_matrix(base_name, data_matrix):
    write_mif(f"{base_name}.mif", data_matrix)
    write_hex(f"{base_name}.hex", data_matrix)
    write_float_txt(f"{base_name}_float.txt", data_matrix)

# ==========================================
# HÀM MAIN: TẠO VÀ LƯU CÁC MA TRẬN
# ==========================================
if __name__ == "__main__":
    print("Đang khởi tạo các ma trận...")
    # Khởi tạo seed để kết quả random cố định (giúp dễ debug RTL Waveform)
    np.random.seed(42)

    # Sinh các ma trận ngẫu nhiên (Giá trị từ -0.2 đến 0.2 để chống tràn bộ MAC)
    min_val, max_val = -0.2, 0.2
    
    matrix_X  = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wq = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wk = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wv = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))
    matrix_Wo = np.random.uniform(low=min_val, high=max_val, size=(ROWS, COLS))

    print("\n--- XUẤT MA TRẬN ĐẦU VÀO X ---")
    export_matrix("matrix_X", matrix_X)

    print("\n--- XUẤT CÁC MA TRẬN TRỌNG SỐ ---")
    export_matrix("weights_Wq", matrix_Wq)
    export_matrix("weights_Wk", matrix_Wk)
    export_matrix("weights_Wv", matrix_Wv)
    export_matrix("weights_Wo", matrix_Wo)
    
    print("\n[HOÀN TẤT] Tất cả các file đã sẵn sàng! Hãy chép chúng vào thư mục dự án ModelSim.")
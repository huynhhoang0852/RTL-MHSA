import math

# ==========================================
# CẤU HÌNH THÔNG SỐ ROM EXP
# ==========================================
DEPTH = 4096           # Khớp với 12-bit address (0 -> 4095)
WIDTH = 16             # Chuẩn Q8.8 (16-bit)
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS  # 256

def float_to_q88_unsigned(val_real):
    """Lượng tử hóa số thực sang số nguyên dương Q8.8 (16-bit)"""
    val_q88 = int(round(val_real * SCALE))
    # Vì hàm e^x luôn cho kết quả >= 0, ta kẹp giới hạn từ 0 đến 65535
    return max(min(val_q88, 65535), 0)

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    print(f"Đang tính toán và tạo bảng LUT Exponential (DEPTH = {DEPTH})...")

    with open("exp_lut.mif", "w") as f_mif, open("exp_lut.hex", "w") as f_hex:
        # 1. Viết Header cho file MIF
        f_mif.write(f"DEPTH = {DEPTH};\n")
        f_mif.write(f"WIDTH = {WIDTH};\n")
        f_mif.write("ADDRESS_RADIX = UNS;\n")
        f_mif.write("DATA_RADIX = HEX;\n\n")
        f_mif.write("CONTENT BEGIN\n")
        
        # 2. Quét qua 4096 địa chỉ để tính e^x
        for addr in range(DEPTH):
            # Trong RTL của bạn: addr = x_norm + 4096
            # Suy ra giá trị thực: x_real = (addr - 4096) / 256.0
            x_real = (addr - 4096) / 256.0
            
            # Tính e^x
            exp_real = math.exp(x_real)
            
            # Chuyển đổi sang Hex Q8.8
            hex_str = f"{float_to_q88_unsigned(exp_real):04X}"
            
            # Ghi vào file
            f_mif.write(f"\t{addr:4d} : {hex_str};\n")
            f_hex.write(f"{hex_str}\n")
            
        f_mif.write("END;\n")

    print("[Thành công] Đã tạo xong 'exp_lut.mif' và 'exp_lut.hex'!")
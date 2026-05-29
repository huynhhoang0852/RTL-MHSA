import numpy as np

# ==========================================
# CẤU HÌNH THÔNG SỐ (64x64 | Q8.8)
# ==========================================
ROWS = 64
COLS = 64
TOTAL_WORDS = ROWS * COLS          # 4096 words
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS       # 256  →  Q8.8
Q_MIN = -32768
Q_MAX =  32767

# Dải giá trị float đầu vào (tránh overflow MAC)
MIN_VAL, MAX_VAL = -0.2, 0.2

# ==========================================
# HÀM CHUYỂN ĐỔI: float → Q8.8 hex 16-bit
# ==========================================
def float_to_q88_hex(val: float) -> str:
    """
    Chuyển số thực sang chuẩn Q8.8 (16-bit có dấu, bù 2).
      - Nhân với 256 rồi làm tròn
      - Clipping về [-32768, 32767]
      - Trả về chuỗi hex 4 ký tự (VD: "FF80")
    """
    val_q = int(round(val * SCALE))
    val_q = max(min(val_q, Q_MAX), Q_MIN)
    return f"{val_q & 0xFFFF:04X}"

# ==========================================
# HÀM XUẤT FILE .MIF  (Quartus ROM IP)
# ==========================================
def write_mif(filename: str, data_matrix: np.ndarray) -> None:
    """
    Xuất ma trận sang file .mif theo bố cục column-major (Fortran order).
    Layout: địa chỉ addr = col*ROWS + row  →  flatten(order='C')
    """
    flat = data_matrix.flatten(order='C')        # column-major
    with open(filename, "w") as f:
        f.write(f"DEPTH = {TOTAL_WORDS};\n")
        f.write("WIDTH = 16;\n")
        f.write("ADDRESS_RADIX = UNS;\n")
        f.write("DATA_RADIX = HEX;\n\n")
        f.write("CONTENT BEGIN\n")
        for addr, val in enumerate(flat):
            f.write(f"\t{addr:5d} : {float_to_q88_hex(val)};\n")
        f.write("END;\n")
    print(f"  [MIF]  '{filename}'  ({TOTAL_WORDS} words, column-major)")

# ==========================================
# HÀM XUẤT FILE .HEX  ($readmemh testbench)
# ==========================================
def write_hex(filename: str, data_matrix: np.ndarray) -> None:
    """
    Xuất ma trận sang file .hex thuần (mỗi dòng 1 giá trị hex 16-bit).
    Layout ROW-MAJOR: addr = row*64 + col  (khớp với RTL addr = {row_i, col_j}).
    """
    flat = data_matrix.flatten(order='C')        # column-major
    with open(filename, "w") as f:
        for val in flat:
            f.write(f"{float_to_q88_hex(val)}\n")
    print(f"  [HEX]  '{filename}'  ({TOTAL_WORDS} words, column-major)")

# ==========================================
# HÀM XUẤT FILE .TXT  (float tham chiếu)
# ==========================================
def write_float_txt(filename: str, name: str, data_matrix: np.ndarray) -> None:
    """
    Lưu ma trận float gốc ra file text để tiện kiểm tra / debug
    bằng Python reference model.
    """
    with open(filename, "w") as f:
        f.write(f"# Matrix: {name}  |  shape: {data_matrix.shape}"
                f"  |  dtype: float64  |  Q8.8 scale: {SCALE}\n")
        f.write(f"# format: row-major  |  range: [{MIN_VAL}, {MAX_VAL})\n\n")
        for r in range(ROWS):
            row_str = "  ".join(f"{data_matrix[r, c]:+.6f}" for c in range(COLS))
            f.write(row_str + "\n")
    print(f"  [TXT]  '{filename}'  (float64 reference)")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)          # Seed cố định → dễ debug RTL waveform

    print("=" * 55)
    print("  Sinh ma trận ngẫu nhiên  (64×64 | Q8.8 | seed=42)")
    print("=" * 55)

    # ── 1. Sinh ma trận float ──────────────────────────────
    matrices = {
        "X" : np.random.uniform(MIN_VAL, MAX_VAL, (ROWS, COLS)),
        "Wq": np.random.uniform(MIN_VAL, MAX_VAL, (ROWS, COLS)),
        "Wk": np.random.uniform(MIN_VAL, MAX_VAL, (ROWS, COLS)),
        "Wv": np.random.uniform(MIN_VAL, MAX_VAL, (ROWS, COLS)),
        "Wo": np.random.uniform(MIN_VAL, MAX_VAL, (ROWS, COLS)),
    }

    # ── 2. In thống kê nhanh ──────────────────────────────
    print("\n[Thống kê float trước khi lượng tử hóa]")
    print(f"  {'Tên':<5}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, mat in matrices.items():
        print(f"  {name:<5}  {mat.min():+10.6f}  {mat.max():+10.6f}"
              f"  {mat.mean():+10.6f}  {mat.std():10.6f}")

    # ── 3. Xuất file ──────────────────────────────────────
    

    print("\n[Xuất file .mif  (cho Quartus ROM IP)]")
    # Ma trận X cũng xuất .mif để khởi tạo RAM đầu vào nếu cần
    for name, mat in matrices.items():
        write_mif(f"weights_{name}.mif", mat)

    print("\n[Xuất file .hex  (cho $readmemh testbench)]")
    for name, mat in matrices.items():
        write_hex(f"weights_{name}.hex", mat)

    # ── 4. Tóm tắt ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Hoàn tất! Danh sách file đã tạo:")
    print("=" * 55)
    for name in matrices:
        print(f"  weights_{name}.mif         ← Quartus ROM IP")
        print(f"  weights_{name}.hex         ← $readmemh testbench")
    print()
    print("  Lưu ý:")
    print("  • Chuẩn Q8.8: 1 bit dấu, 7 bit nguyên, 8 bit phần lẻ")
    print("  • Giá trị float được nhân 256 → làm tròn → clamp 16-bit")
    print("  • Layout bộ nhớ: ROW-MAJOR    (addr = row*64 + col)")
    print("  • Seed NumPy = 42  →  kết quả luôn tái tạo được")
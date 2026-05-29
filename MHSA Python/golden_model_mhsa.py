# -*- coding: utf-8 -*-
"""
================================================================================
  GOLDEN MODEL - Multi-Head Self-Attention (MHSA)  |  64x64  |  NH=4
  Bản hoàn chỉnh: Đồng bộ toán học Hardware & Xuất file Debug (Q, Score, Softmax)
  + So sánh RTL (Q8.8) vs Float (IEEE 754) để xuất báo cáo
================================================================================
"""

import numpy as np
import os
import sys

# ============================================================
#  A. CẤU HÌNH
# ============================================================
D    = 64          
NH   = 4           
HD   = D // NH     
ROWS = D
COLS = D

SCALE = 256        
Q_MIN, Q_MAX = -32768, 32767

np.random.seed(42)

def float_to_q88(val: float) -> int:
    v = int(round(val * SCALE))
    return max(min(v, Q_MAX), Q_MIN)

def q88_to_float(val_int: int) -> float:
    if val_int > Q_MAX:
        val_int -= 65536
    return val_int / SCALE

def mat_to_q88(mat: np.ndarray) -> np.ndarray:
    vfunc = np.vectorize(float_to_q88)
    return vfunc(mat).astype(np.int32)

def mat_q88_to_float(mat: np.ndarray) -> np.ndarray:
    return mat.astype(np.float64) / SCALE

# ============================================================
#  ĐỌC FILE INPUT
# ============================================================
def load_hex_rowmajor(filepath: str, rows: int, cols: int) -> np.ndarray:
    flat_q88 = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val = int(line, 16)
                flat_q88.append(q88_to_float(val))
    return np.array(flat_q88, dtype=np.float64).reshape((rows, cols))

def load_mif_rowmajor(filepath: str, rows: int, cols: int) -> np.ndarray:
    flat_q88 = []
    with open(filepath, encoding="utf-8") as f:
        in_content = False
        for line in f:
            line = line.strip()
            if line.upper() == "CONTENT BEGIN":
                in_content = True
                continue
            if not in_content or line.upper() == "END;":
                if line.upper() == "END;": break
                continue
            if ":" in line:
                hex_str = line.split(":")[1].strip().rstrip(";")
                val = int(hex_str, 16)
                flat_q88.append(q88_to_float(val))
    return np.array(flat_q88, dtype=np.float64).reshape((rows, cols))

# ============================================================
#  HÀM TOÁN HỌC BIT-ACCURATE ĐỒNG BỘ VỚI RTL
# ============================================================
def matmul_q88(A_f: np.ndarray, B_f: np.ndarray) -> np.ndarray:
    A_q = mat_to_q88(A_f)
    B_q = mat_to_q88(B_f)
    C_q = A_q @ B_q
    C_q = np.right_shift(C_q + 128, 8)
    C_q = np.clip(C_q, Q_MIN, Q_MAX)
    return C_q.astype(np.float64) / SCALE

def scale_score_rtl(score_f: np.ndarray) -> np.ndarray:
    score_q88 = mat_to_q88(score_f).astype(np.int32)
    out_q88   = np.zeros_like(score_q88)
    for i in range(score_q88.shape[0]):
        for j in range(score_q88.shape[1]):
            x = int(score_q88[i, j])
            if x >= 0:
                out_q88[i, j] = (x + 2) >> 2
            else:
                floor_div4 = x >> 2
                round_bit  = (x & 0xFFFF) >> 1 & 1
                out_q88[i, j] = floor_div4 + round_bit
    out_q88 = np.clip(out_q88, Q_MIN, Q_MAX)
    return mat_q88_to_float(out_q88)

def softmax_q88(x_f: np.ndarray) -> np.ndarray:
    score_q88 = mat_to_q88(x_f)
    M, N = score_q88.shape
    out_q88 = np.zeros_like(score_q88)
    for i in range(M):
        max_val = np.max(score_q88[i])
        sum_exp = 0
        exp_vals = np.zeros(N, dtype=np.int32)
        for j in range(N):
            x_norm = score_q88[i, j] - max_val
            addr_raw = x_norm + 4096
            addr = max(0, min(addr_raw, 4095)) 
            x_real = (addr - 4096) / 256.0
            exp_val = int(round(np.exp(x_real) * 256))
            exp_val = max(min(exp_val, 65535), 0)
            exp_vals[j] = exp_val
            sum_exp += exp_val
        for j in range(N):
            numer = exp_vals[j] << 8
            if sum_exp == 0:
                quotient, round_bit = 0, 0
            else:
                quotient = numer // sum_exp
                remain = numer % sum_exp
                round_bit = 1 if (remain << 1) >= sum_exp else 0
            out_q88[i, j] = quotient + round_bit
    return mat_q88_to_float(out_q88)

# ============================================================
#  MÔ HÌNH FLOAT THUẦN (IEEE 754 - THAM CHIẾU LÝ THUYẾT)
# ============================================================
def softmax_float(x: np.ndarray) -> np.ndarray:
    """Softmax chuẩn IEEE 754 (row-wise, numerically stable)."""
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mhsa_float(X, Wq, Wk, Wv, Wo):
    """
    MHSA tính toán hoàn toàn bằng float64 (IEEE 754).
    Đây là tham chiếu lý thuyết – không có lượng tử hóa, không rounding.
    """
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    heads = []
    for h in range(NH):
        sl  = slice(h * HD, (h + 1) * HD)
        Qh  = Q[:, sl]
        Kh  = K[:, sl]
        Vh  = V[:, sl]
        # Chia cho sqrt(HD) như công thức gốc Attention Is All You Need
        score  = (Qh @ Kh.T) / np.sqrt(HD)
        attn   = softmax_float(score)
        head   = attn @ Vh
        heads.append(head)

    concat = np.concatenate(heads, axis=1)
    return concat @ Wo

# ============================================================
#  MÔ HÌNH Q8.8
# ============================================================
def mhsa_q88(X, Wq, Wk, Wv, Wo):
    Q = matmul_q88(X, Wq)
    K = matmul_q88(X, Wk)
    V = matmul_q88(X, Wv)

    heads = []
    debug_intermediates = {}
    
    for h in range(NH):
        sl = slice(h * HD, (h + 1) * HD)
        Qh = Q[:, sl]
        Kh = K[:, sl]
        Vh = V[:, sl]
        
        score = matmul_q88(Qh, Kh.T)
        score_scaled = scale_score_rtl(score)
        attn  = softmax_q88(score_scaled)
        head  = matmul_q88(attn, Vh)
        heads.append(head)
        
        if h == 0:
            debug_intermediates['Q_head0']       = Qh
            debug_intermediates['K_head0']       = Kh
            debug_intermediates['V_head0']       = Vh
            debug_intermediates['Score_head0']   = score_scaled
            debug_intermediates['Softmax_head0'] = attn

    concat = np.concatenate(heads, axis=1)
    Z = matmul_q88(concat, Wo)
    return Z, debug_intermediates

# ============================================================
#  HÀM XUẤT FILE 
# ============================================================
def load_rtl_output(filepath: str, rows: int, cols: int) -> np.ndarray:
    flat_q88 = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val = int(line, 16)
                flat_q88.append(q88_to_float(val))
    return np.array(flat_q88, dtype=np.float64).reshape((rows, cols))

def save_hex_dump_1d(filepath: str, mat: np.ndarray) -> None:
    mat_q88 = mat_to_q88(mat)
    with open(filepath, 'w', encoding='utf-8') as f:
        for val in mat_q88.flatten():
            f.write(f"{val & 0xFFFF:04x}\n")

# ============================================================
#  PHÂN TÍCH SO SÁNH CHI TIẾT CHO BÁO CÁO
# ============================================================
def print_separator(char="=", width=68):
    print(char * width)

def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB): SNR = 10*log10(var(signal)/var(noise))."""
    var_signal = np.var(signal)
    var_noise  = np.var(noise)
    if var_noise == 0:
        return float('inf')
    return 10.0 * np.log10(var_signal / var_noise)

def compute_enob(snr_db: float) -> float:
    """Effective Number of Bits: ENOB = (SNR - 1.76) / 6.02."""
    return (snr_db - 1.76) / 6.02

def error_histogram(diff_flat: np.ndarray, n_bins: int = 8):
    """In histogram sai số theo khoảng LSB."""
    lsb   = 1.0 / SCALE
    edges = np.linspace(0, diff_flat.max() + lsb, n_bins + 1)
    counts, _ = np.histogram(diff_flat, bins=edges)
    total = len(diff_flat)
    print(f"  {'Khoảng sai số (LSB)':<28}  {'Số phần tử':>10}  {'Tỷ lệ (%)':>10}")
    print("  " + "-" * 54)
    for k in range(n_bins):
        lo = edges[k] * SCALE
        hi = edges[k + 1] * SCALE
        pct = counts[k] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  [{lo:5.2f}, {hi:5.2f}) LSB  {counts[k]:>10d}  {pct:>9.2f}%  {bar}")

def compare_rtl_vs_float(Z_rtl: np.ndarray,
                          Z_q88: np.ndarray,
                          Z_float: np.ndarray) -> None:
    """
    In bảng so sánh 3 chiều:
      (A) RTL Hardware  vs  Float lý thuyết
      (B) RTL Hardware  vs  Python Q8.8 sim
      (C) Python Q8.8   vs  Float lý thuyết

    Các chỉ số: MAE, RMSE, Max Error, SNR, ENOB, % phần tử khớp trong 1 LSB.
    """
    lsb = 1.0 / SCALE   # 1/256 ≈ 0.00390625

    def stats(A: np.ndarray, B: np.ndarray, label_a: str, label_b: str):
        diff      = A - B
        abs_diff  = np.abs(diff)
        mae       = np.mean(abs_diff)
        rmse      = np.sqrt(np.mean(diff ** 2))
        max_err   = abs_diff.max()
        min_err   = abs_diff.min()
        std_err   = abs_diff.std()
        snr       = compute_snr(B, diff)
        enob      = compute_enob(snr) if not np.isinf(snr) else float('inf')
        within_1  = np.mean(abs_diff <= 1 * lsb)  * 100
        within_2  = np.mean(abs_diff <= 2 * lsb)  * 100
        within_4  = np.mean(abs_diff <= 4 * lsb)  * 100
        exact     = np.mean(abs_diff == 0) * 100
        # Số phần tử lớn hơn sai số 1 LSB
        outliers  = int(np.sum(abs_diff > 4 * lsb))

        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  So sánh: [{label_a}]  vs  [{label_b}]")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  Số phần tử phân tích      : {A.size:>10d} ({ROWS}×{COLS})")
        print(f"  │  1 LSB (đơn vị Q8.8)       : {lsb:>14.8f}  (= 1/256)")
        print(f"  ├──────────────── Sai số tuyệt đối (|A-B|) ───────────────┤")
        print(f"  │  MAE  (mean abs error)      : {mae:>14.8f}  ({mae/lsb:.4f} LSB)")
        print(f"  │  RMSE (root mean sq error)  : {rmse:>14.8f}  ({rmse/lsb:.4f} LSB)")
        print(f"  │  Max error                  : {max_err:>14.8f}  ({max_err/lsb:.4f} LSB)")
        print(f"  │  Min error                  : {min_err:>14.8f}  ({min_err/lsb:.4f} LSB)")
        print(f"  │  Std deviation              : {std_err:>14.8f}  ({std_err/lsb:.4f} LSB)")
        print(f"  ├──────────────── Chỉ số chất lượng ─────────────────────┤")
        if np.isinf(snr):
            print(f"  │  SNR                        :          +∞ dB  (khớp hoàn hảo)")
            print(f"  │  ENOB                       :          +∞ bits")
        else:
            print(f"  │  SNR  (signal-to-noise)     : {snr:>11.4f} dB")
            print(f"  │  ENOB (effective # of bits) : {enob:>11.4f} bits")
        print(f"  ├──────────────── Phân phối sai số ───────────────────────┤")
        print(f"  │  Khớp chính xác (err = 0)   : {exact:>10.4f} %")
        print(f"  │  Trong 1 LSB  (≤ 1/256)     : {within_1:>10.4f} %")
        print(f"  │  Trong 2 LSB  (≤ 2/256)     : {within_2:>10.4f} %")
        print(f"  │  Trong 4 LSB  (≤ 4/256)     : {within_4:>10.4f} %")
        print(f"  │  Ngoài 4 LSB  (> 4/256)     : {outliers:>10d} phần tử")
        print(f"  └─────────────────────────────────────────────────────────┘")
        return abs_diff

    print_separator()
    print("  PHÂN TÍCH SO SÁNH ĐA CHIỀU - MA TRẬN ĐẦU RA Z (64×64)")
    print_separator()

    # --- (A) RTL vs Float ---
    diff_A = stats(Z_rtl, Z_float, "RTL Hardware", "Float lý thuyết")
    print(f"\n  [Histogram sai số |RTL − Float|]")
    error_histogram(diff_A.flatten())

    # --- (B) RTL vs Q8.8 Python sim ---
    print()
    diff_B = stats(Z_rtl, Z_q88, "RTL Hardware", "Python Q8.8 sim")

    # --- (C) Q8.8 vs Float ---
    print()
    diff_C = stats(Z_q88, Z_float, "Python Q8.8 sim", "Float lý thuyết")
    print(f"\n  [Histogram sai số |Q8.8 − Float|]")
    error_histogram(diff_C.flatten())

    # ---- Bảng tóm tắt nhanh ----
    lsb = 1.0 / SCALE
    print()
    print_separator("-")
    print("  BẢNG TÓM TẮT (dùng trực tiếp cho báo cáo)")
    print_separator("-")
    header = f"  {'Cặp so sánh':<30}  {'MAE (LSB)':>10}  {'RMSE (LSB)':>11}  {'Max(LSB)':>9}  {'SNR(dB)':>9}  {'ENOB':>6}  {'≤1LSB%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row(label, A, B):
        diff = A - B
        abs_d = np.abs(diff)
        mae   = np.mean(abs_d)
        rmse  = np.sqrt(np.mean(diff**2))
        mx    = abs_d.max()
        snr   = compute_snr(B, diff)
        enob  = compute_enob(snr) if not np.isinf(snr) else float('inf')
        w1    = np.mean(abs_d <= lsb) * 100
        snr_s  = f"{snr:9.2f}" if not np.isinf(snr) else "      +inf"
        enob_s = f"{enob:6.2f}" if not np.isinf(enob) else "   +inf"
        print(f"  {label:<30}  {mae/lsb:>10.4f}  {rmse/lsb:>11.4f}  {mx/lsb:>9.4f}  {snr_s}  {enob_s}  {w1:>6.2f}%")

    row("RTL  vs  Float lý thuyết",  Z_rtl, Z_float)
    row("RTL  vs  Python Q8.8 sim",  Z_rtl, Z_q88)
    row("Q8.8 vs  Float lý thuyết",  Z_q88, Z_float)
    print_separator("-")
    print(f"  * 1 LSB = 1/256 = {lsb:.8f}  |  Q8.8: 8 bit nguyên, 8 bit phân số")

def compare_stages_float(X, Wq, Wk, Wv, Wo, debug_q88: dict) -> None:
    """
    So sánh từng giai đoạn trung gian (Q, K, V, Score, Softmax) của Head 0
    giữa Q8.8 và Float để thấy lỗi tích lũy theo pipeline.
    """
    print()
    print_separator("=")
    print("  SO SÁNH TỪNG GIAI ĐOẠN - HEAD 0 (Q8.8 vs Float lý thuyết)")
    print_separator("=")

    lsb = 1.0 / SCALE

    # Tính float reference cho Head 0
    Q_f   = X @ Wq
    K_f   = X @ Wk
    V_f   = X @ Wv
    sl    = slice(0, HD)
    Qh_f  = Q_f[:, sl]
    Kh_f  = K_f[:, sl]
    Vh_f  = V_f[:, sl]
    score_f = (Qh_f @ Kh_f.T) / np.sqrt(HD)
    attn_f  = softmax_float(score_f)

    stages = [
        ("Q  (head 0)",         debug_q88['Q_head0'],       Qh_f,    ROWS, HD),
        ("K  (head 0)",         debug_q88['K_head0'],       Kh_f,    ROWS, HD),
        ("V  (head 0)",         debug_q88['V_head0'],       Vh_f,    ROWS, HD),
        ("Score scaled (h0)",   debug_q88['Score_head0'],   score_f, ROWS, ROWS),
        ("Softmax (head 0)",    debug_q88['Softmax_head0'], attn_f,  ROWS, ROWS),
    ]

    print(f"\n  {'Giai đoạn':<22}  {'Shape':>10}  {'MAE(LSB)':>9}  {'Max(LSB)':>9}  {'≤1LSB%':>7}  {'SNR(dB)':>9}")
    print("  " + "-" * 76)

    for name, M_q88, M_float, r, c in stages:
        # Resize float về đúng shape (score_f có thể 64x64 thay vì 64x16)
        M_f = M_float[:r, :c] if M_float.shape != (r, c) else M_float
        M_q = M_q88[:r, :c]   if M_q88.shape   != (r, c) else M_q88

        diff  = M_q - M_f
        abs_d = np.abs(diff)
        mae   = np.mean(abs_d)
        mx    = abs_d.max()
        w1    = np.mean(abs_d <= lsb) * 100
        snr   = compute_snr(M_f, diff)
        snr_s = f"{snr:9.2f}" if not np.isinf(snr) else "      +inf"

        shape_str = f"({r}×{c})"
        print(f"  {name:<22}  {shape_str:>10}  {mae/lsb:>9.4f}  {mx/lsb:>9.4f}  {w1:>6.2f}%  {snr_s}")

    print()
    print("  Ghi chú: Sai số tích lũy càng lớn ở các giai đoạn sau")
    print("           là bình thường do bản chất của fixed-point arithmetic.")

# ============================================================
#  HÀM XUẤT FILE ĐIỂM DỮ LIỆU CHO BIỂU ĐỒ BÁO CÁO
# ============================================================
def export_report_csv(BASE: str,
                      Z_rtl: np.ndarray,
                      Z_q88: np.ndarray,
                      Z_float: np.ndarray) -> None:
    """
    Xuất 3 file CSV để vẽ biểu đồ trong báo cáo:
      1. error_rtl_vs_float.csv   – |RTL − Float| theo từng phần tử (flatten)
      2. error_q88_vs_float.csv   – |Q8.8 − Float| theo từng phần tử
      3. scatter_rtl_float.csv    – (Float, RTL) để vẽ scatter plot tương quan
    """
    import csv

    def write_csv(path, header, rows_data):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_data)

    lsb = 1.0 / SCALE

    # 1. Sai số RTL vs Float
    diff_A = np.abs(Z_rtl - Z_float).flatten()
    path1  = os.path.join(BASE, "report_error_rtl_vs_float.csv")
    write_csv(path1,
              ["index", "abs_error_float", "abs_error_lsb"],
              [(i, f"{diff_A[i]:.8f}", f"{diff_A[i]/lsb:.6f}") for i in range(len(diff_A))])

    # 2. Sai số Q8.8 vs Float
    diff_C = np.abs(Z_q88 - Z_float).flatten()
    path2  = os.path.join(BASE, "report_error_q88_vs_float.csv")
    write_csv(path2,
              ["index", "abs_error_float", "abs_error_lsb"],
              [(i, f"{diff_C[i]:.8f}", f"{diff_C[i]/lsb:.6f}") for i in range(len(diff_C))])

    # 3. Scatter RTL vs Float (lấy mẫu 512 điểm để biểu đồ gọn)
    flat_rtl   = Z_rtl.flatten()
    flat_float = Z_float.flatten()
    step       = max(1, len(flat_rtl) // 512)
    path3      = os.path.join(BASE, "report_scatter_rtl_float.csv")
    write_csv(path3,
              ["float_reference", "rtl_hardware"],
              [(f"{flat_float[i]:.8f}", f"{flat_rtl[i]:.8f}")
               for i in range(0, len(flat_rtl), step)])

    print(f"\n  [CSV] Đã xuất 3 file dữ liệu cho biểu đồ báo cáo:")
    print(f"        → {path1}")
    print(f"        → {path2}")
    print(f"        → {path3}")

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print_separator()
    print("  GOLDEN MODEL - MHSA  |  D=64  NH=4  HD=16  Q8.8")
    print_separator()

    BASE = "C:/Users/Dell/Downloads/VSCode/MHSA"

    print("\n[1] Đọc các file input...")
    try:
        X  = load_hex_rowmajor(os.path.join(BASE, "matrix_X.hex"),  ROWS, COLS)
        Wq = load_mif_rowmajor(os.path.join(BASE, "weights_Wq.mif"), ROWS, COLS)
        Wk = load_mif_rowmajor(os.path.join(BASE, "weights_Wk.mif"), ROWS, COLS)
        Wv = load_mif_rowmajor(os.path.join(BASE, "weights_Wv.mif"), ROWS, COLS)
        Wo = load_mif_rowmajor(os.path.join(BASE, "weights_Wo.mif"), ROWS, COLS)
    except FileNotFoundError as e:
        print(f"\n  [LỖI] Không tìm thấy file: {e}")
        sys.exit(1)

    print("\n[2] Tính toán mô hình Q8.8 (bit-accurate, đồng bộ RTL)...")
    Z_q88, debug = mhsa_q88(X, Wq, Wk, Wv, Wo)

    print("\n[3] Tính toán mô hình Float lý thuyết (IEEE 754, tham chiếu)...")
    Z_float = mhsa_float(X, Wq, Wk, Wv, Wo)

    # Xuất file debug
    save_hex_dump_1d(os.path.join(BASE, "py_debug_Q_head0.txt"),       debug['Q_head0'])
    save_hex_dump_1d(os.path.join(BASE, "py_debug_K_head0.txt"),       debug['K_head0'])
    save_hex_dump_1d(os.path.join(BASE, "py_debug_V_head0.txt"),       debug['V_head0'])
    save_hex_dump_1d(os.path.join(BASE, "py_debug_Score_head0.txt"),   debug['Score_head0'])
    save_hex_dump_1d(os.path.join(BASE, "py_debug_Softmax_head0.txt"), debug['Softmax_head0'])
    save_hex_dump_1d(os.path.join(BASE, "Z_golden_q88.txt"), Z_q88)
    print("  -> Đã xuất các file debug trung gian.")

    # ---- So sánh pipeline từng giai đoạn ----
    compare_stages_float(X, Wq, Wk, Wv, Wo, debug)

    # ---- So sánh 3 chiều RTL / Q8.8 / Float ----
    rtl_path = os.path.join(BASE, "matrix_Z_out.txt")
    if os.path.exists(rtl_path):
        print(f"\n[4] Tải kết quả RTL từ '{rtl_path}'...")
        Z_rtl = load_rtl_output(rtl_path, ROWS, COLS)

        compare_rtl_vs_float(Z_rtl, Z_q88, Z_float)

        # Xuất CSV cho biểu đồ báo cáo
        export_report_csv(BASE, Z_rtl, Z_q88, Z_float)

        # Kết luận cuối
        lsb = 1.0 / SCALE
        mae_hw_float = np.mean(np.abs(Z_rtl - Z_float))
        mae_hw_q88   = np.mean(np.abs(Z_rtl - Z_q88))

        print()
        print_separator()
        print("  KẾT LUẬN")
        print_separator()
        if mae_hw_q88 < lsb:
            print(f"  ✔  RTL khớp với Python Q8.8 sim  (MAE = {mae_hw_q88:.8f} < 1 LSB)")
        else:
            print(f"  ✘  RTL lệch với Q8.8 sim  (MAE = {mae_hw_q88:.8f})")

        print(f"  ➜  Sai số lượng tử hóa RTL vs Float : MAE = {mae_hw_float:.6f}  "
              f"({mae_hw_float/lsb:.3f} LSB)")
        print(f"     Đây là sai số cố hữu của biểu diễn Q8.8 (fixed-point).")
        print(f"     Giá trị nhỏ hơn 1 LSB ({lsb:.6f}) là chấp nhận được.")
        print_separator()
    else:
        print("\n[4] Không tìm thấy file RTL. Chạy ModelSim để tạo matrix_Z_out.txt.")
        print("    Hiển thị so sánh Q8.8 vs Float (không có RTL):")

        print()
        diff_qf = np.abs(Z_q88 - Z_float)
        lsb = 1.0 / SCALE
        print(f"  Q8.8 vs Float — MAE  : {np.mean(diff_qf):.8f}  ({np.mean(diff_qf)/lsb:.4f} LSB)")
        print(f"  Q8.8 vs Float — RMSE : {np.sqrt(np.mean((Z_q88-Z_float)**2)):.8f}")
        print(f"  Q8.8 vs Float — Max  : {diff_qf.max():.8f}  ({diff_qf.max()/lsb:.4f} LSB)")
        compare_stages_float(X, Wq, Wk, Wv, Wo, debug)
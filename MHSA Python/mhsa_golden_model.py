import numpy as np
import math

# ==========================================
# CẤU HÌNH THÔNG SỐ
# ==========================================
ROWS = 64
COLS = 64
HEADS = 4
HEAD_DIM = COLS // HEADS  # 16

# ==========================================
# CÁC HÀM TIỆN ÍCH MÔ PHỎNG PHẦN CỨNG (RTL EMULATION)
# ==========================================

def to_signed_16(val):
    """Mô phỏng phép gán vào thanh ghi logic signed [15:0] (Cắt bit [15:0])"""
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val

def float_matrix_to_q88(mat):
    """Chuyển ma trận số thực sang ma trận số nguyên Q8.8 (Có kẹp giới hạn -32768 đến 32767)"""
    q88_mat = np.zeros_like(mat, dtype=np.int64)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(round(mat[i, j] * 256.0))
            val = max(min(val, 32767), -32768)
            q88_mat[i, j] = val
    return q88_mat

def rtl_mac_q88(A, B):
    """
    Mô phỏng khối mac_unit.sv
    - Cộng dồn vào accumulator 40-bit
    - Cộng 128 (0x80) để làm tròn
    - Lấy bit [23:8] làm kết quả ngõ ra
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.int64)
    
    for i in range(M):
        for j in range(N):
            acc_40 = 0
            for k in range(K):
                # full_product (32-bit)
                full_product = A[i, k] * B[k, j] 
                acc_40 += full_product
            
            # Làm tròn và cắt bit: acc_rounded = acc_40 + 0x80; mac_out = acc_rounded[23:8]
            acc_rounded = acc_40 + 128
            mac_out = acc_rounded >> 8
            
            C[i, j] = to_signed_16(mac_out)
    return C

def rtl_scale_score(score):
    """
    Mô phỏng phép Scale trong attention_head.sv
    $signed({ {2{mac1_result[15]}}, mac1_result[15:2] }) + 16'(mac1_result[1])
    """
    M, N = score.shape
    out = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            val = score[i, j]
            round_bit = (val >> 1) & 1
            scaled = (val >> 2) + round_bit
            out[i, j] = to_signed_16(scaled)
    return out

def rtl_exp_lut(x_norm):
    """Mô phỏng ROM chứa e^x"""
    addr_raw = x_norm + 4096
    
    # Kẹp địa chỉ (Clamping)
    if addr_raw < 0: addr = 0
    elif addr_raw > 4096: addr = 4096
    else: addr = addr_raw
    
    # Bảng tra LUT (Giống file tạo .mif)
    x_real = (addr - 4096) / 256.0
    exp_real = math.exp(x_real)
    exp_q88 = int(round(exp_real * 256))
    
    # 16-bit Unsigned
    return max(min(exp_q88, 65535), 0)

def load_hex_or_mif(filename, rows=64, cols=64):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Loại bỏ khoảng trắng và các ký tự đặc biệt
            line = line.strip()
            # Bỏ qua các dòng header của MIF
            if not line or line.startswith(('DEPTH', 'WIDTH', 'ADDRESS_RADIX', 'DATA_RADIX', 'CONTENT', 'BEGIN', 'END', '-')):
                continue
            
            # Xử lý dòng dạng "addr : data;" trong file MIF hoặc chỉ "data" trong HEX
            if ':' in line:
                val_str = line.split(':')[1].replace(';', '').strip()
            else:
                val_str = line
            
            # Chuyển hex 16-bit sang số nguyên có dấu (Bù 2)
            val = int(val_str, 16)
            if val >= 0x8000:
                val -= 0x10000
            data.append(val)
    
    # Chuyển về ma trận numpy (dùng mảng phẳng rồi reshape)
    matrix = np.array(data[:rows*cols]).reshape(rows, cols)
    return matrix

def rtl_softmax(score):
    """
    Mô phỏng khối softmax_top.sv
    """
    M, N = score.shape
    out = np.zeros((M, N), dtype=np.int64)
    
    for i in range(M):
        # 1. Tìm Max của hàng (Pha 1)
        max_val = np.max(score[i, :])
        
        # 2. Tính Tổng Mẫu Số (Pha 2)
        exp_vals = np.zeros(N, dtype=np.int64)
        sum_exp = 0
        for j in range(N):
            x_norm = score[i, j] - max_val
            exp_vals[j] = rtl_exp_lut(x_norm)
            sum_exp += exp_vals[j]
            
        # 3. Phép Chia (Pass 3)
        for j in range(N):
            numer = exp_vals[j] << 8  # Dịch 8 bit (Q8.8)
            denom = sum_exp
            
            if denom == 0:
                quotient, remain = 0, 0
            else:
                quotient = numer // denom
                remain = numer % denom
            
            # Làm tròn: Nếu (số dư * 2) >= mẫu số thì cộng 1
            if (remain << 1) >= denom:
                quotient += 1
                
            out[i, j] = to_signed_16(quotient)
    return out

# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MHSA EXECUTION)
# ==========================================
if __name__ == "__main__":
    print("Khởi tạo ma trận (Seed = 42)...")
    np.random.seed(42)
    min_val, max_val = -0.2, 0.2
    
    print("Đang nạp trọng số từ file...")
    Wq = load_hex_or_mif("weights_Wq.mif") # hoặc file .hex tương ứng
    Wk = load_hex_or_mif("weights_Wk.mif")
    Wv = load_hex_or_mif("weights_Wv.mif")
    Wo = load_hex_or_mif("weights_Wo.mif")
    X  = load_hex_or_mif("matrix_X.hex")


    print("Đang chạy Phase 1: Linear Q, K, V...")
    Q = rtl_mac_q88(X, Wq)
    K = rtl_mac_q88(X, Wk)
    V = rtl_mac_q88(X, Wv)

    Z_heads = []
    
    print("Đang chạy Phase 2: 4 Attention Heads...")
    for h in range(HEADS):
        # Tách Head (Slicing) - Lấy dải cột tương ứng
        start_col = h * HEAD_DIM
        end_col   = start_col + HEAD_DIM
        
        Q_h = Q[:, start_col:end_col]
        K_h = K[:, start_col:end_col]
        V_h = V[:, start_col:end_col]
        
        # Score = Q * K^T
        Score_h = rtl_mac_q88(Q_h, K_h.T)
        
        # Scale (chia 4 làm tròn)
        Score_scaled = rtl_scale_score(Score_h)
        
        # Softmax
        S_h = rtl_softmax(Score_scaled)
        
        # Z_h = Softmax * V
        Z_h = rtl_mac_q88(S_h, V_h)
        
        Z_heads.append(Z_h)
        print(f"  -> Head {h} hoàn tất.")

    print("Đang chạy Phase 3: Linear Output...")
    # Ghép nối 4 Head (Concatenation On-the-fly)
    Z_concat = np.concatenate(Z_heads, axis=1)
    
    # Kết quả cuối cùng
    Z_final = rtl_mac_q88(Z_concat, Wo)

    # Xuất ra file
    output_filename = "matrix_Z_python_out.txt"
    with open(output_filename, "w") as f:
        for val in Z_final.flatten():
            # Chuyển sang mã Hex 16-bit bù 2
            hex_str = f"{val & 0xFFFF:04X}"
            f.write(hex_str + "\n")
            
    print(f"XONG! Đã lưu ma trận Z vào file '{output_filename}'")
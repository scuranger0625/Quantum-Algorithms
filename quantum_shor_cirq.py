# -*- coding: utf-8 -*-
"""
Shor 演算法（教學級模擬）— Cirq + NumPy + pandas + matplotlib
---------------------------------------------------------------
重點：
1) 用 QPE 找到乘法運算 U_a: |y> -> |a*y mod N> 的本徵相位 φ = s/r
2) 由測得的 φ 以連分數還原 r
3) 若 r 為偶數且 a^{r/2} != -1 (mod N)，以 gcd(a^{r/2} ± 1, N) 取得 N 的非平凡因子

說明：
- 我們用 Cirq.LinearPermutationGate 建立「乘以 a (mod N)」的可逆置換，並把 >=N 的狀態定義為恆等，以保持整體是置換（可逆）。
- 第一暫存器（計數暫存器）做 QPE：|+>^{\otimes m} -> 受控 U^{2^k} -> 反QFT -> 測量得到 s
- 第二暫存器（工作暫存器）初始化為 |1>，扮演 U 的作用域。

套件需求：
pip install cirq pandas matplotlib numpy
"""

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cirq


# ---------------------------
# 一些數學小工具
# ---------------------------
def continued_fraction_denominator(s: int, Q: int, max_den: int) -> int:
    """
    將 s/Q ≈ t/r，以連分數收斂子(convergents)找分母 r（<= max_den）。
    回傳候選 r（找不到則回傳 0）。
    """
    # 連分數展開
    a = []
    num, den = s, Q
    while den != 0:
        a0 = num // den
        a.append(a0)
        num, den = den, num - a0 * den

    # 由係數 a 重建收斂子，逐一檢查
    p_m2, p_m1 = 0, 1
    q_m2, q_m1 = 1, 0
    for ai in a:
        p = ai * p_m1 + p_m2
        q = ai * q_m1 + q_m2
        p_m2, p_m1 = p_m1, p
        q_m2, q_m1 = q_m1, q
        if 0 < q <= max_den:
            # 誤差門檻可視需要調整
            if abs(s / Q - p / q) < 1 / (2 * Q):
                return q
    return 0


def gcd(a, b):
    return math.gcd(a, b)


# ---------------------------
# 反 QFT （在計數暫存器上）
# ---------------------------
def inverse_qft(qubits):
    """
    在 qubits（小端序）上構建反QFT電路。
    注意 Cirq 預設 qubit 順序你可自行掌握；這裡以 qubits[0] 是最低位。
    """
    circuit = cirq.Circuit()
    n = len(qubits)
    for j in range(n // 2):
        # 量子位反轉，讓位序與 QPE 標準推導一致（可視實作調整）
        circuit.append(cirq.SWAP(qubits[j], qubits[n - j - 1]))

    # 逐位做反QFT：先做 Hadamard，然後加上往前的受控相位（負相位）
    for k in range(n):
        qk = qubits[k]
        # 對 qk 做 Hadamard 之前，先處理跟更高位之間的受控相位
        for j in range(k):
            # 角度：-π / 2^{k-j}
            theta = -np.pi / (2 ** (k - j))
            circuit.append(cirq.CZPowGate(exponent=theta/np.pi).on(qubits[j], qk))
        circuit.append(cirq.H(qk))
    return circuit

class ModularMultiplyGate(cirq.Gate):
    """以 permutation 矩陣實作：|y> ↦ |(a*y) mod N>，對 y>=N 則保持恆等。"""
    def __init__(self, a: int, N: int, n_qubits: int):
        self.a = a
        self.N = N
        self.n_qubits = n_qubits

    def _num_qubits_(self) -> int:
        return self.n_qubits

    def _unitary_(self):
        # 產生 2^n × 2^n 的 permutation 矩陣
        dim = 1 << self.n_qubits
        U = np.eye(dim, dtype=np.complex128)  # 先設為 I，之後覆寫 < N 的欄位
        for y in range(self.N):               # 只對 0..N-1 做 (a*y) mod N
            dest = (self.a * y) % self.N
            # 把第 y 欄的 1 從 (y,y) 移到 (dest, y)
            U[y, y] = 0.0
            U[dest, y] = 1.0
        # 其餘 y>=N 維持 I（確保整體是單位ary）
        return U

    def _circuit_diagram_info_(self, args):
        return f"×{self.a} mod {self.N}"

def modular_multiply_gate(a: int, N: int, n_target_qubits: int) -> cirq.Gate:
    """回傳自訂的模乘 Gate。"""
    return ModularMultiplyGate(a, N, n_target_qubits)

def controlled_powered_U(a: int, N: int, n_target_qubits: int,
                         power: int, control: cirq.Qid, targets: list[cirq.Qid]):
    """
    受控 U_a^{power}。
    這裡用重複套用 U 的方式（對小 N 夠用；真實大 N 會做快速冪與可逆加法器）。
    """
    U = modular_multiply_gate(a, N, n_target_qubits)
    ops = []
    for _ in range(power):
        ops.append(U.on(*targets).controlled_by(control))
    return ops

# ---------------------------
# QPE：用來估相位 φ=s/r
# ---------------------------
def qpe_for_order(a: int, N: int, m_count: int, simulator_seed: int = None):
    """
    以 QPE 估計 U_a 的本徵相位 φ，回傳：
    - 測量到的整數 s（0..2^m-1）
    - 用到的 Q=2^m
    - 以及全電路（可視化用）
    """
    if simulator_seed is not None:
        np.random.seed(simulator_seed)
        random.seed(simulator_seed)

    # 目標暫存器大小：最少要能表示 0..N-1
    n_work = math.ceil(math.log2(N))
    count_qubits = cirq.LineQubit.range(m_count)            # 計數暫存器（QPE）
    work_qubits = cirq.LineQubit.range(m_count, m_count + n_work)  # 工作暫存器（U 的作用域）

    circuit = cirq.Circuit()

    # 1) 初始化：計數暫存器到 |+>^m；工作暫存器到 |1>
    circuit.append(cirq.H.on_each(*count_qubits))
    # 將工作暫存器準備成 |1>（小端序）：在最低位上 X
    circuit.append(cirq.X(work_qubits[0]))

    # 2) 受控 U^{2^k}
    Q = 1 << m_count
    for k, ctrl in enumerate(count_qubits):
        power = 1 << (m_count - 1 - k)   # 最高位先施加最大的冪次（與我們的 inverse_qft 的位序匹配）
        ops = controlled_powered_U(a, N, n_work, power, ctrl, work_qubits)
        circuit.append(ops)

    # 3) 在計數暫存器上做「反 QFT」
    circuit.append(inverse_qft(count_qubits))

    # 4) 測量計數暫存器
    circuit.append(cirq.measure(*count_qubits, key='result'))

    # 執行模擬
    sim = cirq.Simulator(seed=simulator_seed)
    result = sim.run(circuit, repetitions=1)
    bits = result.measurements['result'][0]  # 形如 array([b_{m-1},...,b_0]) 依我們組線而定
    # bits 目前是高位在前（因我們做了位序翻轉 SWAP）；組合成整數 s
    s = 0
    for bit in bits:  # 逐位累進
        s = (s << 1) | int(bit)

    return s, Q, circuit


# ---------------------------
# Shor 主流程（單次嘗試）
# ---------------------------
def shor_try_once(N: int, m_count: int, a: int | None = None, seed: int | None = None):
    """
    針對單一 N 跑一次完整流程：
    1) 隨機挑 a（或使用給定的 a），若 gcd(a,N) != 1 直接得到因子
    2) 用 QPE 估 φ -> 得 s -> 連分數 -> r
    3) 驗證 r 並用 gcd(a^{r/2}±1, N) 拿因子
    回傳字典（方便 pandas 讀取）
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if N % 2 == 0:
        return dict(success=True, p=2, q=N // 2, a=None, r=None, s=None, Q=None, note="N is even.")

    # 1) 選 a
    if a is None:
        a = random.randrange(2, N - 1)
    g = gcd(a, N)
    if g != 1:
        # 撿到因子
        return dict(success=True, p=g, q=N // g, a=a, r=None, s=None, Q=None, note="gcd(a,N)!=1")

    # 2) QPE 估相位
    s, Q, circuit = qpe_for_order(a, N, m_count, simulator_seed=seed)

    # 3) 連分數 -> r
    r = continued_fraction_denominator(s, Q, max_den=N)
    if r == 0:
        return dict(success=False, p=None, q=None, a=a, r=None, s=s, Q=Q, note="CF failed")

    # 驗證：a^r ≡ 1 (mod N)
    if pow(a, r, N) != 1:
        return dict(success=False, p=None, q=None, a=a, r=r, s=s, Q=Q, note="a^r != 1 (mod N)")

    # r 需為偶數且 a^{r/2} != -1 (mod N)
    if r % 2 == 1:
        return dict(success=False, p=None, q=None, a=a, r=r, s=s, Q=Q, note="r is odd")

    ar2 = pow(a, r // 2, N)
    if ar2 == N - 1:
        return dict(success=False, p=None, q=None, a=a, r=r, s=s, Q=Q, note="a^(r/2) == -1 mod N")

    # 4) 取因子
    p = gcd(ar2 - 1, N)
    q = gcd(ar2 + 1, N)
    if p * q == N and p not in (1, N) and q not in (1, N):
        return dict(success=True, p=p, q=q, a=a, r=r, s=s, Q=Q, note="OK")
    else:
        return dict(success=False, p=None, q=None, a=a, r=r, s=s, Q=Q, note="gcd step failed")


# ---------------------------
# 主程式：跑多次、統計、畫圖
# ---------------------------
if __name__ == "__main__":
    # 教學案例：N=15；m_count（計數暫存器 qubits 數）建議 5~8（越大解析度越高但耗時增加）
    N = 15
    m_count = 6        # Q = 64
    trials = 50        # 試跑次數（你可以加大，看直方圖更漂亮）
    seed_base = 42

    rows = []
    for t in range(trials):
        row = shor_try_once(N=N, m_count=m_count, a=None, seed=seed_base + t)
        row['trial'] = t
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        'trial', 'success', 'p', 'q', 'a', 'r', 's', 'Q', 'note'
    ])
    print(df)

    # 成功率
    succ_rate = df['success'].mean()
    print(f"\n成功率：{succ_rate:.2%}")

    # 繪圖：r 的出現次數分佈（只看成功或 r 有抓到的記錄）
    df_r = df.dropna(subset=['r'])
    plt.figure(figsize=(7, 4))
    df_r['r'].value_counts().sort_index().plot(kind='bar')
    plt.title(f"Estimated order r distribution (N={N}, trials={trials})")
    plt.xlabel("r")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # 顯示最常見的 r 與常見的 a
    print("\n常見 r：")
    print(df_r['r'].value_counts().sort_index())

    print("\n常見 a（含 gcd 撿到因子的情況）：")
    print(df['a'].value_counts(dropna=False).head(10))

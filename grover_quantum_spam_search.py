import cirq
import pandas as pd
import matplotlib.pyplot as plt

# === Step 1. 定義資料集 ===
df = pd.DataFrame({
    'ID':   ['000','001','010','011','100','101','110','111'],
    'Subject': ["Meeting notes", "Your invoice", "Free coupon", "Your bank alert",
                "Discount offer", "Lottery winner!", "Job offer", "Spam?"],
    'IsSpam': [0,0,1,0,0,0,0,0]
})

print("資料集如下：")
print(df)

# === Step 2. 自動找出唯一 spam 的 target ID ===
target_id = df.loc[df['IsSpam']==1, 'ID'].values[0]
print(f"\n要搜尋的 spam 信 ID: {target_id}")

# === Step 3. Grover 電路設計 ===

# 3 qubit 對應 8 筆資料
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit()

# 疊加初始狀態
circuit.append([cirq.H(q) for q in qubits])

# --- Oracle（只標記 target_id 為 -1，其餘不變）---
def grover_oracle(target):
    # 讓 target_id 映射到 |111>
    ops = []
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            ops.append(cirq.X(qubits[i]))
    # 對 |111> 做 Z gate
    ops.append(cirq.Z(qubits[2]).controlled_by(qubits[0], qubits[1]))
    # 還原 X
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            ops.append(cirq.X(qubits[i]))
    return ops

# --- Diffuser（反射）---
def grover_diffuser():
    ops = []
    # 全 H
    ops += [cirq.H(q) for q in qubits]
    # 全 X
    ops += [cirq.X(q) for q in qubits]
    # 控制-Z
    ops.append(cirq.Z(qubits[2]).controlled_by(qubits[0], qubits[1]))
    # 全 X
    ops += [cirq.X(q) for q in qubits]
    # 全 H
    ops += [cirq.H(q) for q in qubits]
    return ops

# 執行次數 √N ≈ 2（這裡只做 1 輪，3 qubit 剛好足夠）
circuit.append(grover_oracle(target_id))
circuit.append(grover_diffuser())

# 測量
circuit.append([cirq.measure(q, key=f'q{i}') for i, q in enumerate(qubits)])

print("\nGrover 電路：")
print(circuit)

# === Step 4. 模擬執行 ===
sim = cirq.Simulator()
result = sim.run(circuit, repetitions=100)

# 統計結果
hist = dict(result.multi_measurement_histogram(keys=['q0', 'q1', 'q2']))
labels = ['{0:03b}'.format(k[0]*4 + k[1]*2 + k[2]) for k in hist.keys()]
values = list(hist.values())

# 視覺化
plt.bar(labels, values)
plt.xlabel("Result (q0 q1 q2)")
plt.ylabel("Frequency")
plt.title("Grover's Result")
plt.show()

# 印出最常出現的ID與對應信件主題
max_idx = labels[values.index(max(values))]
subject = df.loc[df['ID']==max_idx, 'Subject'].values[0]
print(f"最有可能的 spam 信 ID：{max_idx}，主題：{subject}")
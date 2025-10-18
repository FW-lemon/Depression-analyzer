import pandas as pd
import numpy as np

# 读取数据
file_path = "/home/project/yihao/total_data_04.csv"
df = pd.read_csv(file_path)

# 1️⃣ 删除不需要的列
df = df.drop(columns=["KIQ029", "KIQ481"])

# 2️⃣ dim1_ckd 和 dim2_kidney_stone 随机生成 0/1 替代
np.random.seed(42)
df["dim1_ckd"] = np.random.randint(0, 2, size=len(df))
df["dim2_kidney_stone"] = np.random.randint(0, 2, size=len(df))

# 3️⃣ 对 dim3_urinary_incontinence 和 dim4_impact_nocturia 做 0-1 Min-Max 缩放，并保留两位小数
for col in ["dim3_urinary_incontinence", "dim4_impact_nocturia"]:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = ((df[col] - min_val) / (max_val - min_val)).round(2)

# 保存处理后的文件
output_path = "/home/project/yihao/total_data_05.csv"
df.to_csv(output_path, index=False)

print(f"✅ 数据处理完成，已保存到 {output_path}")
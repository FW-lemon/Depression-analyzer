import pandas as pd

input_file = "/home/project/yihao/total_data_03.csv"
output_file = "/home/project/yihao/total_data_04.csv"

df = pd.read_csv(input_file)

# ---- 定义维度列 ----
dim1_cols = ["KIQ022","KIQ025"]  # 慢性肾病/透析
dim2_cols = ["KIQ026","KID028"]  # 肾结石
dim3_cols = ["KIQ005","KIQ010","KIQ042","KIQ430","KIQ044","KIQ450","KIQ046","KIQ470"]
dim4_cols = ["KIQ050","KIQ052","KIQ480"]

# ---- 过滤存在的列 ----
def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

# ---- 归一化计算函数 ----
def normalize_cols(df, cols):
    cols_exist = existing_cols(df, cols)
    if not cols_exist:
        return pd.Series([0]*len(df))
    # 每列归一化到0-1
    df_norm = df[cols_exist].div(df[cols_exist].max())
    return df_norm.mean(axis=1)

# ---- 计算四个维度 ----
df["dim1_ckd"] = normalize_cols(df, dim1_cols)  # 慢性肾病/透析
df["dim2_kidney_stone"] = normalize_cols(df, dim2_cols)  # 肾结石
df["dim3_urinary_incontinence"] = normalize_cols(df, dim3_cols)  # 尿失禁类型/频率
df["dim4_impact_nocturia"] = normalize_cols(df, dim4_cols)  # 尿失禁影响/夜尿

# ---- 保留两位小数 ----
df[["dim1_ckd","dim2_kidney_stone","dim3_urinary_incontinence","dim4_impact_nocturia"]] = \
    df[["dim1_ckd","dim2_kidney_stone","dim3_urinary_incontinence","dim4_impact_nocturia"]].round(2)

# ---- 删除原 KIQ 列 ----
all_kiq_cols = dim1_cols + dim2_cols + dim3_cols + dim4_cols
df.drop(columns=existing_cols(df, all_kiq_cols), inplace=True)

# ---- 保存最终文件 ----
df.to_csv(output_file, index=False)
print(f"✅ 已生成归一化四维数据并保存至 {output_file}")

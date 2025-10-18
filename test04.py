import pandas as pd

# DPQ 列名，包括 DPQ100
DPQ_COLS = [f"DPQ0{i}" for i in range(10, 100, 10)] + ["DPQ100"]

def assign_label(score):
    """根据 PHQ-9 总分生成抑郁程度标签"""
    if score <= 4:
        return 0  # 无抑郁
    elif score <= 9:
        return 1  # 轻度
    elif score <= 14:
        return 2  # 中度
    elif score <= 19:
        return 3  # 中重度
    else:
        return 4  # 重度

def process_total_data(file_path, output_path=None):
    """处理 total_data_02 文件，只保留 KIQ 列 + PHQ-9 总分和标签"""
    df = pd.read_csv(file_path)

    # 筛选存在的 DPQ 列
    available_dpq_cols = [c for c in DPQ_COLS if c in df.columns]

    # 计算 PHQ-9 总分
    for col in available_dpq_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phq9_score"] = df[available_dpq_cols].sum(axis=1, skipna=True)

    # 打标签
    df["depression_label"] = df["phq9_score"].apply(assign_label)

    # 删除 DPQ 原始列，包括 DPQ100
    df = df.drop(columns=available_dpq_cols)

    # 保存结果
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ 已保存带标签文件: {output_path}")

    return df

# 使用示例
input_file = "total_data_02.csv"
output_file = "total_data_03.csv"

df_labeled = process_total_data(input_file, output_file)

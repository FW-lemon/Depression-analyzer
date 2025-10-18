import os
import pandas as pd

# PHQ-9 题目列
DPQ_COLS = [f"DPQ0{i}" for i in range(10, 100, 10)]  # DPQ010, DPQ020, ... DPQ090

def assign_label(score):
    """根据PHQ-9总分生成抑郁程度label"""
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

def process_file(file_path):
    """读取CSV，计算PHQ-9分数并打标签"""
    df = pd.read_csv(file_path)

    # 只保留DPQ相关列中存在的
    available_cols = [c for c in DPQ_COLS if c in df.columns]

    if not available_cols:
        print(f"⚠️ {file_path} 中没有找到 DPQ 相关列，跳过")
        return None

    # 将非数值转成NaN
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算总分
    df["phq9_score"] = df[available_cols].sum(axis=1, skipna=True)

    # 打标签
    df["depression_label"] = df["phq9_score"].apply(assign_label)

    return df

def main():
    input_dir = "/home/project/yihao/total_data"
    output_dir = "/home/project/yihao/total_data"

    for file in os.listdir(input_dir):
        if file.endswith("_total.csv") and not file.endswith(":Zone.Identifier"):
            file_path = os.path.join(input_dir, file)
            print(f"处理文件: {file}")

            df = process_file(file_path)
            if df is not None:
                output_path = os.path.join(output_dir, file)
                df.to_csv(output_path, index=False)
                print(f"✅ 已保存带标签的文件: {output_path}")

if __name__ == "__main__":
    main()

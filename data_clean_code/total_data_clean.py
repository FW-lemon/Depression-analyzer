import pandas as pd
import glob
from functools import reduce
import os

def process_year_folder(input_dir, threshold_ratio=0.7):
    """
    清洗并合并一个年份文件夹下的所有 *_clean.csv 文件：
    - 删除列名里包含 'missing' 的列
    - 只保留数值列
    - 删除没有达到 threshold_ratio 的 SEQN
    - 填补空值为 0
    - 输出文件命名为 {年份}_total.csv
    """
    files = glob.glob(f"{input_dir}/*_clean.csv")
    if not files:
        print(f"{input_dir} 下没有找到 *_clean.csv 文件，跳过")
        return

    print(f"处理文件夹 {input_dir}, 找到 {len(files)} 个文件")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()  # 去掉列名首尾空格
        if "SEQN" not in df.columns:
            raise ValueError(f"文件 {f} 缺少 SEQN 列")
        # 确保 SEQN 在最前面
        cols = ["SEQN"] + [c for c in df.columns if c != "SEQN"]
        df = df[cols]
        dfs.append(df)

    # 统计每个 SEQN 出现的次数
    seqn_counts = pd.concat([df[["SEQN"]] for df in dfs]).value_counts().reset_index()
    seqn_counts.columns = ["SEQN", "count"]

    # 至少出现在 threshold_ratio 比例的文件里
    threshold = int(threshold_ratio * len(files))
    valid_seqn = seqn_counts[seqn_counts["count"] >= threshold]["SEQN"].tolist()
    print(f"阈值: 至少出现在 {threshold}/{len(files)} 个文件")
    print(f"满足条件的 SEQN 数量: {len(valid_seqn)}")

    # 过滤每个 df
    filtered_dfs = [df[df["SEQN"].isin(valid_seqn)] for df in dfs]

    # outer 合并
    merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), filtered_dfs)
    print(f"最终合并后的形状: {merged.shape}")

    # 只保留数值列，并排除包含 'missing' 的列
    numeric_cols = [c for c in merged.select_dtypes(include=["number"]).columns if "missing" not in c.lower()]
    numeric_df = merged[numeric_cols]

    # 填补空值为 0
    filled_df = numeric_df.fillna(0)

    # 保存最终结果
    year_prefix = input_dir.split("-")[0]  # 文件夹名开头年份
    output_file = f"{year_prefix}_total.csv"
    filled_df.to_csv(output_file, index=False)
    print(f"{input_dir} 清洗完成，结果已保存到 {output_file}, 形状: {filled_df.shape}\n")


if __name__ == "__main__":
    # 所有年份文件夹列表，例如 ["2005-2006", "2007-2008", ...]
    year_folders = sorted([d for d in os.listdir(".") if os.path.isdir(d)])
    for folder in year_folders:
        process_year_folder(folder)

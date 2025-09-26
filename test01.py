import pandas as pd

def clean_features(input_file, output_file):
    # 读取原始数据
    df = pd.read_csv(input_file)

    # 找到要删除的列
    cols_to_drop = []

    for col in df.columns:
        # 删除列名里有 consistency 的
        if "consistency" in col.lower():
            cols_to_drop.append(col)
        # 删除列名里有 missing 或 all_ 的
        elif "missing" in col.lower() or "all_" in col.lower():
            cols_to_drop.append(col)
        # 删除列值全是 'valid'/'invalid' 的
        elif df[col].dropna().isin(['valid', 'invalid', 'invalid_no_pregnancy', 'invalid_non_diabetic']).all():
            cols_to_drop.append(col)

    print(f"删除 {len(cols_to_drop)} 个列，例如: {cols_to_drop[:10]} ...")

    # 删除列
    df_cleaned = df.drop(columns=cols_to_drop)

    # 保存结果
    df_cleaned.to_csv(output_file, index=False)
    print(f"清理后的数据已保存到 {output_file}, 当前形状: {df_cleaned.shape}")

if __name__ == "__main__":
    input_file = "/home/project/yihao/2005-2006_merged.csv"      # 替换为你的合并大表路径
    output_file = "merged_cleaned.csv"  # 输出清理后的表
    clean_features(input_file, output_file)

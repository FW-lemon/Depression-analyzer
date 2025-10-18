import pandas as pd
import numpy as np
import os
import re

# =========================
# DPQ 清洗函数
# =========================
def clean_dpq_simple(df):
    """
    清洗 DPQ 数据：
    - 异常值 5.397605346934028e-79 替换为 NaN
    - 只保留有效值 {0,1,2,3,7,9}，其余替换为 NaN
    """
    df.replace(5.397605346934028e-79, np.nan, inplace=True)
    
    symptom_cols = [col for col in df.columns if col.startswith('DPQ0') or col == 'DPQ100']
    valid_values = {0, 1, 2, 3, 7, 9}
    
    for col in symptom_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    return df

# =========================
# KIQ 清洗函数
# =========================
def clean_kiq(df):
    """
    清洗 KIQ 数据：
    - 异常值 5.397605346934028e-79 替换为 NaN
    - 拒绝回答（7）和不知道（9）替换为 NaN
    - 删除完全为空的行
    """
    df.replace(5.397605346934028e-79, np.nan, inplace=True)
    
    kiq_cols = [col for col in df.columns if col != 'SEQN']
    df[kiq_cols] = df[kiq_cols].apply(lambda x: x.replace({7: np.nan, 9: np.nan}))
    df[kiq_cols] = df[kiq_cols].apply(pd.to_numeric, errors='coerce')
    
    df.dropna(how='all', subset=kiq_cols, inplace=True)
    
    return df

# =========================
# 批量处理文件函数
# =========================
def process_all_files(root_dir='.'):
    """
    遍历根目录及子目录，自动清洗 DPQ 和 KIQ 文件
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if '_clean' in file:
                continue  # 跳过已清洗文件

            file_path = os.path.join(subdir, file)
            output_path = os.path.join(subdir, file.replace('.csv', '_clean.csv'))

            # DPQ 文件
            if re.match(r'^DPQ.*\.csv$', file) or re.match(r'^P_DPQ.*\.csv$', file):
                print(f"清洗 DPQ 文件: {file_path}")
                df = pd.read_csv(file_path)
                df_clean = clean_dpq_simple(df)
                df_clean.to_csv(output_path, index=False)

            # KIQ 文件
            elif re.match(r'^KIQ.*\.csv$', file):
                print(f"清洗 KIQ 文件: {file_path}")
                df = pd.read_csv(file_path)
                df_clean = clean_kiq(df)
                df_clean.to_csv(output_path, index=False)

    print("所有文件清洗完成！")

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    # 假设脚本在项目根目录下
    process_all_files()

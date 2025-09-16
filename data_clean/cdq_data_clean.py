import pandas as pd
import numpy as np
import os

def clean_cdq_data(df):
    # 定义有效值范围（基于 CDQ_D 文档）
    valid_ranges = {
        'CDQ001': {1, 2, 7, 9},
        'CDQ002': {1, 2, 3, 7, 9},
        'CDQ003': {1, 2, 7, 9},
        'CDQ004': {1, 2, 7, 9},
        'CDQ005': {1, 2, 7, 9},
        'CDQ006': {1, 2, 7, 9},
        'CDQ008': {1, 2, 7, 9},
        'CDQ010': {1, 2, 7, 9},
        'CDQ009A': {1, 77, 99},
        'CDQ009B': {2},
        'CDQ009C': {3},
        'CDQ009D': {4},
        'CDQ009E': {5},
        'CDQ009F': {6},
        'CDQ009G': {7},
        'CDQ009H': {8}
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 计算 Rose 心绞痛等级
    def calculate_rose_angina(row):
        # 确保所有相关字段非空且有效
        required_cols = ['CDQ001', 'CDQ002', 'CDQ004', 'CDQ005', 'CDQ006']
        if all(pd.notna(row[col]) for col in required_cols):
            # 公共条件
            if (row['CDQ001'] == 1 and row['CDQ002'] == 1 and
                row['CDQ004'] == 1 and row['CDQ005'] == 1 and row['CDQ006'] == 1):
                # 疼痛位置条件
                location_condition = (
                    pd.notna(row.get('CDQ009D')) and row.get('CDQ009D') == 4 or
                    pd.notna(row.get('CDQ009E')) and row.get('CDQ009E') == 5 or
                    (pd.notna(row.get('CDQ009F')) and row.get('CDQ009F') == 6 and
                     pd.notna(row.get('CDQ009G')) and row.get('CDQ009G') == 7)
                )
                if location_condition:
                    # 检查 CDQ003 以确定等级
                    if pd.notna(row.get('CDQ003')) and row['CDQ003'] == 1:
                        return 2  # Grade 2 Angina
                    else:
                        return 1  # Grade 1 Angina
        return 0  # 无心绞痛

    df['rose_angina_grade'] = df.apply(calculate_rose_angina, axis=1)

    # 统计缺失值
    cdq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if cdq_cols:
        df['missing_cdq_count'] = df[cdq_cols].isna().sum(axis=1)
        df['all_cdq_missing'] = df['missing_cdq_count'] == len(cdq_cols)

    # 一致性检查：如果 CDQ001 = 2, 7, 9，则 CDQ002 至 CDQ008 应为空
    follow_up_cols = ['CDQ002', 'CDQ003', 'CDQ004', 'CDQ005', 'CDQ006', 'CDQ008']
    for col in follow_up_cols:
        if col in df.columns:
            df[f'{col}_consistency'] = np.where(
                (pd.notna(df['CDQ001'])) & (df['CDQ001'].isin({2, 7, 9})) & (pd.notna(df[col])),
                'invalid_non_pain',
                'valid'
            )

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 CDQ 数据
    df = clean_cdq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_cdq_files(root_dir='.'):
    # 遍历目录，处理 CDQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('CDQ_') or file.startswith('P_CDQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 CDQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_cdq_files()
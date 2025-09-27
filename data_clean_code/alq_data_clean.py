import pandas as pd
import numpy as np
import os

# DPQ 数据清洗函数
def clean_dpq_data(df):
    # 验证数据范围
    symptom_cols = [f'DPQ0{i}0' for i in range(1, 10)]
    valid_symptom_cols = [col for col in symptom_cols if col in df.columns]
    valid_values = {0, 1, 2, 3, 7, 9}
    
    for col in valid_symptom_cols + (['DPQ100'] if 'DPQ100' in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    # 处理缺失数据
    if valid_symptom_cols:
        df['missing_symptom_count'] = df[valid_symptom_cols].isna().sum(axis=1)
        df['all_symptoms_missing'] = df['missing_symptom_count'] == len(valid_symptom_cols)
        
        if 'DPQ100' in df.columns:
            df['any_symptom_endorsed'] = df[valid_symptom_cols].gt(0).any(axis=1)
            df['dpq100_validity'] = np.where(
                df['any_symptom_endorsed'] & df['DPQ100'].isna(),
                'missing_when_expected',
                'valid'
            )
    
    # 计算总分
    if valid_symptom_cols:
        df['total_score'] = df[valid_symptom_cols].apply(
            lambda x: x.sum() if all(x.notna()) and all(x.isin({0, 1, 2, 3})) else np.nan,
            axis=1
        )
    
    return df

# ALQ 数据清洗函数
def clean_alq_data(df):
    # 定义变量及其有效值范围（基于 ALQ_L 文档）
    valid_ranges = {
        'ALQ111': {1, 2, 7, 9},
        'ALQ121': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ130': set(range(1, 16)) | {777, 999},  # 1-14, 15+, Refused, Don't know
        'ALQ142': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ270': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ280': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ151': {1, 2, 7, 9},
        'ALQ170': set(range(0, 21)) | {30, 777, 999},  # 0-20, 30+, Refused, Don't know
        # 早期变量（可能出现在旧年份）
        'ALQ101': {1, 2, 7, 9},
        'ALQ110': {1, 2, 7, 9},
        'ALQ120Q': set(range(1, 366)),  # 假设为饮酒频率（天数）
        'ALQ120U': {1, 2, 3, 7, 9},  # 单位：周、月、年
        'ALQ140Q': set(range(1, 366)),
        'ALQ140U': {1, 2, 3, 7, 9},
        'ALQ150': {1, 2, 7, 9}
    }
    
    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    # 一致性检查：ALQ142, ALQ270, ALQ280 不能低于 ALQ121（除非为 0）
    consistency_cols = ['ALQ142', 'ALQ270', 'ALQ280']
    if 'ALQ121' in df.columns:
        for col in consistency_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (df[col].notna()) & (df['ALQ121'].notna()) & (df[col] != 0) & (df[col] < df['ALQ121']),
                    'inconsistent',
                    'valid'
                )
    
    # 计算缺失值数量
    alq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if alq_cols:
        df['missing_alq_count'] = df[alq_cols].isna().sum(axis=1)
    
    return df

# 加载数据
def load_data(file_path):
    # 读取CSV，将5.397605346934028e-79和空字段视为NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

# 主清洗函数
def clean_data(file_path, output_path, file_type):
    # 加载数据
    df = load_data(file_path)
    
    # 根据文件类型调用相应的清洗函数
    if file_type == 'dpq':
        df = clean_dpq_data(df)
    elif file_type == 'alq':
        df = clean_alq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

# 遍历所有年份文件夹，处理 DPQ 和 ALQ 文件
def process_all_files(root_dir='.'):
    # 遍历根目录下的所有子目录
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 处理 DPQ 文件
            if (file.startswith('DPQ_') or file.startswith('P_DPQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 DPQ 文件: {input_file}")
                clean_data(input_file, output_file, 'dpq')
            # 处理 ALQ 文件
            elif (file.startswith('ALQ_') or file.startswith('P_ALQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 ALQ 文件: {input_file}")
                clean_data(input_file, output_file, 'alq')

# 示例用法
if __name__ == "__main__":
    process_all_files()
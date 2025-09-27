import pandas as pd
import numpy as np
import os

def clean_inq_data(df):
    # 定义有效值范围（基于 INQ_G 文档）
    valid_ranges = {
        'INQ020': {1, 2, 7, 9},  # 工资/薪水收入
        'INQ012': {1, 2, 7, 9},  # 自雇收入
        'INQ030': {1, 2, 7, 9},  # 社会保障或铁路退休金
        'INQ060': {1, 2, 7, 9},  # 其他残疾养老金
        'INQ080': {1, 2, 7, 9},  # 退休/遗属养老金
        'INQ090': {1, 2, 7, 9},  # 补充保障收入
        'INQ132': {1, 2, 7, 9},  # 州/县现金援助
        'INQ140': {1, 2, 7, 9},  # 利息/股息或租金收入
        'INQ150': {1, 2, 7, 9},  # 其他收入来源
        'IND235': set(range(1, 13)) | {77, 99},  # 月收入范围
        'INDFMMPI': set(np.arange(0, 5.00, 0.01)) | {5.0},  # 月贫困水平指数
        'INDFMMPC': {1, 2, 3, 7, 9},  # 月贫困水平指数类别
        'INQ244': {1, 2, 7, 9},  # 储蓄/现金资产是否超过5000美元
        'IND247': set(range(1, 7)) | {77, 99}  # 储蓄/现金资产范围
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 IND235 = 77 或 99（拒绝或不知道），INDFMMPI 和 INDFMMPC 应为空
    if 'IND235' in df.columns:
        for col in ['INDFMMPI', 'INDFMMPC']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['IND235'])) & (df['IND235'].isin({77, 99})) & (pd.notna(df[col])),
                    'invalid_income_missing',
                    'valid'
                )

    # 2. 如果 INDFMMPI 非空，INDFMMPC 应非空且匹配类别
    if 'INDFMMPI' in df.columns and 'INDFMMPC' in df.columns:
        df['INDFMMPC_consistency'] = np.where(
            pd.notna(df['INDFMMPI']),
            np.where(
                ((df['INDFMMPI'] <= 1.30) & (df['INDFMMPC'] == 1)) |
                ((df['INDFMMPI'] > 1.30) & (df['INDFMMPI'] <= 1.85) & (df['INDFMMPC'] == 2)) |
                ((df['INDFMMPI'] > 1.85) & (df['INDFMMPC'] == 3)),
                'valid',
                'invalid_category_mismatch'
            ),
            np.where(
                pd.isna(df['INDFMMPC']),
                'missing_category',
                'valid'
            )
        )

    # 3. 如果 INQ244 = 1, 7, 或 9（有储蓄>5000美元、拒绝或不知道），IND247 应为空
    if 'INQ244' in df.columns and 'IND247' in df.columns:
        df['IND247_consistency'] = np.where(
            (pd.notna(df['INQ244'])) & (df['INQ244'].isin({1, 7, 9})) & (pd.notna(df['IND247'])),
            'invalid_savings_high',
            'valid'
        )

    # 4. INQ244 和 IND247 仅适用于年收入 ≤ 200% 贫困线，需家庭规模 (DMDFMSIZ) 检查
    # 假设 DMDFMSIZ 存在，若无此列需移除此检查
    if 'DMDFMSIZ' in df.columns and 'INQ244' in df.columns:
        poverty_threshold = 21660 + (df['DMDFMSIZ'] - 1) * 7480
        df['INQ244_consistency'] = np.where(
            (pd.notna(df['IND235'])) & (~df['IND235'].isin({77, 99})) &
            (df['IND235'] * 12 > 2 * poverty_threshold) & (pd.notna(df['INQ244'])),
            'invalid_income_too_high',
            'valid'
        )
        if 'IND247' in df.columns:
            df['IND247_consistency'] = np.where(
                (pd.notna(df['IND235'])) & (~df['IND235'].isin({77, 99})) &
                (df['IND235'] * 12 > 2 * poverty_threshold) & (pd.notna(df['IND247'])),
                'invalid_income_too_high',
                df['IND247_consistency'] if 'IND247_consistency' in df.columns else 'valid'
            )

    # 统计缺失值
    inq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if inq_cols:
        df['missing_inq_count'] = df[inq_cols].isna().sum(axis=1)
        df['all_inq_missing'] = df['missing_inq_count'] == len(inq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 INQ 数据
    df = clean_inq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_inq_files(root_dir='.'):
    # 遍历目录，处理 INQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('INQ_') or file.startswith('P_INQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 INQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_inq_files()
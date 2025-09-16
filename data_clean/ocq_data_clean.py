import pandas as pd
import numpy as np
import os

def clean_ocq_data(df):
    # 定义有效值范围（基于 OCQ_G 文档）
    valid_ranges = {
        'OCD150': {1, 2, 3, 4, 7, 9},  # 上周工作类型
        'OCQ180': set(range(1, 169)) | {777, 999},  # 上周工作小时数
        'OCQ210': {1, 2, 7, 9},  # 是否通常每周工作35小时以上
        'OCD231': set(range(1, 23)) | {77, 99},  # 当前工作职业组
        'OCD241': set(range(1, 23)) | {77, 99},  # 当前工作行业组
        'OCQ260': {1, 2, 3, 4, 7, 9},  # 当前工作部门（私营、联邦等）
        'OCQ265': {1, 2, 7, 9},  # 当前工作场所吸烟政策
        'OCD270': set(range(0, 960)) | {777, 999},  # 当前工作持续月份
        'OCQ290G': {1, 2, 3, 7, 9},  # 是否暴露于二手烟
        'OCQ290Q': set(range(1, 169)) | {777, 999},  # 暴露二手烟小时数
        'OCQ380': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},  # 未工作原因
        'OCD390G': {1, 2, 3, 7, 9},  # 最长工作类型
        'OCD391': set(range(1, 23)) | {77, 99},  # 最长工作职业组
        'OCD392': set(range(1, 23)) | {77, 99},  # 最长工作行业组
        'OCD395': set(range(0, 960)) | {777, 999},  # 最长工作持续月份
        'OCQ510': {1, 2, 7, 9},  # 是否暴露矿物/有机粉尘
        'OCQ520': {1, 2, 7, 9},  # 暴露矿物粉尘年数
        'OCQ530': {1, 2, 7, 9},  # 暴露有机粉尘年数
        'OCQ540': {1, 2, 7, 9},  # 暴露有机粉尘年数（重复定义，文档中为有机粉尘）
        'OCQ550': {1, 2, 7, 9},  # 是否暴露尾气
        'OCQ560': set(range(0, 80)) | {7777, 9999},  # 暴露尾气年数
        'OCQ570': {1, 2, 7, 9},  # 是否暴露其他气体/烟雾
        'OCQ580': set(range(0, 80)) | {7777, 9999}  # 暴露其他气体/烟雾年数
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 OCD150 ≠ 1 或 2（未工作），当前工作相关列应为空
    if 'OCD150' in df.columns:
        current_job_cols = ['OCQ180', 'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCQ265', 'OCD270', 'OCQ290G', 'OCQ290Q']
        for col in current_job_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OCD150'])) & (~df['OCD150'].isin({1, 2})) & (pd.notna(df[col])),
                    'invalid_no_current_job',
                    'valid'
                )

    # 2. 如果 OCD390G = 2（最长工作=当前工作），OCD391/OCD392/OCD395 应为空或匹配当前
    if 'OCD390G' in df.columns:
        for col_map in {'OCD391': 'OCD231', 'OCD392': 'OCD241', 'OCD395': 'OCD270'}.items():
            src_col, tgt_col = col_map
            if src_col in df.columns and tgt_col in df.columns:
                df[f'{src_col}_consistency'] = np.where(
                    (pd.notna(df['OCD390G'])) & (df['OCD390G'] == 2) & (pd.notna(df[src_col])) & (df[src_col] != df[tgt_col]),
                    'invalid_longest_matches_current',
                    'valid'
                )

    # 3. 如果 OCD390G ≠ 1（最长工作非其他），OCD391/OCD392/OCD395 应为空
    if 'OCD390G' in df.columns:
        longest_cols = ['OCD391', 'OCD392', 'OCD395']
        for col in longest_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OCD390G'])) & (df['OCD390G'] != 1) & (pd.notna(df[col])),
                    'invalid_no_longest_job',
                    'valid'
                )

    # 4. 暴露年数不能超过年龄（假设 RIDAGEYR 存在，若无需移除）
    if 'RIDAGEYR' in df.columns:
        exposure_cols = ['OCQ520', 'OCQ530', 'OCQ540', 'OCQ560', 'OCQ580', 'OCD270', 'OCD395']
        for col in exposure_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df[col])) & (~df[col].isin({7777, 9999})) & (df[col] >= df['RIDAGEYR']),
                    'invalid_exceeds_age',
                    'valid'
                )

    # 5. 如果 OCQ510 = 2,7,9（未暴露粉尘），OCQ520/OCQ530/OCQ540 应为空
    if 'OCQ510' in df.columns:
        dust_cols = ['OCQ520', 'OCQ530', 'OCQ540']
        for col in dust_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OCQ510'])) & (df['OCQ510'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_dust_exposure',
                    'valid'
                )

    # 类似检查 OCQ550 与 OCQ560, OCQ570 与 OCQ580
    if 'OCQ550' in df.columns and 'OCQ560' in df.columns:
        df['OCQ560_consistency'] = np.where(
            (pd.notna(df['OCQ550'])) & (df['OCQ550'].isin({2, 7, 9})) & (pd.notna(df['OCQ560'])),
            'invalid_no_exhaust_exposure',
            'valid'
        )
    if 'OCQ570' in df.columns and 'OCQ580' in df.columns:
        df['OCQ580_consistency'] = np.where(
            (pd.notna(df['OCQ570'])) & (df['OCQ570'].isin({2, 7, 9})) & (pd.notna(df['OCQ580'])),
            'invalid_no_other_fumes',
            'valid'
        )

    # 衍生变量：总暴露年数
    exposure_cols = ['OCQ520', 'OCQ530', 'OCQ540', 'OCQ560', 'OCQ580']
    if all(col in df.columns for col in exposure_cols):
        df['total_exposure_years'] = df[exposure_cols].sum(axis=1, skipna=True)

    # 统计缺失值
    ocq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if ocq_cols:
        df['missing_ocq_count'] = df[ocq_cols].isna().sum(axis=1)
        df['all_ocq_missing'] = df['missing_ocq_count'] == len(ocq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 OCQ 数据
    df = clean_ocq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_ocq_files(root_dir='.'):
    # 遍历目录，处理 OCQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('OCQ_') or file.startswith('P_OCQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 OCQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_ocq_files()
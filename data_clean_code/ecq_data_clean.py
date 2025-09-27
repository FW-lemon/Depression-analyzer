import pandas as pd
import numpy as np
import os

def clean_ecq_data(df):
    # 定义有效值范围（基于 ECQ_D 文档）
    valid_ranges = {
        'ECD010': set(range(14, 60)) | {7777, 9999},  # 母亲生育年龄
        'ECQ020': {1, 2, 7, 9},  # 母亲怀孕时吸烟
        'ECQ030': {1, 2, 7, 9},  # 怀孕期间戒烟
        'ECQ040': {1, 2, 3, 4, 5, 6, 7, 8, 9, 77, 99},  # 戒烟月份
        'ECQ060': {1, 2, 7, 9},  # 新生儿特殊护理
        'ECD070A': set(range(1, 14)) | {7777, 9999},  # 出生体重（磅）
        'ECD070B': set(range(0, 16)) | {7777, 9999},  # 出生体重（盎司）
        'ECQ080': {1, 2, 7, 9},  # 体重是否大于5.5磅
        'ECQ090': {1, 2, 7, 9},  # 体重是否大于9磅
        'WHQ030E': {1, 2, 3, 7, 9},  # 当前体重评估
        'MCQ080E': {1, 2, 7, 9},  # 医生告知超重
        'ECQ150': {1, 2, 7, 9},  # 是否控制体重
        'FSQ121': {1, 2, 7, 9}  # 是否参加 Head Start
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 ECQ020 ≠ 1（母亲未吸烟、拒绝或不知道），ECQ030 和 ECQ040 应为空
    if 'ECQ020' in df.columns:
        for col in ['ECQ030', 'ECQ040']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['ECQ020'])) & (df['ECQ020'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_non_smoker',
                    'valid'
                )

    # 2. 如果 ECQ030 ≠ 1（未戒烟、拒绝或不知道），ECQ040 应为空
    if 'ECQ030' in df.columns and 'ECQ040' in df.columns:
        df['ECQ040_consistency'] = np.where(
            (pd.notna(df['ECQ030'])) & (df['ECQ030'].isin({2, 7, 9})) & (pd.notna(df['ECQ040'])),
            'invalid_non_quit',
            'valid'
        )

    # 3. 如果 ECD070A 非拒绝/不知道（7777/9999），ECQ080 和 ECQ090 应为空
    if 'ECD070A' in df.columns:
        for col in ['ECQ080', 'ECQ090']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['ECD070A'])) & (~df['ECD070A'].isin({7777, 9999})) & (pd.notna(df[col])),
                    'invalid_weight_known',
                    'valid'
                )

    # 4. 如果 ECQ080 = 2（小于5.5磅），ECQ090 应为空
    if 'ECQ080' in df.columns and 'ECQ090' in df.columns:
        df['ECQ090_consistency'] = np.where(
            (pd.notna(df['ECQ080'])) & (df['ECQ080'] == 2) & (pd.notna(df['ECQ090'])),
            'invalid_low_weight',
            'valid'
        )

    # 5. 年龄限制检查：WHQ030E, MCQ080E, ECQ150 仅适用于 2-15 岁，FSQ121 仅适用于 0-5 岁
    if 'RIDAGEYR' in df.columns:  # 假设数据包含年龄变量 RIDAGEYR
        if 'WHQ030E' in df.columns:
            df['WHQ030E_consistency'] = np.where(
                (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < 2) & (pd.notna(df['WHQ030E'])),
                'invalid_age',
                'valid'
            )
        if 'MCQ080E' in df.columns:
            df['MCQ080E_consistency'] = np.where(
                (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < 2) & (pd.notna(df['MCQ080E'])),
                'invalid_age',
                'valid'
            )
        if 'ECQ150' in df.columns:
            df['ECQ150_consistency'] = np.where(
                (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < 2) & (pd.notna(df['ECQ150'])),
                'invalid_age',
                'valid'
            )
        if 'FSQ121' in df.columns:
            df['FSQ121_consistency'] = np.where(
                (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] > 5) & (pd.notna(df['FSQ121'])),
                'invalid_age',
                'valid'
            )

    # 6. 如果 MCQ080E ≠ 1（未被告知超重），ECQ150 应为空
    if 'MCQ080E' in df.columns and 'ECQ150' in df.columns:
        df['ECQ150_consistency'] = np.where(
            (pd.notna(df['MCQ080E'])) & (df['MCQ080E'].isin({2, 7, 9})) & (pd.notna(df['ECQ150'])),
            'invalid_non_overweight',
            'valid'
        )

    # 计算总出生体重（磅+盎司转换为盎司）
    if 'ECD070A' in df.columns and 'ECD070B' in df.columns:
        df['birth_weight_oz'] = np.where(
            (pd.notna(df['ECD070A'])) & (~df['ECD070A'].isin({7777, 9999})) &
            (pd.notna(df['ECD070B'])) & (~df['ECD070B'].isin({7777, 9999})),
            df['ECD070A'] * 16 + df['ECD070B'],
            np.nan
        )

    # 统计缺失值
    ecq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if ecq_cols:
        df['missing_ecq_count'] = df[ecq_cols].isna().sum(axis=1)
        df['all_ecq_missing'] = df['missing_ecq_count'] == len(ecq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 ECQ 数据
    df = clean_ecq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_ecq_files(root_dir='.'):
    # 遍历目录，处理 ECQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('ECQ_') or file.startswith('P_ECQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 ECQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_ecq_files()
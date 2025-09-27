import pandas as pd
import numpy as np
import os

# DPQ 数据清洗函数
def clean_dpq_data(df):
    symptom_cols = [f'DPQ0{i}0' for i in range(1, 10)]
    valid_symptom_cols = [col for col in symptom_cols if col in df.columns]
    valid_values = {0, 1, 2, 3, 7, 9}
    
    for col in valid_symptom_cols + (['DPQ100'] if 'DPQ100' in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
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
    
    if valid_symptom_cols:
        df['total_score'] = df[valid_symptom_cols].apply(
            lambda x: x.sum() if all(x.notna()) and all(x.isin({0, 1, 2, 3})) else np.nan,
            axis=1
        )
    
    return df

# ALQ 数据清洗函数
def clean_alq_data(df):
    valid_ranges = {
        'ALQ111': {1, 2, 7, 9},
        'ALQ121': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ130': set(range(1, 16)) | {777, 999},
        'ALQ142': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ270': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ280': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99},
        'ALQ151': {1, 2, 7, 9},
        'ALQ170': set(range(0, 21)) | {30, 777, 999},
        'ALQ101': {1, 2, 7, 9},
        'ALQ110': {1, 2, 7, 9},
        'ALQ120Q': set(range(1, 366)),
        'ALQ120U': {1, 2, 3, 7, 9},
        'ALQ140Q': set(range(1, 366)),
        'ALQ140U': {1, 2, 3, 7, 9},
        'ALQ150': {1, 2, 7, 9}
    }
    
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    consistency_cols = ['ALQ142', 'ALQ270', 'ALQ280']
    if 'ALQ121' in df.columns:
        for col in consistency_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (df[col].notna()) & (df['ALQ121'].notna()) & (df[col] != 0) & (df[col] < df['ALQ121']),
                    'inconsistent',
                    'valid'
                )
    
    alq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if alq_cols:
        df['missing_alq_count'] = df[alq_cols].isna().sum(axis=1)
    
    return df

# DIQ 数据清洗函数
def clean_diq_data(df):
    valid_ranges = {
        'DIQ010': {1, 2, 3, 7, 9},
        'DID040': set(range(1, 81)) | {666, 777, 999},
        'DIQ160': {1, 2, 7, 9},
        'DIQ180': {1, 2, 7, 9},
        'DIQ050': {1, 2, 7, 9},
        'DID060': set(range(1, 61)) | {666, 777, 999},
        'DIQ060U': {1, 2},
        'DIQ070': {1, 2, 7, 9},
        'DIQ220': {1, 2, 7, 9},
        'DIQ190A': {1, 2, 7, 9},
        'DIQ190B': {1, 2, 7, 9},
        'DIQ190C': {1, 2, 7, 9},
        'DIQ200A': {1, 2, 7, 9},
        'DIQ200B': {1, 2, 7, 9},
        'DIQ200C': {1, 2, 7, 9},
        'DIQ230': {1, 2, 7, 9},
        'DIQ240': {1, 2, 7, 9},
        'DID250': set(range(1, 100)) | {666, 777, 999},
        'DID260': set(range(1, 100)) | {666, 777, 999},
        'DIQ260U': {1, 2, 7, 9},
        'DID270': set(range(1, 100)) | {666, 777, 999},
        'DIQ280': {1, 2, 7, 9},
        'DIQ290': {1, 2, 7, 9},
        'DIQ300S': set(range(0, 300)) | {6666, 7777, 9999},
        'DIQ300D': set(range(0, 300)) | {6666, 7777, 9999},
        'DID310S': set(range(0, 300)) | {6666, 7777, 9999},
        'DID310D': set(range(0, 300)) | {6666, 7777, 9999},
        'DID320': set(range(0, 300)) | {6666, 7777, 9999},
        'DID330': set(range(0, 300)) | {6666, 7777, 9999},
        'DID340': set(range(0, 100)) | {6666, 7777, 9999},
        'DID350': set(range(0, 100)) | {6666, 7777, 9999},
        'DIQ350U': {1, 2, 7, 9},
        'DIQ360': {1, 2, 7, 9},
        'DIQ080': {1, 2, 7, 9}
    }
    
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    if 'DIQ010' in df.columns:
        diabetes_cols = ['DID040', 'DIQ050', 'DID060', 'DIQ060U', 'DIQ070']
        for col in diabetes_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (df['DIQ010'].notna()) & (df['DIQ010'] != 1) & (df[col].notna()),
                    'invalid_non_diabetic',
                    'valid'
                )
    
    if 'DIQ050' in df.columns and 'DID060' in df.columns and 'DIQ060U' in df.columns:
        df['DID060_consistency'] = np.where(
            (df['DIQ050'].notna()) & (df['DIQ050'] != 1) & (df['DID060'].notna()),
            'invalid_non_insulin',
            'valid'
        )
        df['DIQ060U_consistency'] = np.where(
            (df['DIQ050'].notna()) & (df['DIQ050'] != 1) & (df['DIQ060U'].notna()),
            'invalid_non_insulin',
            'valid'
        )
    
    diq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if diq_cols:
        df['missing_diq_count'] = df[diq_cols].isna().sum(axis=1)
    
    return df

# DBQ 数据清洗函数
def clean_dbq_data(df):
    # 定义变量及其有效值范围（基于 DBQ_L 文档）
    valid_ranges = {
        'DBQ010': {1, 2, 7, 9},
        'DBD030': set(range(0, 1096)) | {666666, 777777, 999999},
        'DBD041': set(range(0, 366)) | {666666, 777777, 999999},
        'DBD050': set(range(0, 1096)) | {666666, 777777, 999999},
        'DBD055': set(range(0, 731)) | {666666, 777777, 999999},
        'DBD061': set(range(0, 1096)) | {666666, 777777, 999999},
        'DBQ073A': {10},  # 牛奶类型，文档中为10-30
        'DBQ073B': {11},
        'DBQ073C': {12},
        'DBQ073D': {13},
        'DBQ073U': {30},
        'DBQ700': {1, 2, 3, 4, 5, 7, 9},  # 健康饮食评分
        'DBQ197': {1, 2, 3, 4, 5, 7, 9},  # 过去30天奶制品摄入
        'DBQ301': {1, 2, 7, 9},  # 社区餐食
        'DBQ330': {1, 2, 7, 9},
        'DBQ360': {1, 2, 7, 9},
        'DBQ370': {1, 2, 7, 9},
        'DBD381': set(range(0, 6)) | {7777, 9999},
        'DBQ390': {1, 2, 3, 7, 9},
        'DBQ400': {1, 2, 7, 9},
        'DBD411': set(range(0, 6)) | {7777, 9999},
        'DBQ421': {1, 2, 3, 7, 9},
        'DBQ424': {1, 2, 3, 7, 9},
        'DBD895': set(range(0, 100)) | {7777, 9999},  # 假设为数值变量
        'DBD900': set(range(0, 100)) | {7777, 9999},
        'DBD905': set(range(0, 100)) | {7777, 9999},
        'DBD910': set(range(0, 100)) | {7777, 9999},
        'DBQ915': {1, 2, 7, 9},
        'DBQ920': {1, 2, 7, 9},
        'DBQ925A': {1},
        'DBQ925B': {2},
        'DBQ925C': {3},
        'DBQ925D': {4},
        'DBQ925E': {5},
        'DBQ925F': {6},
        'DBQ925G': {7},
        'DBQ925H': {8},
        'DBQ925I': {9},
        'DBQ925J': {10},
        'DBQ930': {1, 2, 7, 9},
        'DBQ935': {1, 2, 7, 9},
        'DBQ940': {1, 2, 7, 9},
        'DBQ945': {1, 2, 7, 9},
        # 样本数据中的额外变量（假设范围，早期可能不同）
        'DBD020': set(range(0, 1096)) | {666666, 777777, 999999},  # 假设为年龄（天）
        'DBD040': set(range(0, 1096)) | {666666, 777777, 999999},
        'DBD060': set(range(0, 1096)) | {666666, 777777, 999999},
        'DBD072A': {10, 11, 12, 13, 30},
        'DBD072B': {10, 11, 12, 13, 30},
        'DBD072C': {10, 11, 12, 13, 30},
        'DBD072D': {10, 11, 12, 13, 30},
        'DBD072U': {10, 11, 12, 13, 30},
        'DBD080': set(range(0, 366)) | {666666, 777777, 999999},
        'DBD222A': {10, 11, 12, 13, 30},
        'DBD222B': {10, 11, 12, 13, 30},
        'DBD222C': {10, 11, 12, 13, 30},
        'DBD222D': {10, 11, 12, 13, 30},
        'DBD222U': {10, 11, 12, 13, 30},
        'DBQ229': {1, 2, 3, 4, 5, 7, 9},
        'DBQ235A': {1, 2, 3, 7, 9},
        'DBQ235B': {1, 2, 3, 7, 9},
        'DBQ235C': {1, 2, 3, 7, 9}
    }
    
    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    # 一致性检查
    # 1. 如果 DBQ010 ≠ 1（未母乳喂养），则 DBD030 应为缺失
    if 'DBQ010' in df.columns and 'DBD030' in df.columns:
        df['DBD030_consistency'] = np.where(
            (df['DBQ010'].notna()) & (df['DBQ010'] != 1) & (df['DBD030'].notna()),
            'invalid_non_breastfed',
            'valid'
        )
    
    # 2. 对于年龄变量，确保特殊代码一致（例如如果超过上限，标记为666666）
    age_cols = ['DBD030', 'DBD041', 'DBD050', 'DBD055', 'DBD061', 'DBD020', 'DBD040', 'DBD060', 'DBD080']
    for col in age_cols:
        if col in df.columns:
            # 示例：对于 DBD030 (停止母乳)，如果 >1095，标记为666666
            if col == 'DBD030' or col == 'DBD050' or col == 'DBD061':
                df[col] = np.where(df[col] > 1095, 666666, df[col])
            elif col == 'DBD041':
                df[col] = np.where(df[col] > 365, 666666, df[col])
            elif col == 'DBD055':
                df[col] = np.where(df[col] > 730, 666666, df[col])
    
    # 计算缺失值数量
    dbq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if dbq_cols:
        df['missing_dbq_count'] = df[dbq_cols].isna().sum(axis=1)
    
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
    elif file_type == 'diq':
        df = clean_diq_data(df)
    elif file_type == 'dbq':
        df = clean_dbq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

# 遍历所有年份文件夹，处理 DPQ、ALQ、DIQ 和 DBQ 文件
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
            # 处理 DIQ 文件
            elif (file.startswith('DIQ_') or file.startswith('P_DIQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 DIQ 文件: {input_file}")
                clean_data(input_file, output_file, 'diq')
            # 处理 DBQ 文件
            elif (file.startswith('DBQ_') or file.startswith('P_DBQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 DBQ 文件: {input_file}")
                clean_data(input_file, output_file, 'dbq')

# 示例用法
if __name__ == "__main__":
    process_all_files()
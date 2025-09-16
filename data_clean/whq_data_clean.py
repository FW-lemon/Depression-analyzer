import pandas as pd
import numpy as np
import os

def clean_whq_data(df):
    # 定义有效值范围（基于 WHQ_G 文档）
    valid_ranges = {
        'WHD010': set(range(53, 85)) | {7777, 9999},  # 当前自报身高（英寸）
        'WHD020': set(range(70, 465)) | {7777, 9999},  # 当前自报体重（磅）
        'WHQ030': {1, 2, 3, 7, 9},  # 自我体重认知
        'WHQ040': {1, 2, 3, 7, 9},  # 是否尝试改变体重
        'WHD050': set(range(50, 551)) | {7777, 9999},  # 一年前体重
        'WHQ060': {1, 2, 7, 9},  # 是否尝试减重（过去12个月）
        'WHQ070': {1, 2, 7, 9},  # 是否尝试维持体重
        'WHD080A': {10},  # 减重方法：少吃
        'WHD080B': {11},  # 减重方法：低热量饮食
        'WHD080C': {12},  # 减重方法：低脂/低胆固醇
        'WHD080D': {13},  # 减重方法：低盐/低钠
        'WHD080E': {14},  # 减重方法：无糖/低糖
        'WHD080F': {15},  # 减重方法：低纤维
        'WHD080G': {16},  # 减重方法：高蛋白
        'WHD080H': {17},  # 减重方法：减肥药
        'WHD080I': {18},  # 减重方法：减肥计划
        'WHD080J': {19},  # 减重方法：运动
        'WHD080K': {20},  # 减重方法：喝水
        'WHD080L': {33},  # 减重方法：其他
        'WHD080M': {34},  # 减重方法：不吃晚餐
        'WHD080N': {41},  # 减重方法：流食
        'WHD080O': {42},  # 减重方法：低碳水化合物
        'WHD080P': {43},  # 减重方法：减肥手术
        'WHD080Q': {44},  # 减重方法：抽脂
        'WHD080R': {45},  # 减重方法：减肥茶
        'WHD080S': {46},  # 减重方法：减肥补品
        'WHD080T': {47},  # 减重方法：不吃早餐
        'WHD080U': {48},  # 减重方法：改变饮食习惯
        'WHQ225': {1, 2, 3, 4, 5, 7, 9},  # 过去12个月体重变化意图
        'WHD110': set(range(70, 551)) | {7777, 9999},  # 10年前体重（36+岁）
        'WHD120': set(range(80, 414)) | {7777, 9999},  # 25岁时体重（27+岁）
        'WHD130': set(range(49, 84)) | {7777, 9999},  # 25岁时身高（50+岁）
        'WHD140': set(range(75, 551)) | {7777, 9999},  # 最大体重（18+岁）
        'WHQ150': set(range(7, 81)) | {77777, 99999},  # 最大体重年龄（18+岁）
        'WHQ190': {1, 2, 7, 9},  # 是否考虑减重
        'WHQ200': {1, 2, 7, 9}  # 是否考虑维持体重
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 年龄限制检查（假设 RIDAGEYR 存在）
    if 'RIDAGEYR' in df.columns:
        age_checks = {
            'WHD110': 36,  # 10年前体重（36+岁）
            'WHD120': 27,  # 25岁时体重（27+岁）
            'WHD130': 50,  # 25岁时身高（50+岁）
            'WHD140': 18,  # 最大体重（18+岁）
            'WHQ150': 18   # 最大体重年龄（18+岁）
        }
        for col, min_age in age_checks.items():
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < min_age) & (pd.notna(df[col])),
                    f'invalid_age_under_{min_age}',
                    'valid'
                )

    # 2. 最大体重年龄 (WHQ150) 不能大于当前年龄
    if 'WHQ150' in df.columns and 'RIDAGEYR' in df.columns:
        df['WHQ150_consistency'] = np.where(
            (pd.notna(df['WHQ150'])) & (~df['WHQ150'].isin({77777, 99999})) & (df['WHQ150'] > df['RIDAGEYR']),
            'invalid_exceeds_age',
            'valid'
        )

    # 3. 如果 WHQ060 = 2,7,9（未尝试减重），减重方法 (WHD080*) 应为空
    if 'WHQ060' in df.columns:
        weight_loss_cols = [f'WHD080{c}' for c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']]
        for col in weight_loss_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['WHQ060'])) & (df['WHQ060'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_weight_loss_attempt',
                    'valid'
                )

    # 4. 体重一致性：WHD140（最大体重）应不小于 WHD020（当前体重）、WHD050（一年前）、WHD110（10年前）、WHD120（25岁时）
    weight_cols = ['WHD020', 'WHD050', 'WHD110', 'WHD120']
    if 'WHD140' in df.columns:
        for col in weight_cols:
            if col in df.columns:
                df[f'{col}_consistency_max'] = np.where(
                    (pd.notna(df['WHD140'])) & (~df['WHD140'].isin({7777, 9999})) &
                    (pd.notna(df[col])) & (~df[col].isin({7777, 9999})) & (df['WHD140'] < df[col]),
                    'invalid_max_weight_too_low',
                    'valid'
                )

    # 5. 身高合理性：WHD010（当前身高）应与 WHD130（25岁时身高）接近（允许±2英寸）
    if 'WHD010' in df.columns and 'WHD130' in df.columns:
        df['WHD130_consistency_height'] = np.where(
            (pd.notna(df['WHD010'])) & (~df['WHD010'].isin({7777, 9999})) &
            (pd.notna(df['WHD130'])) & (~df['WHD130'].isin({7777, 9999})) &
            (abs(df['WHD010'] - df['WHD130']) > 2),
            'invalid_height_mismatch',
            'valid'
        )

    # 统计缺失值
    whq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if whq_cols:
        df['missing_whq_count'] = df[whq_cols].isna().sum(axis=1)
        df['all_whq_missing'] = df['missing_whq_count'] == len(whq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 WHQ 数据
    df = clean_whq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    # 删除原始文件
    try:
        os.remove(file_path)
        print(f"原始文件已删除: {file_path}")
    except OSError as e:
        print(f"删除原始文件失败 {file_path}: {e}")
    
    return df

def process_whq_files(root_dir='.'):
    # 遍历目录，处理 WHQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('WHQ_') or file.startswith('P_WHQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 WHQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_whq_files()
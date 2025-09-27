import pandas as pd
import numpy as np
import os

def clean_paq_data(df):
    # 定义有效值范围（基于 PAQ_G 文档）
    valid_ranges = {
        'PAQ706': set(range(0, 8)) | {77, 99},  # 至少60分钟活动天数 (2-11岁)
        'PAD590': {0, 1, 2, 3, 4, 5, 8, 77, 99},  # 过去30天看TV小时
        'PAD600': {0, 1, 2, 3, 4, 5, 8, 77, 99},  # 过去30天使用电脑小时
        'PAQ605': {1, 2, 7, 9},  # 剧烈工作活动
        'PAQ610': set(range(0, 8)) | {77, 99},  # 剧烈工作天数
        'PAD615': set(range(0, 961)) | {777, 999},  # 剧烈工作分钟
        'PAQ620': {1, 2, 7, 9},  # 中等工作活动
        'PAQ625': set(range(0, 8)) | {77, 99},  # 中等工作天数
        'PAD630': set(range(0, 961)) | {777, 999},  # 中等工作分钟
        'PAQ635': {1, 2, 7, 9},  # 步行或骑自行车
        'PAQ640': set(range(0, 8)) | {77, 99},  # 步行/骑车天数
        'PAD645': set(range(0, 961)) | {777, 999},  # 步行/骑车分钟
        'PAQ650': {1, 2, 7, 9},  # 剧烈休闲活动
        'PAQ655': set(range(0, 8)) | {77, 99},  # 剧烈休闲天数
        'PAD660': set(range(0, 961)) | {777, 999},  # 剧烈休闲分钟
        'PAQ665': {1, 2, 7, 9},  # 中等休闲活动
        'PAQ670': set(range(0, 8)) | {77, 99},  # 中等休闲天数
        'PAD675': set(range(0, 961)) | {777, 999},  # 中等休闲分钟
        'PAD680': set(range(0, 961)) | {777, 999},  # 久坐分钟
        'PAAQUEX': {1, 2}  # 问卷来源标志
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 PAQ605 = 2,7,9（无剧烈工作），PAQ610 和 PAD615 应为空
    if 'PAQ605' in df.columns:
        for col in ['PAQ610', 'PAD615']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['PAQ605'])) & (df['PAQ605'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_vigorous_work',
                    'valid'
                )

    # 类似检查其他活动类型
    if 'PAQ620' in df.columns:
        for col in ['PAQ625', 'PAD630']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['PAQ620'])) & (df['PAQ620'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_moderate_work',
                    'valid'
                )

    if 'PAQ635' in df.columns:
        for col in ['PAQ640', 'PAD645']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['PAQ635'])) & (df['PAQ635'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_transport',
                    'valid'
                )

    if 'PAQ650' in df.columns:
        for col in ['PAQ655', 'PAD660']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['PAQ650'])) & (df['PAQ650'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_vigorous_leisure',
                    'valid'
                )

    if 'PAQ665' in df.columns:
        for col in ['PAQ670', 'PAD675']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['PAQ665'])) & (df['PAQ665'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_moderate_leisure',
                    'valid'
                )

    # 2. 总活动时间不超过24小时/天 (假设每天7天计算)
    time_cols = ['PAD615', 'PAD630', 'PAD645', 'PAD660', 'PAD675', 'PAD680']
    if all(col in df.columns for col in time_cols):
        df['total_time_consistency'] = np.where(
            (df[time_cols].sum(axis=1) > 24 * 60),
            'invalid_exceeds_24hrs',
            'valid'
        )

    # 3. 年龄限制：PAQ706 仅2-11岁，成人问题仅12+岁 (假设 RIDAGEYR 存在)
    if 'RIDAGEYR' in df.columns:
        if 'PAQ706' in df.columns:
            df['PAQ706_consistency'] = np.where(
                (pd.notna(df['RIDAGEYR'])) & ((df['RIDAGEYR'] < 2) | (df['RIDAGEYR'] > 11)) & (pd.notna(df['PAQ706'])),
                'invalid_age',
                'valid'
            )
        adult_cols = ['PAQ605', 'PAQ610', 'PAD615', 'PAQ620', 'PAQ625', 'PAD630', 'PAQ635', 'PAQ640', 'PAD645', 'PAQ650', 'PAQ655', 'PAD660', 'PAQ665', 'PAQ670', 'PAD675', 'PAD680']
        for col in adult_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < 12) & (pd.notna(df[col])),
                    'invalid_age_under_12',
                    'valid'
                )

    # 衍生变量：计算总MET分数 (使用附录1 MET值)
    met_values = {
        'PAD615': 8.0,  # 剧烈工作
        'PAD630': 4.0,  # 中等工作
        'PAD645': 4.0,  # 交通
        'PAD660': 8.0,  # 剧烈休闲
        'PAD675': 4.0   # 中等休闲
    }
    if all(col in df.columns for col in met_values.keys()):
        df['total_met_score'] = sum(df[col] * met_values[col] / 60 * df.get(col.replace('PAD', 'PAQ'), 1) for col in met_values)  # 假设天数为1如果缺失

    # 统计缺失值
    paq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if paq_cols:
        df['missing_paq_count'] = df[paq_cols].isna().sum(axis=1)
        df['all_paq_missing'] = df['missing_paq_count'] == len(paq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 PAQ 数据
    df = clean_paq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_paq_files(root_dir='.'):
    # 遍历目录，处理 PAQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('PAQ_') or file.startswith('P_PAQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 PAQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_paq_files()
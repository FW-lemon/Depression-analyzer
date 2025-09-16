import pandas as pd
import numpy as np
import os

def clean_sxq_data(df):
    # 定义有效值范围（基于 SXQ_G 文档）
    valid_ranges = {
        'SXD021': {1, 2},  # 是否有过阴道/肛门/口交
        'SXQ800': {1, 2, 7, 9},  # 是否与女性有过阴道性交
        'SXQ803': {1, 2, 7, 9},  # 是否对女性进行过口交
        'SXQ806': {1, 2, 7, 9},  # 是否与女性有过肛交
        'SXQ809': {1, 2, 7, 9},  # 是否与男性有过任何性行为
        'SXQ700': {1, 2, 7, 9},  # 是否与男性有过阴道性交
        'SXQ703': {1, 2, 7, 9},  # 是否对男性进行过口交
        'SXQ706': {1, 2, 7, 9},  # 是否接受过男性口交
        'SXQ709': {1, 2, 7, 9},  # 是否与男性有过肛交
        'SXD031': set(range(0, 122)) | {777, 999},  # 第一次性交年龄
        'SXD171': set(range(0, 1001)) | {7777, 9999},  # 终身女性性伴侣数
        'SXD510': set(range(0, 1001)) | {7777, 9999},  # 终身男性性伴侣数
        'SXQ824': set(range(0, 1001)) | {7777, 9999},  # 过去12个月女性性伴侣数
        'SXQ827': set(range(0, 1001)) | {7777, 9999},  # 过去12个月男性性伴侣数
        'SXD633': set(range(0, 122)) | {777, 999},  # 第一次对女性口交年龄
        'SXQ636': set(range(0, 1001)) | {7777, 9999},  # 终身对女性口交人数
        'SXQ639': set(range(0, 1001)) | {7777, 9999},  # 过去12个月对女性口交人数
        'SXD642': set(range(0, 1001)) | {7777, 9999},  # 过去12个月新女性口交伴侣数
        'SXQ410': set(range(0, 1001)) | {7777, 9999},  # 终身对男性口交人数
        'SXQ550': set(range(0, 1001)) | {7777, 9999},  # 过去12个月对男性口交人数
        'SXQ836': set(range(0, 1001)) | {7777, 9999},  # 过去12个月新男性口交伴侣数
        'SXQ841': set(range(0, 1001)) | {7777, 9999},  # 过去12个月接受口交人数
        'SXQ853': set(range(0, 1001)) | {7777, 9999},  # 过去12个月新接受口交伴侣数
        'SXD621': set(range(0, 122)) | {777, 999},  # 第一次接受口交年龄
        'SXQ624': set(range(0, 1001)) | {7777, 9999},  # 终身接受口交人数
        'SXQ627': set(range(0, 1001)) | {7777, 9999},  # 过去12个月接受口交人数
        'SXD630': set(range(0, 1001)) | {7777, 9999},  # 过去12个月新接受口交伴侣数
        'SXQ645': {1, 2, 3, 4, 7, 9},  # 过去12个月口交使用保护频率
        'SXQ648': {1, 2, 7, 9},  # 过去12个月是否有新性伴侣
        'SXQ610': set(range(0, 1001)) | {7777, 9999},  # 过去12个月肛交人数
        'SXQ251': {1, 2, 3, 4, 5, 7, 9},  # 过去12个月无套阴道/肛交频率
        'SXQ590': set(range(0, 1001)) | {7777, 9999},  # 过去12个月5岁以上伴侣数
        'SXQ600': set(range(0, 1001)) | {7777, 9999},  # 过去12个月5岁以下伴侣数
        'SXD101': set(range(0, 1001)) | {7777, 9999},  # 终身性伴侣数
        'SXD450': set(range(0, 1001)) | {7777, 9999},  # 过去12个月性伴侣数
        'SXQ724': {1, 2, 3, 7, 9},  # 性取向
        'SXQ727': {1, 2, 3, 7, 9},  # 性吸引
        'SXQ130': {1, 2, 3, 7, 9},  # 包茎情况
        'SXQ490': {1, 2, 7, 9},  # 是否包皮环切
        'SXQ741': {1, 2, 3, 4, 5, 7, 9},  # 性伴侣类型
        'SXQ753': {1, 2, 7, 9},  # 是否有 HPV
        'SXQ260': {1, 2, 7, 9},  # 是否有生殖器疱疹
        'SXQ265': {1, 2, 7, 9},  # 是否有生殖器疣
        'SXQ267': {1, 2, 7, 9},  # 是否有梅毒
        'SXQ270': {1, 2, 7, 9},  # 过去12个月是否有淋病
        'SXQ272': {1, 2, 7, 9},  # 过去12个月是否有衣原体
        'SXQ280': {1, 2, 7, 9},  # 是否有 HIV
        'SXQ295': {1, 2, 7, 9},  # 是否有艾滋病
        'SXQ296': {1, 2, 7, 9}  # 是否接受 HIV 治疗
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 SXD021 = 2（从未有过性行为），所有性行为相关列应为空
    if 'SXD021' in df.columns:
        sex_cols = ['SXQ800', 'SXQ803', 'SXQ806', 'SXQ809', 'SXQ700', 'SXQ703', 'SXQ706', 'SXQ709', 'SXD031', 'SXD171', 'SXD510', 'SXQ824', 'SXQ827', 'SXD633', 'SXQ636', 'SXQ639', 'SXD642', 'SXQ410', 'SXQ550', 'SXQ836', 'SXQ841', 'SXQ853', 'SXD621', 'SXQ624', 'SXQ627', 'SXD630', 'SXQ645', 'SXQ648', 'SXQ610', 'SXQ251', 'SXQ590', 'SXQ600', 'SXD101', 'SXD450', 'SXQ741']
        for col in sex_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['SXD021'])) & (df['SXD021'] == 2) & (pd.notna(df[col])),
                    'invalid_no_sexual_experience',
                    'valid'
                )

    # 2. 第一次性交年龄 (SXD031) 不能大于当前年龄 (RIDAGEYR)
    if 'SXD031' in df.columns and 'RIDAGEYR' in df.columns:
        df['SXD031_consistency'] = np.where(
            (pd.notna(df['SXD031'])) & (~df['SXD031'].isin({777, 999})) & (df['SXD031'] > df['RIDAGEYR']),
            'invalid_exceeds_age',
            'valid'
        )

    # 3. 终身伴侣数应大于等于过去12个月伴侣数
    lifetime_cols = ['SXD171', 'SXD510', 'SXQ636', 'SXQ410', 'SXQ550', 'SXQ836', 'SXQ841', 'SXQ624', 'SXQ627', 'SXD630', 'SXD101', 'SXD450']
    past_year_cols = ['SXQ824', 'SXQ827', 'SXQ639', 'SXQ410', 'SXQ550', 'SXQ836', 'SXQ841', 'SXQ624', 'SXQ627', 'SXD630', 'SXQ610', 'SXQ251', 'SXQ590', 'SXQ600']
    for lt, py in zip(lifetime_cols, past_year_cols):
        if lt in df.columns and py in df.columns:
            df[f'{py}_consistency'] = np.where(
                (pd.notna(df[lt])) & (~df[lt].isin({7777, 9999})) &
                (pd.notna(df[py])) & (~df[py].isin({7777, 9999})) & (df[lt] < df[py]),
                'invalid_lifetime_less_than_past_year',
                'valid'
            )

    # 4. 如果 SXQ648 = 2 (无新伴侣)，新伴侣相关列应为空
    if 'SXQ648' in df.columns:
        new_partner_cols = ['SXD642', 'SXQ836', 'SXQ853', 'SXD630']
        for col in new_partner_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['SXQ648'])) & (df['SXQ648'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_new_partners',
                    'valid'
                )

    # 5. STD 历史：如果否定某些疾病，相关变量应一致
    if 'SXQ260' in df.columns:  # 生殖器疱疹
        df['SXQ260_consistency'] = np.where(
            (pd.notna(df['SXQ260'])) & (df['SXQ260'].isin({2, 7, 9})) & (pd.notna(df.get('SXQ262', pd.NA))),
            'invalid_no_herpes',
            'valid'
        )

    # 统计缺失值
    sxq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if sxq_cols:
        df['missing_sxq_count'] = df[sxq_cols].isna().sum(axis=1)
        df['all_sxq_missing'] = df['missing_sxq_count'] == len(sxq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 SXQ 数据
    df = clean_sxq_data(df)
    
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

def process_sxq_files(root_dir='.'):
    # 遍历目录，处理 SXQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('SXQ_') or file.startswith('P_SXQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 SXQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_sxq_files()
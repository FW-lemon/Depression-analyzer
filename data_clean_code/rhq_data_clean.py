import pandas as pd
import numpy as np
import os

def clean_rhq_data(df):
    # 定义有效值范围（基于 RHQ_G 文档）
    valid_ranges = {
        'RHQ010': set(range(0, 121)) | {777, 999},  # 第一次月经年龄
        'RHQ020': {1, 2, 3, 4, 7, 9},  # 第一次月经年龄范围
        'RHQ031': {1, 2, 7, 9},  # 是否有规律月经
        'RHD042': {1, 2, 7, 9},  # 过去12个月是否有月经
        'RHQ060': set(range(1, 76)) | {777, 999},  # 上次月经年龄
        'RHQ070': {1, 2, 7, 9},  # 上次月经年龄范围
        'RHQ076': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 77, 99},  # 上次月经原因
        'RHQ078': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 77, 99},  # 没有月经原因
        'RHD143': {1, 2, 3, 7, 9},  # 当前怀孕状态
        'RHQ160': {1, 2, 7, 9},  # 是否曾经怀孕
        'RHQ162': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 怀孕次数
        'RHQ163': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 活产次数
        'RHQ166': {1, 2, 7, 9},  # 当前哺乳
        'RHQ169': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 流产次数
        'RHQ171': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 活产婴儿数
        'RHQ172': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 多胎次数
        'RHD173': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 早产次数
        'RHD180': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 死产次数
        'RHD190': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 流产次数
        'RHQ197': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 管婴儿次数
        'RHQ200': {1, 2, 7, 9},  # 妊娠糖尿病
        'RHQ205': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 99},  # 妊娠糖尿病次数
        'RHQ270': {1, 2, 7, 9},  # 卵巢移除
        'RHQ280A': {1, 2, 7, 9},  # 卵巢移除 - 右卵巢
        'RHQ280B': {1, 2, 7, 9},  # 卵巢移除 - 左卵巢
        'RHQ280C': {1, 2, 7, 9},  # 卵巢移除 - 两个
        'RHQ280D': {1, 2, 7, 9},  # 卵巢移除 - 未知
        'RHQ291': set(range(0, 80)) | {777, 999},  # 卵巢移除年龄
        'RHQ300': {1, 2, 7, 9},  # 子宫切除
        'RHQ305': set(range(0, 80)) | {777, 999},  # 子宫切除年龄
        'RHQ420': {1, 2, 7, 9},  # 是否使用避孕药
        'RHQ430': set(range(0, 80)) | {777, 999},  # 开始使用避孕药年龄
        'RHQ440': {1, 2, 7, 9},  # 现在使用避孕药
        'RHQ450': set(range(0, 80)) | {777, 999},  # 停止使用避孕药年龄
        'RHQ460Q': set(range(0, 80)) | {777, 999},  # 使用避孕药时长
        'RHQ460U': {1, 2, 7, 9},  # 时长单位
        'RHQ510': {1, 2, 7, 9},  # 使用Depo-Provera或注射剂
        'RHQ520': {1, 2, 7, 9},  # 现在使用Depo-Provera或注射剂
        'RHQ540': {1, 2, 7, 9},  # 是否使用女性激素
        'RHQ541A': {1},  # 使用激素药片
        'RHQ541B': {2},  # 使用激素贴片
        'RHQ541C': {3},  # 使用激素霜/栓剂/注射
        'RHQ550': {1, 2, 7, 9},  # 开始激素时是否有月经
        'RHQ551A': {1},  # 使用雌激素/孕激素 - 更年期
        'RHQ551B': {2},  # 使用雌激素/孕激素 - 情绪
        'RHQ551C': {3},  # 使用雌激素/孕激素 - 子宫切除/卵巢切除
        'RHQ551D': {4},  # 使用雌激素/孕激素 - 骨质疏松
        'RHQ551E': {5},  # 使用雌激素/孕激素 - 心血管疾病
        'RHQ551F': {6},  # 使用雌激素/孕激素 - 不规则月经
        'RHQ551G': {7},  # 使用雌激素/孕激素 - 其他原因
        'RHQ554': {1, 2, 7, 9},  # 使用仅雌激素药片
        'RHQ556': set(range(0, 80)) | {777, 999},  # 开始仅雌激素药片年龄
        'RHQ558': {1, 2, 7, 9},  # 现在使用仅雌激素药片
        'RHQ560Q': set(range(0, 80)) | {777, 999},  # 使用仅雌激素药片时长
        'RHQ560U': {1, 2, 7, 9},  # 时长单位
        'RHQ562': {1, 2, 7, 9},  # 使用仅孕激素药片
        'RHQ564': set(range(0, 80)) | {777, 999},  # 开始仅孕激素药片年龄
        'RHQ566': {1, 2, 7, 9},  # 现在使用仅孕激素药片
        'RHQ568Q': set(range(0, 80)) | {777, 999},  # 使用仅孕激素药片时长
        'RHQ568U': {1, 2, 7, 9},  # 时长单位
        'RHQ570': {1, 2, 7, 9},  # 使用雌激素/孕激素组合药片
        'RHQ572': set(range(0, 80)) | {777, 999},  # 开始组合药片年龄
        'RHQ574': {1, 2, 7, 9},  # 现在使用组合药片
        'RHQ576Q': set(range(0, 80)) | {777, 999},  # 使用组合药片时长
        'RHQ576U': {1, 2, 7, 9},  # 时长单位
        'RHQ580': {1, 2, 7, 9},  # 使用仅雌激素贴片
        'RHQ582': set(range(0, 80)) | {777, 999},  # 开始仅雌激素贴片年龄
        'RHQ584': {1, 2, 7, 9},  # 现在使用仅雌激素贴片
        'RHQ586Q': set(range(0, 80)) | {777, 999},  # 使用仅雌激素贴片时长
        'RHQ586U': {1, 2, 7, 9},  # 时长单位
        'RHQ596': {1, 2, 7, 9},  # 使用雌激素/孕激素组合贴片
        'RHQ598': set(range(0, 80)) | {777, 999},  # 开始组合贴片年龄
        'RHQ600': {1, 2, 7, 9},  # 现在使用组合贴片
        'RHQ602Q': set(range(0, 80)) | {777, 999},  # 使用组合贴片时长
        'RHQ602U': {1, 2, 7, 9},  # 时长单位
        'RHQ700': {1, 2, 7, 9},  # 过去一个月使用女性卫生产品
        'RHQ710A': {1},  # 使用卫生棉条
        'RHQ710B': {2},  # 使用卫生巾
        'RHQ710C': {3},  # 使用阴道冲洗
        'RHQ710D': {4},  # 使用女性喷雾
        'RHQ710E': {5},  # 使用女性粉末
        'RHQ710F': {6},  # 使用女性湿巾
        'RHQ710G': {7},  # 使用其他女性卫生产品
        'RHQ720': {1, 2, 7, 9},  # 过去6个月阴道冲洗
        'RHQ730': {1, 2, 3, 4, 5, 6, 7, 9},  # 过去6个月阴道冲洗频率
        'RHQ740': {1, 2, 7, 9},  # 过去一个月阴道问题
        'RHQ750A': {1},  # 阴道瘙痒
        'RHQ750B': {2},  # 阴道异味
        'RHQ750C': {3}  # 阴道分泌物
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 月经年龄不能大于当前年龄（假设 RIDAGEYR 存在）
    if 'RIDAGEYR' in df.columns and 'RHQ010' in df.columns:
        df['RHQ010_consistency'] = np.where(
            (pd.notna(df['RHQ010'])) & (df['RHQ010'] > 0) & (df['RHQ010'] > df['RIDAGEYR']),
            'invalid_exceeds_age',
            'valid'
        )

    # 2. 如果 RHQ010 = 0 (未开始月经)，月经相关问题应为空
    if 'RHQ010' in df.columns:
        period_cols = ['RHQ031', 'RHD042', 'RHQ060', 'RHQ070', 'RHQ076', 'RHQ078']
        for col in period_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RHQ010'])) & (df['RHQ010'] == 0) & (pd.notna(df[col])),
                    'invalid_no_periods',
                    'valid'
                )

    # 3. 怀孕次数一致性：RHQ162 (总怀孕) >= RHQ163 (活产) + RHQ169 (流产) + RHQ171 (活产婴儿) 等
    if all(col in df.columns for col in ['RHQ162', 'RHQ163', 'RHQ169', 'RHQ172', 'RHD173', 'RHD180', 'RHD190', 'RHQ197']):
        df['pregnancy_consistency'] = np.where(
            (pd.notna(df['RHQ162'])) & (~df['RHQ162'].isin({77, 99})) &
            (df['RHQ162'] >= df['RHQ163'] + df['RHQ169'] + df['RHQ172'] + df['RHD173'] + df['RHD180'] + df['RHD190'] + df['RHQ197']),
            'valid',
            'invalid_count_mismatch'
        )

    # 4. 如果 RHQ160 = 2 (从未怀孕)，怀孕相关问题应为空
    if 'RHQ160' in df.columns:
        preg_cols = ['RHQ162', 'RHQ163', 'RHQ166', 'RHQ169', 'RHQ171', 'RHQ172', 'RHD173', 'RHD180', 'RHD190', 'RHQ197', 'RHQ200', 'RHQ205']
        for col in preg_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RHQ160'])) & (df['RHQ160'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_pregnancy',
                    'valid'
                )

    # 5. 如果 RHD143 = 1 (当前怀孕)，某些激素使用问题应考虑
    if 'RHD143' in df.columns:
        hormone_cols = ['RHQ420', 'RHQ430', 'RHQ440', 'RHQ450', 'RHQ460Q', 'RHQ460U', 'RHQ510', 'RHQ520', 'RHQ540']
        for col in hormone_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RHD143'])) & (df['RHD143'] == 1) & (pd.notna(df[col])),
                    'check_pregnant_usage',
                    'valid'
                )

    # 统计缺失值
    rhq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if rhq_cols:
        df['missing_rhq_count'] = df[rhq_cols].isna().sum(axis=1)
        df['all_rhq_missing'] = df['missing_rhq_count'] == len(rhq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 RHQ 数据
    df = clean_rhq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_rhq_files(root_dir='.'):
    # 遍历目录，处理 RHQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('RHQ_') or file.startswith('P_RHQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 RHQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_rhq_files()
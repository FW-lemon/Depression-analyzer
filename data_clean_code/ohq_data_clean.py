import pandas as pd
import numpy as np
import os

def clean_ohq_data(df):
    # 定义有效值范围（基于 OHQ_G 文档）
    valid_ranges = {
        'OHQ030': {1, 2, 3, 4, 5, 6, 7, 77, 99},  # 上次看牙医时间
        'OHQ033': {1, 2, 3, 4, 5, 7, 9},  # 上次看牙医原因
        'OHQ770': {1, 2, 7, 9},  # 过去12个月需要但未得到牙科护理
        'OHQ780A': {10, 77, 99},  # 原因: 无法负担费用
        'OHQ780B': {11},  # 原因: 保险不覆盖
        'OHQ780C': {12},  # 原因: 牙医不接受保险类型
        'OHQ780D': {13},  # 原因: 没有交通工具
        'OHQ780E': {14},  # 原因: 办公室太远
        'OHQ780F': {15},  # 原因: 营业时间不方便
        'OHQ780G': {16},  # 原因: 预计问题会消失
        'OHQ780H': {17},  # 原因: 无法从工作中抽时间
        'OHQ780I': {18},  # 原因: 等待预约时间太长
        'OHQ780J': {19},  # 原因: 无法找到接受医疗补助的牙医
        'OHQ780K': {20},  # 原因: 其他
        'OHQ835': {1, 2, 3, 4, 5, 7, 9},  # 牙龈健康状况
        'OHQ845': {1, 2, 3, 4, 5, 7, 9},  # 牙齿健康状况
        'OHQ850': {1, 2, 7, 9},  # 牙龈疾病治疗
        'OHQ855': {1, 2, 7, 9},  # 牙齿松动
        'OHQ860': {1, 2, 7, 9},  # 骨质流失
        'OHQ865': {1, 2, 7, 9},  # 牙齿外观异常
        'OHQ870': set(range(0, 8)) | {77, 99},  # 使用牙线天数
        'OHQ875': set(range(0, 8)) | {77, 99},  # 使用漱口水天数
        'OHQ880': {1, 2, 7, 9},  # 口腔癌检查 - 拉舌头
        'OHQ885': {1, 2, 7, 9},  # 口腔癌检查 - 摸脖子
        'OHQ895': {1, 2, 3, 7, 9},  # 最近口腔癌检查时间
        'OHQ900': {1, 2, 3, 4, 5, 7, 9}  # 检查专业人士类型
    }

    # 验证数据范围
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # 一致性检查
    # 1. 如果 OHQ030 = 7 (从未看过牙医)，OHQ033, OHQ770 等应为空
    if 'OHQ030' in df.columns:
        visit_cols = ['OHQ033', 'OHQ770']
        for col in visit_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OHQ030'])) & (df['OHQ030'] == 7) & (pd.notna(df[col])),
                    'invalid_never_visited',
                    'valid'
                )

    # 2. 如果 OHQ770 ≠ 1 (不需要或未得到护理)，OHQ780 系列应为空
    if 'OHQ770' in df.columns:
        reason_cols = ['OHQ780A', 'OHQ780B', 'OHQ780C', 'OHQ780D', 'OHQ780E', 'OHQ780F', 'OHQ780G', 'OHQ780H', 'OHQ780I', 'OHQ780J', 'OHQ780K']
        for col in reason_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OHQ770'])) & (df['OHQ770'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_need',
                    'valid'
                )

    # 3. 年龄限制检查：OHQ835 到 OHQ900 仅适用于30+岁（假设 RIDAGEYR 存在，若无移除）
    if 'RIDAGEYR' in df.columns:
        adult_cols = ['OHQ835', 'OHQ845', 'OHQ850', 'OHQ855', 'OHQ860', 'OHQ865', 'OHQ870', 'OHQ875', 'OHQ880', 'OHQ885', 'OHQ895', 'OHQ900']
        for col in adult_cols:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['RIDAGEYR'])) & (df['RIDAGEYR'] < 30) & (pd.notna(df[col])),
                    'invalid_age_under_30',
                    'valid'
                )

    # 4. 如果 OHQ880 和 OHQ885 均为2,7,9，OHQ895 和 OHQ900 应为空
    if 'OHQ880' in df.columns and 'OHQ885' in df.columns:
        for col in ['OHQ895', 'OHQ900']:
            if col in df.columns:
                df[f'{col}_consistency'] = np.where(
                    (pd.notna(df['OHQ880'])) & (df['OHQ880'].isin({2, 7, 9})) &
                    (pd.notna(df['OHQ885'])) & (df['OHQ885'].isin({2, 7, 9})) & (pd.notna(df[col])),
                    'invalid_no_cancer_exam',
                    'valid'
                )

    # 统计缺失值
    ohq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if ohq_cols:
        df['missing_ohq_count'] = df[ohq_cols].isna().sum(axis=1)
        df['all_ohq_missing'] = df['missing_ohq_count'] == len(ohq_cols)

    return df

def load_data(file_path):
    # 读取 CSV，将 5.397605346934028e-79 和空字段视为 NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 清洗 OHQ 数据
    df = clean_ohq_data(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

def process_ohq_files(root_dir='.'):
    # 遍历目录，处理 OHQ 文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('OHQ_') or file.startswith('P_OHQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理 OHQ 文件: {input_file}")
                clean_data(input_file, output_file)

if __name__ == "__main__":
    process_ohq_files()
import pandas as pd
import numpy as np
import os

# 步骤1：加载数据
def load_data(file_path):
    # 读取CSV，将5.397605346934028e-79和空字段视为NaN
    # 注意：根据样本数据，可能需要调整na_values以匹配实际缺失表示
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

# 步骤2：验证数据范围
def validate_ranges(df):
    # DPQ010-DPQ090有效值：{0, 1, 2, 3, 7, 9}
    # DPQ100有效值：{0, 1, 2, 3, 7, 9}
    symptom_cols = [f'DPQ0{i}0' for i in range(1, 10)]  # DPQ010至DPQ090
    valid_values = {0, 1, 2, 3, 7, 9}
    
    for col in symptom_cols + ['DPQ100']:
        if col in df.columns:
            # 转换为数值，无效值转为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 标记不在有效值集合中的值
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)
    
    return df

# 步骤3：处理缺失数据
def handle_missing(df):
    symptom_cols = [f'DPQ0{i}0' for i in range(1, 10)]
    valid_symptom_cols = [col for col in symptom_cols if col in df.columns]
    
    if valid_symptom_cols:
        # 计算每行症状问题的缺失值数量
        df['missing_symptom_count'] = df[valid_symptom_cols].isna().sum(axis=1)
        
        # 标记所有症状问题均缺失的行
        df['all_symptoms_missing'] = df['missing_symptom_count'] == len(valid_symptom_cols)
        
        # 检查DPQ100是否应存在（至少一个症状>0）
        if 'DPQ100' in df.columns:
            df['any_symptom_endorsed'] = df[valid_symptom_cols].gt(0).any(axis=1)
            df['dpq100_validity'] = np.where(
                df['any_symptom_endorsed'] & df['DPQ100'].isna(),
                'missing_when_expected',
                'valid'
            )
    
    return df

# 步骤4：计算总分
def calculate_total_score(df):
    symptom_cols = [f'DPQ0{i}0' for i in range(1, 10)]
    valid_symptom_cols = [col for col in symptom_cols if col in df.columns]
    
    if valid_symptom_cols:
        # 仅当所有症状问题非缺失且不含拒绝回答（7）或不知道（9）时计算总分
        df['total_score'] = df[valid_symptom_cols].apply(
            lambda x: x.sum() if all(x.notna()) and all(x.isin({0, 1, 2, 3})) else np.nan,
            axis=1
        )
    return df

# 步骤5：主清洗函数
def clean_phq9_data(file_path, output_path):
    # 加载数据
    df = load_data(file_path)
    
    # 验证范围
    df = validate_ranges(df)
    
    # 处理缺失数据
    df = handle_missing(df)
    
    # 计算总分
    df = calculate_total_score(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")
    
    return df

# 主函数：遍历所有年份文件夹，处理DPQ相关的CSV文件
def process_all_dpq_files(root_dir='.'):
    # 遍历根目录下的所有子目录
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 匹配DPQ_*.csv 或 P_DPQ.csv 等文件（忽略已清洗的文件）
            if file.startswith('DPQ_') and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理文件: {input_file}")
                clean_phq9_data(input_file, output_file)
            elif file.startswith('P_DPQ') and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"处理文件: {input_file}")
                clean_phq9_data(input_file, output_file)

# 示例用法：假设脚本运行在项目根目录
if __name__ == "__main__":
    process_all_dpq_files()
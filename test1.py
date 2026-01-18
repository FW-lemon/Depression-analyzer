import pandas as pd

# 读取原始文件
input_file = 'data/GSE102556.tsv'
output_file = 'data/GSE102556_clean_59samples.tsv'

df = pd.read_csv(input_file, sep='\t', index_col=0)

print("原始数据形状:", df.shape)  # 应该显示 (genes, 86)

# 保留前59列（根据 GEO 信息：29 control + 30 MDD）
clean_df = df.iloc[:, :59]

# 快速检查：非零基因数统计（前59 vs 原始）
print("\n前59列 非零基因数统计：")
print((clean_df > 0).sum(axis=0).describe())

# 保存为新文件（保持 tab 分隔，基因作为 index）
clean_df.to_csv(output_file, sep='\t')

print(f"\n清理完成！文件已保存到: {output_file}")
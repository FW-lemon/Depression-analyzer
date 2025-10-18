import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 读取合并后的数据
df = pd.read_csv('/home/project/yihao/total_data.csv')

# 1. 删除缺失比例超过60%的行
threshold = 0.6
df = df[df.isna().mean(axis=1) <= threshold]

# 2. KNN 填补缺失值
imputer = KNNImputer(n_neighbors=5)  # 最近5个个体
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 如果想保持整数型，可以四舍五入
df_filled = df_filled.round(0)

# 保存处理后的文件
df_filled.to_csv('merged_filled_knn.csv', index=False)

print("KNN填补完成，结果已保存为 merged_filled_knn.csv")
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 创建结果文件夹 ==================
out_dir = "ML-result"
os.makedirs(out_dir, exist_ok=True)

# ================== 2. 读取数据 ==================
tpm_file = "data/GSE101521_TPM_39genes.tsv"
df = pd.read_csv(tpm_file, sep='\t', index_col=0)

num_control = 29
num_samples = df.shape[1]

y = np.array([0]*num_control + [1]*(num_samples - num_control))
X = df.T.values
genes = df.index.tolist()

# ================== 3. 标准化 ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== 4. 随机森林训练 ==================
rf = RandomForestClassifier(n_estimators=500, random_state=42, oob_score=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
print(f"5折交叉验证准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f}")

rf.fit(X_scaled, y)
print(f"OOB score: {rf.oob_score_:.4f}")

# ================== 5. 特征重要性 ==================
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({'Gene': genes, 'Importance': rf_importances})
rf_df = rf_df.sort_values('Importance', ascending=True)  # 为条形图倒序显示

# 保存结果
rf_df.to_csv(os.path.join(out_dir, "RF_importance_39genes.csv"), index=False)
print(f"✅ 基因重要性已保存到 {out_dir}/RF_importance_39genes.csv")

# ================== 6. 美化绘图 ==================
plt.figure(figsize=(10,8))
sns.set_style("whitegrid")

# 条形图
bar = sns.barplot(
    x='Importance', 
    y='Gene', 
    data=rf_df, 
    palette=sns.color_palette("viridis", n_colors=len(rf_df))
)

# 添加数值标签
for idx, val in enumerate(rf_df['Importance']):
    bar.text(val + 0.001, idx, f"{val:.3f}", color='black', va='center', fontsize=9)

plt.title('Random Forest Feature Importance (39 genes)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Gene', fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig(os.path.join(out_dir, "RF_importance_39genes.png"), dpi=300)
plt.show()

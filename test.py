import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 1. 加载数据
file_path = '/home/project/yihao/data/GSE179265_TPM.tsv'
df = pd.read_csv(file_path, sep='\t', index_col=0)

# 2. 定义目标基因
target_genes = ['BAG4', 'BTBD1', 'CNOT6L', 'ZNF22', 'PWWP2B', 'SINHCAF', 'PDE7A', 'SCOC', 'MYBBP1A']
available_genes = [g for g in target_genes if g in df.index]
print(f"✅ 匹配到的基因: {available_genes}")

# 3. 提取数据并【根据官方 Metadata 强制分组】
data = df.loc[available_genes].T

# 根据 GSE179265 官方定义：前 7 个是 Control，后 17 个是 Patient
# 也可以通过 ID 范围来判断，更加保险
control_ids = [f'GSM541274{i}' for i in range(5, 10)] + ['GSM5412750', 'GSM5412751']
# 或者直接按顺序：
groups = []
for i, col in enumerate(data.index):
    # GSM5412745 到 GSM5412751 是 Control
    num = int(''.join(filter(str.isdigit, col)))
    if 5412745 <= num <= 5412751:
        groups.append('Control')
    else:
        groups.append('Patient')

data['Group'] = groups
print(f"📊 分组确认: {data['Group'].value_counts().to_dict()}")

# 4. 绘图环境设置 (使用通用字体)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 验证 A: 表达一致性 (箱线图) ---
plt.figure(figsize=(15, 10))
for i, gene in enumerate(available_genes):
    plt.subplot(3, 4, i+1)
    # 修复了 seaborn 的 palette 警告
    sns.boxplot(x='Group', y=gene, data=data, hue='Group', palette='Set2', legend=False)
    sns.stripplot(x='Group', y=gene, data=data, color='black', alpha=0.3)
    
    ctrl = data[data['Group']=='Control'][gene]
    pat = data[data['Group']=='Patient'][gene]
    
    # 计算 T 检验 P 值
    _, p_val = stats.ttest_ind(ctrl, pat)
    plt.title(f'{gene}\nP-val: {p_val:.4f}')

plt.suptitle('Validation A: Expression Level (GSE179265)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/project/yihao/boxplots.png')
print("💾 表达量对比图已保存: boxplots.png")

# --- 验证 B: 诊断能力 (ROC) ---
y_true = (data['Group'] == 'Patient').astype(int)
X = data[available_genes]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(8, 6))
# 组合模型
lr = LogisticRegression(solver='liblinear')
lr.fit(X_scaled, y_true)
y_score = lr.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, color='red', lw=3, label=f'Combined (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation B: ROC Curve (GSE179265)')
plt.legend()
plt.savefig('/home/project/yihao/roc_curve.png')
print("💾 ROC 曲线图已保存: roc_curve.png")

# --- 验证 C: 样本聚类 (PCA) ---
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
pca_df['Group'] = data['Group'].values

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Group', data=pca_df, s=150, palette='Set1', edgecolor='w')
plt.title('Validation C: PCA Plot (GSE179265)')
plt.savefig('/home/project/yihao/pca_plot.png')
print("💾 PCA 聚类图已保存: pca_plot.png")

print("\n🏁 分析全部完成！请检查 /home/project/yihao/ 目录下的 png 文件。")
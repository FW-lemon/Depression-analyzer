import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os

# ====================== 文件路径 ======================
wgcna_file = 'WGCNA_results_top8000/Key_Module_green_Genes.csv'

deg_files = {
    'deseq2_GSE101521_loose': 'DEG_results/deseq2_results/DEGs_loose.csv',
    'deseq2_GSE101521_strict': 'DEG_results/deseq2_results/DEGs_strict.csv',
    'limma_GSE101521_loose': 'DEG_results/limma_results/DEGs_loose.csv',
    'limma_GSE101521_strict': 'DEG_results/limma_results/DEGs_strict.csv'
}


# ====================== 输出文件夹 ======================
output_dir = 'WGCNA_DEG_overlap_results'
os.makedirs(output_dir, exist_ok=True)

# ====================== 读取 WGCNA 绿色模块基因 ======================
wgcna_df = pd.read_csv(wgcna_file)
# 取 gene 列
if 'gene' in wgcna_df.columns:
    wgcna_genes = set(wgcna_df['gene'])
elif 'Gene' in wgcna_df.columns:
    wgcna_genes = set(wgcna_df['Gene'])
else:
    wgcna_genes = set(wgcna_df.iloc[:,0])

# ====================== 创建 2x2 子图 ======================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, path) in enumerate(deg_files.items()):
    deg_df = pd.read_csv(path)
    # 取 gene 列
    if 'gene' in deg_df.columns:
        deg_genes = set(deg_df['gene'])
    elif 'Gene' in deg_df.columns:
        deg_genes = set(deg_df['Gene'])
    else:
        deg_genes = set(deg_df.iloc[:,0])
    
    # 计算交集
    overlap = wgcna_genes & deg_genes
    print(f"{name} 重叠基因数: {len(overlap)}")
    
    # 保存重叠基因到 CSV
    if len(overlap) > 0:
        overlap_df = pd.DataFrame({'gene': list(overlap)})
        overlap_csv_path = os.path.join(output_dir, f'overlap_WGCNA_{name}.csv')
        overlap_df.to_csv(overlap_csv_path, index=False)
        print(f"重叠基因已保存到: {overlap_csv_path}")

    # 画 Venn 图
    venn2([wgcna_genes, deg_genes], set_labels=('WGCNA green', name), ax=axes[i])
    axes[i].set_title(f'{name}\nOverlap: {len(overlap)}')

# 保存整张 2x2 图
fig_path = os.path.join(output_dir, 'WGCNA_vs_DEGs_4plots.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"四个 Venn 图已保存到: {fig_path}")

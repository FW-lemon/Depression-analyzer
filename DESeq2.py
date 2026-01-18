import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore

# ====================== 配置 ======================
INPUT_FILE = 'data/GSE101521_clean_59samples.tsv'
OUTPUT_DIR = 'deg_GSE101521'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CONTROL = 29
N_MDD = 30

print("🎯 GSE101521 MDD vs Control 分析 - 热图最终修复版")

# 1. 加载 & 去重（保持原样）
print("\n加载数据...")
counts = pd.read_csv(INPUT_FILE, sep='\t', index_col=0).astype(int)
print(f"原始基因数: {len(counts):,}")

duplicated_mask = counts.index.duplicated(keep=False)
print(f"重复位置: {duplicated_mask.sum():,}")

if duplicated_mask.any():
    print("去重中...")
    row_sums = counts.sum(axis=1)
    selected_indices = []
    for gene, grp in row_sums.groupby(counts.index):
        if len(grp) == 1:
            selected_indices.append(grp.index[0])
        else:
            selected_indices.append(grp.idxmax())
    counts = counts.loc[selected_indices]

if counts.index.duplicated().any():
    print("强制唯一化 index...")
    seen = {}
    new_index = []
    for name in counts.index:
        if name in seen:
            seen[name] += 1
            new_index.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_index.append(name)
    counts.index = new_index

print(f"最终基因数: {len(counts):,}")
print(f"index 唯一: {counts.index.is_unique}")

# 2. Metadata
samples = counts.columns.tolist()
condition = ['control'] * N_CONTROL + ['MDD'] * N_MDD
metadata = pd.DataFrame({'condition': condition}, index=samples)

# 3. DESeq2（保持原样）
print("\n运行 DESeq2...")
dds = DeseqDataSet(
    counts=counts.T,
    metadata=metadata,
    design_factors='condition',
    refit_cooks=True
)
dds.deseq2()

# 4. 差异结果
stat_res = DeseqStats(dds, contrast=['condition', 'MDD', 'control'])
stat_res.summary()

results = stat_res.results_df
results['gene'] = results.index
results.to_csv(f'{OUTPUT_DIR}/full_results.csv', index=False)

degs_strict = results[(results['padj'] < 0.05) & (results['log2FoldChange'].abs() > 1)]
print(f"\n严格 DEGs: {len(degs_strict)}")

degs_loose = results[(results['padj'] < 0.05) & (results['log2FoldChange'].abs() > 0.58)]
print(f"宽松 DEGs: {len(degs_loose)}")

# 火山图（已修复，保持原样）
print("\n生成火山图...")
results['-log10_padj'] = -np.log10(results['padj'].clip(1e-300))

plt.figure(figsize=(12,9))
sns.scatterplot(data=results, x='log2FoldChange', y='-log10_padj', color='lightgray', alpha=0.35, s=10)

if not degs_strict.empty:
    up_genes = degs_strict[degs_strict['log2FoldChange'] > 1].index
    down_genes = degs_strict[degs_strict['log2FoldChange'] < -1].index
    
    if len(up_genes) > 0:
        sns.scatterplot(data=results.loc[up_genes], x='log2FoldChange', y='-log10_padj', color='red', s=80, label=f'Up ({len(up_genes)})')
    if len(down_genes) > 0:
        sns.scatterplot(data=results.loc[down_genes], x='log2FoldChange', y='-log10_padj', color='blue', s=80, label=f'Down ({len(down_genes)})')

plt.axvline(1, c='gray', ls='--')
plt.axvline(-1, c='gray', ls='--')
plt.axhline(-np.log10(0.05), c='gray', ls='--')
plt.title('Volcano Plot - GSE101521')
plt.xlabel('log2FC')
plt.ylabel('-log10 padj')
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/volcano.png', dpi=400)
plt.close()
print("火山图保存成功")

# ====================== 热图 - 修复版 ======================
# ====================== 热图部分 - 最终适配你的版本 ======================
print("\n生成热图...")

if len(degs_strict) > 0:
    top_genes = degs_strict.sort_values('padj').head(50).index.tolist()
    
    # 你的版本中 normed_counts 已经是 samples x genes (59 x 39376)
    norm_matrix = dds.layers['normed_counts']
    print(f"norm_matrix 形状: {norm_matrix.shape}")  # 确认 (59, 39376)
    
    norm_df = pd.DataFrame(
        norm_matrix,                           # 直接用，不转置！
        index=dds.obs_names,                   # 59 个样本名作为行
        columns=dds.var_names                  # 39376 个基因作为列
    )
    
    plot_df = norm_df[top_genes]               # (59 samples, 50 genes)
    print(f"plot_df 形状: {plot_df.shape}")    # 调试确认
    
    # Z-score 标准化（axis=0，按基因列标准化）
    plot_z = pd.DataFrame(
        zscore(plot_df, axis=0),
        index=plot_df.index,
        columns=plot_df.columns
    )
    
    # 样本颜色条
    group_colors = metadata['condition'].map({'control': '#1f77b4', 'MDD': '#ff7f0e'})
    
    g = sns.clustermap(
        plot_z,
        cmap='RdBu_r',
        center=0,
        row_cluster=True,
        col_cluster=True,
        row_colors=group_colors,
        figsize=(11, 13),
        xticklabels=True,
        yticklabels=False,
        cbar_kws={'label': 'Z-score'},
        dendrogram_ratio=0.08,
        colors_ratio=0.015
    )
    
    g.ax_heatmap.set_title('Heatmap - Top 50 DEGs (by padj)\nMDD vs Control', fontsize=14, pad=20)
    g.ax_heatmap.set_xlabel('Genes', fontsize=12)
    g.ax_heatmap.set_ylabel('Samples', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Control'),
        Patch(facecolor='#ff7f0e', label='MDD')
    ]
    g.ax_heatmap.legend(handles=legend_elements, bbox_to_anchor=(1.35, 1.05), loc='upper right')
    
    g.savefig(f'{OUTPUT_DIR}/heatmap_top50_strict.png', dpi=400, bbox_inches='tight')
    plt.close(g.fig)
    
    print("热图已保存成功！")
else:
    print("无足够 DEGs")
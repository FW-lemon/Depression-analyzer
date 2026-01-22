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

# ====================== 热图部分 - 最终适配你的版本 ======================
# ====================== 最终版 DEG Heatmap（论文终稿级） ======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

print("\n生成 DEG 表达量热图（Z-score，终稿版）...")

if len(degs_strict) > 0:
    # ===== 1. 选 top 50 DEGs（padj 最小优先）=====
    top_df = degs_strict.sort_values('padj').head(50)

    # 上调在上，下调在下（非常关键的“论文感”）
    top_df = pd.concat([
        top_df[top_df['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False),
        top_df[top_df['log2FoldChange'] < 0].sort_values('log2FoldChange')
    ])

    top_genes = top_df.index.tolist()

    # ===== 2. 取标准化表达量 =====
    expr_matrix = dds.layers['normed_counts']
    expr_df = pd.DataFrame(
        expr_matrix,
        index=dds.obs_names,
        columns=dds.var_names
    )

    # genes × samples
    plot_df = expr_df[top_genes].T

    # ===== 3. gene-wise Z-score =====
    plot_z = plot_df.sub(plot_df.mean(axis=1), axis=0) \
                    .div(plot_df.std(axis=1), axis=0)

    # ===== 4. 分组信息 =====
    n_control = 29
    n_mdd = 30
    sample_names = plot_z.columns.tolist()

    # ===== 5. 作图（竖向拉长）=====
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.82])

    hm = sns.heatmap(
        plot_z,
        cmap='RdBu_r',
        center=0,
        vmin=-2,
        vmax=2,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        cbar=False,
        ax=ax
    )

    ax.set_ylabel('Top 50 DEGs', fontsize=13)
    ax.set_xlabel('Samples', fontsize=13)

    # ===== 6. 组间分割线 =====
    ax.axvline(x=n_control, color='black', linewidth=2.8)

    # ===== 7. 顶部分组颜色条 =====
    group_colors = ['#4C72B0'] * n_control + ['#DD8452'] * n_mdd
    color_array = np.array([to_rgba(c) for c in group_colors])[None, :, :]

    ax_bar = fig.add_axes([0.06, 0.91, 0.76, 0.025])
    ax_bar.imshow(color_array, aspect='auto')
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    # 顶部左右组名（非常关键）
    ax_bar.text(
        n_control / 2,
        -0.6,
        'Control',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )
    ax_bar.text(
        n_control + n_mdd / 2,
        -0.6,
        'MDD',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

    # ===== 8. colorbar =====
    cbar_ax = fig.add_axes([0.84, 0.18, 0.025, 0.60])
    cbar = fig.colorbar(hm.collections[0], cax=cbar_ax)
    cbar.set_label('Z-score (row-wise)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # ===== 9. 右上角 legend（靠近但不抢）=====
    legend_elements = [
        Patch(facecolor='#4C72B0', label=f'Control (n={n_control})'),
        Patch(facecolor='#DD8452', label=f'MDD (n={n_mdd})')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.08, 1.00),
        frameon=False,
        fontsize=12
    )

    # ===== 10. 总标题 =====
    fig.suptitle(
        'Heatmap of Differentially Expressed Genes\nZ-score normalized expression',
        fontsize=15,
        fontweight='bold',
        y=0.985
    )

    # ===== 11. 保存 =====
    output_path = f'{OUTPUT_DIR}/heatmap_top50_DEGs_expression_FINAL.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print("✅ 最终版 DEG 热图已保存：", output_path)

else:
    print("没有足够的 DEGs 生成热图")


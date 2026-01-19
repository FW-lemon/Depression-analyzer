import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ================= 配置 =================
INPUT = 'data/GSE101521_clean_59samples.tsv'
OUT = 'deg_GSE101521'
N_CTRL, N_MDD = 29, 30
os.makedirs(OUT, exist_ok=True)

print("🎯 GSE101521 | MDD vs Control")

# ================= 1. 读入 & 去重 =================
counts = pd.read_csv(INPUT, sep='\t', index_col=0).astype(int)
if counts.index.duplicated().any():
    seen = {}
    new_idx = []
    for g in counts.index:
        if g in seen:
            seen[g] += 1
            new_idx.append(f"{g}_{seen[g]}")
        else:
            seen[g] = 0
            new_idx.append(g)
    counts.index = new_idx

print(f"Genes: {len(counts):,}")

# ================= 2. Metadata =================
meta = pd.DataFrame(
    {'condition': ['control'] * N_CTRL + ['MDD'] * N_MDD},
    index=counts.columns
)

# ================= 3. DESeq2 =================
dds = DeseqDataSet(counts=counts.T, metadata=meta, design_factors='condition', refit_cooks=True)
dds.deseq2()

stat = DeseqStats(dds, contrast=['condition', 'MDD', 'control'])
stat.summary()
res = stat.results_df.assign(gene=lambda x: x.index)
res.to_csv(f'{OUT}/full_results.csv', index=False)

# ================= 4. DEG 筛选 =================
degs = res[(res.padj < 0.05) & (res.log2FoldChange.abs() > 1)]
print(f"Strict DEGs: {len(degs)}")

# ================= 5. 火山图 =================
res['-log10_padj'] = -np.log10(res.padj.clip(1e-300))
plt.figure(figsize=(12,9))
sns.scatterplot(res, x='log2FoldChange', y='-log10_padj', color='lightgray', s=10, alpha=.35)

for c, g, lab in [('red', degs.log2FoldChange > 1, 'Up'),
                  ('blue', degs.log2FoldChange < -1, 'Down')]:
    idx = degs[g].index
    if len(idx):
        sns.scatterplot(res.loc[idx], x='log2FoldChange', y='-log10_padj',
                        color=c, s=80, label=f'{lab} ({len(idx)})')

for x in [1, -1]: plt.axvline(x, ls='--', c='gray')
plt.axhline(-np.log10(0.05), ls='--', c='gray')
plt.title('Volcano Plot - GSE101521')
plt.xlabel('log2FC'); plt.ylabel('-log10 padj')
plt.legend()
plt.savefig(f'{OUT}/volcano.png', dpi=400)
plt.close()

# ================= 6. Heatmap =================
if len(degs):
    top = degs.sort_values('padj').head(50)
    top = pd.concat([
        top[top.log2FoldChange > 0].sort_values('log2FoldChange', ascending=False),
        top[top.log2FoldChange < 0].sort_values('log2FoldChange')
    ])

    expr = pd.DataFrame(dds.layers['normed_counts'],
                        index=dds.obs_names,
                        columns=dds.var_names)[top.index].T
    z = (expr - expr.mean(1).values[:,None]) / expr.std(1).values[:,None]

    fig = plt.figure(figsize=(18,16))
    ax = fig.add_axes([0.06,0.08,0.76,0.82])
    hm = sns.heatmap(z, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                     xticklabels=False, yticklabels=True, cbar=False, ax=ax)

    ax.axvline(N_CTRL, lw=2.8, c='black')
    ax.set_ylabel('Top 50 DEGs'); ax.set_xlabel('Samples')

    # 顶部 group bar
    bar = np.array([to_rgba(c) for c in ['#4C72B0']*N_CTRL + ['#DD8452']*N_MDD])[None,:,:]
    axb = fig.add_axes([0.06,0.91,0.76,0.025])
    axb.imshow(bar, aspect='auto'); axb.axis('off')
    axb.text(N_CTRL/2, -0.6, 'Control', ha='center', va='bottom', weight='bold')
    axb.text(N_CTRL+N_MDD/2, -0.6, 'MDD', ha='center', va='bottom', weight='bold')

    # colorbar + legend
    cax = fig.add_axes([0.84,0.18,0.025,0.6])
    fig.colorbar(hm.collections[0], cax=cax).set_label('Z-score (row-wise)')

    ax.legend(
        handles=[Patch(fc='#4C72B0', label=f'Control (n={N_CTRL})'),
                 Patch(fc='#DD8452', label=f'MDD (n={N_MDD})')],
        loc='upper left', bbox_to_anchor=(1.08,1), frameon=False
    )

    fig.suptitle('Heatmap of Differentially Expressed Genes\nZ-score normalized expression',
                 y=0.985, weight='bold')

    plt.savefig(f'{OUT}/heatmap_top50_DEGs_expression_FINAL.png',
                dpi=600, bbox_inches='tight')
    plt.close()

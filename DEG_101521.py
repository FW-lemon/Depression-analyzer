# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 解决服务器 Tcl 崩溃
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from scipy.stats import zscore

# Rpy2 配置 (limma)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# PyDESeq2 配置
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ====================== 1. 配置 ======================
INPUT_FILE = '/home/project/yihao/data/GSE101521.tsv'
BASE_OUT   = 'MDD_DEG_results'
LIMMA_OUT  = os.path.join(BASE_OUT, 'limma_results')
DESEQ2_OUT = os.path.join(BASE_OUT, 'deseq2_results')

N_CONTROL = 29
N_MDD = 30
CTRL_LAB = 'control'
CASE_LAB = 'MDD'

# 阈值
L_LFC, L_PADJ = 0.58, 0.10  # Loose
S_LFC, S_PADJ = 1.0, 0.05   # Strict

for d in [LIMMA_OUT, DESEQ2_OUT]:
    os.makedirs(d, exist_ok=True)

# ====================== 2. 加载 & 原始去重逻辑 ======================
print("🎯 开始处理数据...")
counts_raw = pd.read_csv(INPUT_FILE, sep='\t', index_col=0).astype(int)
print(f"原始基因数: {len(counts_raw):,}")

# --- 移植你的 idxmax 去重逻辑 ---
if counts_raw.index.duplicated().any():
    print("正在使用 row_sums.idxmax() 逻辑去重...")
    row_sums = counts_raw.sum(axis=1)
    selected_indices = []
    for gene, grp in row_sums.groupby(counts_raw.index):
        if len(grp) == 1:
            selected_indices.append(grp.index[0])
        else:
            selected_indices.append(grp.idxmax())
    counts = counts_raw.loc[selected_indices]
else:
    counts = counts_raw

# 强制唯一化 (你的二次备份逻辑)
if counts.index.duplicated().any():
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
counts.index.name = "gene"

# ====================== 3. 绘图函数 (终稿级布局) ======================
def save_deg_lists(df, out_dir):
    df.to_csv(os.path.join(out_dir, "all_genes.csv"), index=False)
    loose = df[(df.padj < L_PADJ) & (df.log2FoldChange.abs() > L_LFC)]
    loose.to_csv(os.path.join(out_dir, "DEGs_loose.csv"), index=False)
    strict = df[(df.padj < S_PADJ) & (df.log2FoldChange.abs() > S_LFC)]
    strict.to_csv(os.path.join(out_dir, "DEGs_strict.csv"), index=False)
    return loose, strict

def plot_volcano(df, out_dir, title):
    # 移植你的 0.35 alpha 和 s=10 火山图风格
    df['-log10_padj'] = -np.log10(df['padj'].clip(1e-300))
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df, x='log2FoldChange', y='-log10_padj', color='lightgray', alpha=0.35, s=10)
    
    up = df[(df.padj < S_PADJ) & (df.log2FoldChange > S_LFC)]
    down = df[(df.padj < S_PADJ) & (df.log2FoldChange < -S_LFC)]
    
    if not up.empty:
        sns.scatterplot(data=up, x='log2FoldChange', y='-log10_padj', color='red', s=80, label=f'Up ({len(up)})')
    if not down.empty:
        sns.scatterplot(data=down, x='log2FoldChange', y='-log10_padj', color='blue', s=80, label=f'Down ({len(down)})')
    
    plt.axvline(S_LFC, c='gray', ls='--')
    plt.axvline(-S_LFC, c='gray', ls='--')
    plt.axhline(-np.log10(S_PADJ), c='gray', ls='--')
    plt.title(title); plt.legend(); plt.savefig(os.path.join(out_dir, "volcano.png"), dpi=400); plt.close()

def plot_heatmap_final(expr_df, degs_df, out_dir, method_name):
    # 移植你的“论文终稿级”热图逻辑
    if len(degs_df) < 5: return
    top_df = degs_df.sort_values('padj').head(50)
    top_df = pd.concat([
        top_df[top_df['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False),
        top_df[top_df['log2FoldChange'] < 0].sort_values('log2FoldChange')
    ])
    
    plot_z = expr_df.loc[top_df['gene']].sub(expr_df.loc[top_df['gene']].mean(axis=1), axis=0) \
                                        .div(expr_df.loc[top_df['gene']].std(axis=1), axis=0)

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.82])
    hm = sns.heatmap(plot_z, cmap='RdBu_r', center=0, vmin=-2, vmax=2, xticklabels=False, yticklabels=True, cbar=False, ax=ax)
    ax.axvline(x=N_CONTROL, color='black', linewidth=2.8)
    
    # 组名标注逻辑
    group_colors = ['#4C72B0'] * N_CONTROL + ['#DD8452'] * N_MDD
    color_array = np.array([to_rgba(c) for c in group_colors])[None, :, :]
    ax_bar = fig.add_axes([0.06, 0.91, 0.76, 0.025])
    ax_bar.imshow(color_array, aspect='auto'); ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.text(N_CONTROL/2, -0.6, 'Control', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax_bar.text(N_CONTROL + N_MDD/2, -0.6, 'MDD', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    cbar_ax = fig.add_axes([0.84, 0.18, 0.025, 0.60])
    fig.colorbar(hm.collections[0], cax=cbar_ax).set_label('Z-score (row-wise)')
    
    legend_elements = [Patch(facecolor='#4C72B0', label=f'Control (n={N_CONTROL})'),
                       Patch(facecolor='#DD8452', label=f'MDD (n={N_MDD})')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.08, 1.00), frameon=False)
    plt.suptitle(f'Heatmap of DEGs | {method_name}', fontsize=15, fontweight='bold', y=0.985)
    plt.savefig(os.path.join(out_dir, "heatmap_FINAL.png"), dpi=600, bbox_inches='tight'); plt.close()

# ====================== 4. 运行流程 ======================
# --- A. limma-voom ---
print("\n🚀 运行 limma-voom...")
with localconverter(ro.default_converter + pandas2ri.converter):
    ro.globalenv["counts"] = ro.conversion.py2rpy(counts)
    ro.globalenv["group"] = ro.conversion.py2rpy(pd.Categorical([CTRL_LAB]*N_CONTROL + [CASE_LAB]*N_MDD))

ro.r("""
library(limma); library(edgeR)
dge <- DGEList(counts=counts); dge <- calcNormFactors(dge)
design <- model.matrix(~ 0 + factor(group, levels=c("control", "MDD")))
colnames(design) <- c("control", "MDD")
v <- voom(dge, design, plot=FALSE); fit <- lmFit(v, design)
cont <- makeContrasts(MDD - control, levels=design); fit <- contrasts.fit(fit, cont); fit <- eBayes(fit)
res_limma <- topTable(fit, number=Inf, sort.by="none"); expr_limma <- v$E
""")

with localconverter(ro.default_converter + pandas2ri.converter):
    res_limma = ro.conversion.rpy2py(ro.r("res_limma")).reset_index().rename(columns={"index":"gene","logFC":"log2FoldChange","P.Value":"pvalue","adj.P.Val":"padj"})
    expr_limma = pd.DataFrame(ro.conversion.rpy2py(ro.r("expr_limma")), index=counts.index, columns=counts.columns)

l_limma, s_limma = save_deg_lists(res_limma, LIMMA_OUT)
plot_volcano(res_limma, LIMMA_OUT, "Volcano Plot | limma-voom")
plot_heatmap_final(expr_limma, s_limma, LIMMA_OUT, "limma-voom")

# --- B. DESeq2 ---
print("\n🚀 运行 PyDESeq2...")
metadata = pd.DataFrame({'condition': [CTRL_LAB]*N_CONTROL + [CASE_LAB]*N_MDD}, index=counts.columns)
dds = DeseqDataSet(counts=counts.T, metadata=metadata, design="~condition", refit_cooks=True)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=['condition', CASE_LAB, CTRL_LAB])
stat_res.summary()
res_deseq = stat_res.results_df.copy(); res_deseq['gene'] = res_deseq.index
expr_deseq = pd.DataFrame(dds.layers['normed_counts'].T, index=counts.index, columns=counts.columns)

l_deseq, s_deseq = save_deg_lists(res_deseq, DESEQ2_OUT)
plot_volcano(res_deseq, DESEQ2_OUT, "Volcano Plot | DESeq2")
plot_heatmap_final(expr_deseq, s_deseq, DESEQ2_OUT, "DESeq2")

print("\n" + "="*40 + "\n✅ 所有分析已圆满完成！\n" + "="*40)
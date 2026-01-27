# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

# Rpy2 配置 (limma)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# PyDESeq2 配置
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ====================== 1. 配置 ======================
INPUT_FILE = '/home/project/yihao/data/GSE136825.tsv'
BASE_OUT   = 'SINUSITIS_DEG_results'
LIMMA_OUT  = os.path.join(BASE_OUT, 'limma_results')
DESEQ2_OUT = os.path.join(BASE_OUT, 'deseq2_results')

N_CASE = 42
N_CONTROL = 33
CASE_LAB = 'Sinusitis'
CTRL_LAB = 'Control'

# 阈值配置
L_LFC, L_PADJ = 0.58, 0.10  # 松 (1.5倍)
S_LFC, S_PADJ = 1.0, 0.05   # 严 (2倍)

for d in [LIMMA_OUT, DESEQ2_OUT]:
    os.makedirs(d, exist_ok=True)

# ====================== 2. 数据加载 & 严谨去重 ======================
print("🎯 开始处理数据...")
counts_raw = pd.read_csv(INPUT_FILE, sep=None, engine='python', index_col=0).fillna(0).astype(int)
counts_raw.index = counts_raw.index.astype(str).str.strip()

# 使用 row_sums.idxmax() 逻辑去重
print("正在去重...")
counts_raw['tmp_sum'] = counts_raw.sum(axis=1)
counts = (counts_raw.sort_values('tmp_sum', ascending=False)
          .groupby(level=0)
          .head(1)
          .drop(columns='tmp_sum'))
counts.index.name = "gene"

groups = [CASE_LAB] * N_CASE + [CTRL_LAB] * N_CONTROL
counts.columns = [f"S{i+1}_{g}" for i, g in enumerate(groups)]
print(f"✅ 处理完成。基因数: {len(counts)}, 样本数: {len(counts.columns)}")

# ====================== 3. 通用功能函数 ======================
def save_deg_lists(df, out_dir):
    """保存全基因、松阈值、严阈值三个文件"""
    df.to_csv(os.path.join(out_dir, "all_genes.csv"), index=False)
    
    # 过滤掉无效值
    clean_df = df.dropna(subset=['padj', 'log2FoldChange'])
    
    loose = clean_df[(clean_df.padj < L_PADJ) & (clean_df.log2FoldChange.abs() > L_LFC)]
    loose.to_csv(os.path.join(out_dir, "DEGs_loose.csv"), index=False)
    
    strict = clean_df[(clean_df.padj < S_PADJ) & (clean_df.log2FoldChange.abs() > S_LFC)]
    strict.to_csv(os.path.join(out_dir, "DEGs_strict.csv"), index=False)
    
    print(f"📂 {out_dir} 已保存: Loose({len(loose)}), Strict({len(strict)})")
    return loose, strict

def plot_volcano(df, out_dir, title):
    plot_df = df.copy().dropna(subset=['padj', 'log2FoldChange'])
    plot_df['-log10_padj'] = -np.log10(plot_df['padj'].clip(1e-300))
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x='log2FoldChange', y='-log10_padj', color='lightgray', alpha=0.35, s=15)
    
    up = plot_df[(plot_df.padj < S_PADJ) & (plot_df.log2FoldChange > S_LFC)]
    down = plot_df[(plot_df.padj < S_PADJ) & (plot_df.log2FoldChange < -S_LFC)]
    
    if not up.empty:
        sns.scatterplot(data=up, x='log2FoldChange', y='-log10_padj', color='red', s=60, label=f'Up ({len(up)})')
    if not down.empty:
        sns.scatterplot(data=down, x='log2FoldChange', y='-log10_padj', color='blue', s=60, label=f'Down ({len(down)})')
    
    plt.axvline(S_LFC, c='gray', ls='--'); plt.axvline(-S_LFC, c='gray', ls='--')
    plt.axhline(-np.log10(S_PADJ), c='gray', ls='--')
    plt.title(title); plt.legend(); plt.savefig(os.path.join(out_dir, "volcano.png"), dpi=300); plt.close()

def plot_heatmap_final(expr_df, degs_df, out_dir, method_name):
    if len(degs_df) < 5: return
    top_df = degs_df.sort_values('padj').head(50)
    top_genes = [g for g in top_df['gene'] if g in expr_df.index]
    plot_z = expr_df.loc[top_genes].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    sns.heatmap(plot_z, cmap='RdBu_r', center=0, vmin=-2, vmax=2, xticklabels=False, ax=ax, cbar_ax=fig.add_axes([0.85, 0.2, 0.02, 0.6]))
    
    ax_bar = fig.add_axes([0.1, 0.91, 0.7, 0.02])
    colors = ['#DD8452'] * N_CASE + ['#4C72B0'] * N_CONTROL
    ax_bar.imshow([ [to_rgba(c) for c in colors] ], aspect='auto')
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.text(N_CASE/2, 0.5, 'Sinusitis', va='center', ha='center', fontweight='bold', color='white')
    ax_bar.text(N_CASE + N_CONTROL/2, 0.5, 'Control', va='center', ha='center', fontweight='bold', color='white')
    
    plt.suptitle(f'Top 50 DEGs Heatmap | {method_name}', y=0.98, fontsize=15)
    plt.savefig(os.path.join(out_dir, "heatmap_FINAL.png"), dpi=300, bbox_inches='tight'); plt.close()

# ====================== 4. 运行流程 ======================

# --- A. limma-voom ---
print("\n🚀 运行 limma-voom...")
try:
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["counts_r"] = ro.conversion.py2rpy(counts)
        ro.globalenv["group_r"] = ro.conversion.py2rpy(pd.Categorical(groups, categories=[CTRL_LAB, CASE_LAB]))

    # 在 R 侧完成所有计算，并显式转换数据类型
    ro.r(f"""
    library(limma); library(edgeR)
    dge <- DGEList(counts=counts_r); dge <- calcNormFactors(dge)
    design <- model.matrix(~ 0 + group_r)
    colnames(design) <- c("{CTRL_LAB}", "{CASE_LAB}")
    v <- voom(dge, design, plot=FALSE); fit <- lmFit(v, design)
    cont <- makeContrasts({CASE_LAB} - {CTRL_LAB}, levels=design)
    fit <- contrasts.fit(fit, cont); fit <- eBayes(fit)
    
    # 提取结果
    res_limma_r <- topTable(fit, number=Inf, sort.by="none")
    # 强制将表达矩阵转为数值向量，避免 FloatMatrix 转换问题
    expr_vec <- as.numeric(v$E)
    expr_rows <- nrow(v$E)
    expr_cols <- ncol(v$E)
    """)

    # 1. 处理差异分析统计表
    res_limma_r = ro.r("res_limma_r")
    # 转换 R 的 DataFrame 为 Pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        res_limma = ro.conversion.rpy2py(res_limma_r)
    
    # 统一列名
    res_limma = pd.DataFrame(res_limma)
    res_limma = res_limma.rename(columns={"logFC":"log2FoldChange","P.Value":"pvalue","adj.P.Val":"padj"})
    res_limma['gene'] = res_limma.index

    # 2. 处理表达矩阵 (通过向量重组，避开 .shape 报错)
    expr_vec = np.array(ro.r("expr_vec"))
    rows = int(ro.r("expr_rows")[0])
    cols = int(ro.r("expr_cols")[0])
    
    # R 是列优先 (Fortran order)
    expr_data = expr_vec.reshape((rows, cols), order='F')
    expr_limma = pd.DataFrame(expr_data, index=counts.index, columns=counts.columns)

    # 3. 保存双阈值结果并绘图
    # 这里会调用你之前的 save_deg_lists 函数，保存 all, loose, strict 三个文件
    l_limma, s_limma = save_deg_lists(res_limma, LIMMA_OUT)
    plot_volcano(res_limma, LIMMA_OUT, "Volcano Plot | limma-voom")
    plot_heatmap_final(expr_limma, s_limma, LIMMA_OUT, "limma-voom")
    
    print("✅ limma-voom 运行成功并保存完成")

except Exception as e:
    print(f"❌ limma-voom 失败: {e}")
    # 如果报错，打印出具体类型方便调试
    if 'res_limma_r' in locals() or 'res_limma_r' in globals():
        print(f"DEBUG: res_limma_r type is {type(ro.r('res_limma_r'))}")

# --- B. DESeq2 ---
print("\n🚀 运行 PyDESeq2...")
try:
    metadata = pd.DataFrame({'condition': groups}, index=counts.columns)
    dds = DeseqDataSet(counts=counts.T, metadata=metadata, design="~condition")
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=['condition', CASE_LAB, CTRL_LAB])
    stat_res.summary()
    
    res_deseq = stat_res.results_df.copy()
    res_deseq['gene'] = res_deseq.index
    expr_deseq = pd.DataFrame(dds.layers['normed_counts'].T, index=counts.index, columns=counts.columns)

    l_deseq, s_degs = save_deg_lists(res_deseq, DESEQ2_OUT)
    plot_volcano(res_deseq, DESEQ2_OUT, "Volcano Plot | DESeq2")
    plot_heatmap_final(expr_deseq, s_degs, DESEQ2_OUT, "DESeq2")
    print("✅ PyDESeq2 完成")
except Exception as e:
    print(f"❌ PyDESeq2 失败: {e}")

print("\n" + "="*40 + "\n✅ 所有分析已圆满完成！\n" + "="*40)


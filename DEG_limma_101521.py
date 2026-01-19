# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# ==============================
# 参数（与 DESeq2 流程保持一致）
# ==============================
INPUT = "data/GSE101521_clean_59samples.tsv"
OUT   = "limma_GSE101521"

N_CTRL = 29
N_MDD  = 30

CTRL = "control"
CASE = "MDD"

os.makedirs(OUT, exist_ok=True)

print("🎯 limma-voom | MDD vs Control")

# ==============================
# 1. 读 counts
# ==============================
counts = pd.read_csv(INPUT, sep="\t", index_col=0).astype(int)
counts.columns = counts.columns.astype(str).str.strip()

print(f"Genes (raw): {counts.shape[0]:,}")
print(f"Samples: {counts.shape[1]}")

assert counts.shape[1] == N_CTRL + N_MDD, "❌ 样本数不匹配"

# ==============================
# 2. gene 去重
# ==============================
if counts.index.duplicated().any():
    seen, new_idx = {}, []
    for g in counts.index:
        if g in seen:
            seen[g] += 1
            new_idx.append(f"{g}_{seen[g]}")
        else:
            seen[g] = 0
            new_idx.append(g)
    counts.index = new_idx

counts.index.name = "gene"

# ==============================
# 3. 构造分组
# ==============================
group = pd.Categorical(
    [CTRL] * N_CTRL + [CASE] * N_MDD,
    categories=[CTRL, CASE]
)

# ==============================
# 4. 送入 R
# ==============================
with localconverter(ro.default_converter + pandas2ri.converter):
    ro.globalenv["counts"] = ro.conversion.py2rpy(counts)
    ro.globalenv["group"]  = ro.conversion.py2rpy(group)

# ==============================
# 5. limma-voom 计算
# ==============================
ro.r("""
library(limma)
library(edgeR)

group <- factor(group, levels=c("control", "MDD"))

dge <- DGEList(counts=counts)
dge <- calcNormFactors(dge)

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

v <- voom(dge, design, plot=FALSE)

fit <- lmFit(v, design)
cont <- makeContrasts(MDD - control, levels=design)
fit <- contrasts.fit(fit, cont)
fit <- eBayes(fit)

res  <- topTable(fit, number=Inf, sort.by="none")
expr <- v$E  # 提取 log2-CPM 表达矩阵
""")

# ==============================
# 6. 回到 Python (修复关键点)
# ==============================
with localconverter(ro.default_converter + pandas2ri.converter):
    res = ro.conversion.rpy2py(ro.r("res"))
    r_expr = ro.conversion.rpy2py(ro.r("expr"))

# 核心修复：手动将 NumPy 数组转回带标签的 DataFrame
expr = pd.DataFrame(r_expr, index=counts.index, columns=counts.columns)

# ==============================
# 7. 结果结构对齐
# ==============================
res = res.reset_index().rename(columns={
    "logFC": "log2FoldChange",
    "P.Value": "pvalue",
    "adj.P.Val": "padj",
    "t": "stat",
    "AveExpr": "baseMean",
    "index": "gene"
})

res = res[["gene", "log2FoldChange", "pvalue", "padj", "baseMean", "stat", "B"]]

# ==============================
# 8. 保存结果
# ==============================
res.to_csv(f"{OUT}/all_genes.csv", index=False)

degs_loose  = res[(res.padj < 0.10) & (res.log2FoldChange.abs() > 0.58)]
degs_strict = res[(res.padj < 0.05) & (res.log2FoldChange.abs() > 1)]

degs_loose.to_csv(f"{OUT}/DEGs_loose.csv", index=False)
degs_strict.to_csv(f"{OUT}/DEGs_strict.csv", index=False)

print(f"DEGs loose : {len(degs_loose)}")
print(f"DEGs strict: {len(degs_strict)}")

# ==============================
# 9. 火山图
# ==============================
res["-log10_padj"] = -np.log10(res.padj.clip(1e-300))

plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=res,
    x="log2FoldChange",
    y="-log10_padj",
    color="lightgray",
    s=10,
    alpha=0.4
)

for c, cond, lab in [
    ("red",  degs_strict.log2FoldChange > 1, "Up"),
    ("blue", degs_strict.log2FoldChange < -1, "Down")
]:
    idx = degs_strict[cond].index
    if len(idx):
        sns.scatterplot(
            data=res.loc[idx],
            x="log2FoldChange",
            y="-log10_padj",
            color=c,
            s=80,
            label=f"{lab} ({len(idx)})"
        )

plt.axvline(1,  ls="--", c="gray")
plt.axvline(-1, ls="--", c="gray")
plt.axhline(-np.log10(0.05), ls="--", c="gray")

plt.xlabel("log2FC")
plt.ylabel("-log10 padj")
plt.title("Volcano Plot | limma-voom")
plt.legend()
plt.savefig(f"{OUT}/volcano.png", dpi=400)
plt.close()

# ==============================
# 10. 热图 (修复 .loc 问题)
# ==============================
if len(degs_strict) > 0:
    # 选出 Top 50 差异最显著的基因
    top_genes_df = degs_strict.sort_values("padj").head(50)
    
    # 按照 Up/Down 排序让热图更好看
    top_genes_df = pd.concat([
        top_genes_df[top_genes_df.log2FoldChange > 0].sort_values("log2FoldChange", ascending=False),
        top_genes_df[top_genes_df.log2FoldChange < 0].sort_values("log2FoldChange")
    ])

    # 提取表达数据 (此时 expr 已经是 DataFrame)
    expr_plot = expr.loc[top_genes_df.gene]

    # 标准化 (Z-score)
    z = (expr_plot - expr_plot.mean(axis=1).values[:, None]) / expr_plot.std(axis=1).values[:, None]

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_axes([0.1, 0.08, 0.75, 0.82])

    hm = sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        xticklabels=False,
        yticklabels=True,
        cbar=False,
        ax=ax
    )

    # 在 Control 和 MDD 之间画一条分界线
    ax.axvline(N_CTRL, lw=3, c="black")
    ax.set_ylabel("Top 50 DEGs")
    ax.set_xlabel("Samples")

    # 顶部颜色条 (分组标注)
    bar = np.array(
        [to_rgba("#4C72B0")] * N_CTRL +
        [to_rgba("#DD8452")] * N_MDD
    )[None, :, :]

    axb = fig.add_axes([0.1, 0.91, 0.75, 0.025])
    axb.imshow(bar, aspect="auto")
    axb.axis("off")

    # 侧边 Colorbar
    cax = fig.add_axes([0.88, 0.18, 0.02, 0.6])
    fig.colorbar(hm.collections[0], cax=cax).set_label("Z-score")

    # 图例
    ax.legend(
        handles=[
            Patch(fc="#4C72B0", label=f"Control (n={N_CTRL})"),
            Patch(fc="#DD8452", label=f"MDD (n={N_MDD})")
        ],
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        frameon=False
    )

    fig.suptitle(
        "Heatmap of Differentially Expressed Genes\nlimma-voom (Z-score)",
        y=0.98,
        weight="bold",
        fontsize=16
    )

    plt.savefig(f"{OUT}/heatmap_top50_DEGs.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("📈 Heatmap saved successfully.")
else:
    print("⚠️ No strict DEGs found, skipping heatmap.")

print("✅ limma-voom pipeline finished.")
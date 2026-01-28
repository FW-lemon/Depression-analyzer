import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
import sys

# ====================== 参数 ======================
if len(sys.argv) < 2:
    print("用法: python overlap.py MDD 或 python overlap.py SINUSITIS")
    sys.exit(1)

PROJECT = sys.argv[1].upper()

# 每个项目对应关键模块颜色
KEY_MODULE = {
    'MDD': 'green',
    'SINUSITIS': 'turquoise'
}

if PROJECT not in KEY_MODULE:
    print(f"未定义项目: {PROJECT}")
    sys.exit(1)

MODULE = KEY_MODULE[PROJECT]

print(f"\n当前项目: {PROJECT}")
print(f"关键模块: {MODULE}")

# ====================== 文件路径 ======================
wgcna_file = f'{PROJECT}_WGCNA_results/Key_Module_{MODULE}_Genes.csv'

deg_files = {
    f'deseq2_{PROJECT}_loose': f'{PROJECT}_DEG_results/deseq2_results/DEGs_loose.csv',
    f'deseq2_{PROJECT}_strict': f'{PROJECT}_DEG_results/deseq2_results/DEGs_strict.csv',
    f'limma_{PROJECT}_loose': f'{PROJECT}_DEG_results/limma_results/DEGs_loose.csv',
    f'limma_{PROJECT}_strict': f'{PROJECT}_DEG_results/limma_results/DEGs_strict.csv'
}

output_dir = f'{PROJECT}_WGCNA_DEG_overlap_results'
os.makedirs(output_dir, exist_ok=True)

# ====================== 读取 WGCNA ======================
if not os.path.exists(wgcna_file):
    raise FileNotFoundError(f"WGCNA 文件不存在: {wgcna_file}")

wgcna_df = pd.read_csv(wgcna_file)

# ====================== 清洗 WGCNA 基因名 ======================
wgcna_genes = set(
    wgcna_df.iloc[:, 0]      # 第一列一般是基因名
    .astype(str)             # 转成字符串
    .str.strip()             # 去掉首尾空格
    .replace('', pd.NA)      # 空字符串转成 NA
    .dropna()                # 去掉 NA
)

print(f"WGCNA 模块基因数: {len(wgcna_genes)}")
print("示例基因:", list(wgcna_genes)[:5])

# ====================== 2x2 Venn ======================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, path) in enumerate(deg_files.items()):
    if not os.path.exists(path):
        print(f"跳过缺失文件: {path}")
        continue

    deg_df = pd.read_csv(path)

    # ====================== 清洗 DEG 基因名 ======================
    if 'gene' in deg_df.columns:
        deg_genes = deg_df['gene']
    elif 'Gene' in deg_df.columns:
        deg_genes = deg_df['Gene']
    elif 'GeneSymbol' in deg_df.columns:
        deg_genes = deg_df['GeneSymbol']
    else:
        deg_genes = deg_df.iloc[:, -1]  # 最后一列一般是 gene

    deg_genes = set(
        deg_genes
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
    )

    print(f"{name} DEG 基因数: {len(deg_genes)}")
    print(f"{name} 示例基因:", list(deg_genes)[:5])

    # ====================== 交集 ======================
    overlap = wgcna_genes & deg_genes
    print(f"{name} 重叠基因数: {len(overlap)}")

    # 保存重叠基因
    overlap_df = pd.DataFrame({'gene': sorted(list(overlap))})
    overlap_csv_path = os.path.join(output_dir, f'overlap_WGCNA_{name}.csv')
    overlap_df.to_csv(overlap_csv_path, index=False)
    print(f"已保存: {overlap_csv_path}")

    # 画 Venn 图
    venn2([wgcna_genes, deg_genes],
          set_labels=(f'WGCNA {MODULE}', name),
          ax=axes[i])
    axes[i].set_title(f'{name}\nOverlap: {len(overlap)}')

# ====================== 保存总图 ======================
fig_path = os.path.join(output_dir, 'WGCNA_vs_DEGs_4plots.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"\n结果已输出到: {output_dir}")

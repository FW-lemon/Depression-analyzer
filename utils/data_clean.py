import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

FILES = [
    "GSE101521_TPM.tsv",
    "GSE101521.tsv",
    "GSE136825_TPM.tsv",
    "GSE136825.tsv",
]

# 不同数据集保留的样本数
KEEP_N = {
    "GSE101521": 59,
    "GSE136825": 75,
}

# 1. 读取 gene_table
print("Loading gene_table...")
gene_table = pd.read_csv(DATA_DIR / "gene_table.tsv", sep="\t")
gene_map = dict(zip(gene_table["GeneID"].astype(str),
                    gene_table["Symbol"].astype(str)))

print(f"Gene mapping loaded: {len(gene_map)} genes")

# 2. 逐个处理文件
for fname in FILES:
    path = DATA_DIR / fname
    print(f"\nProcessing {fname} ...")

    df = pd.read_csv(path, sep="\t")
    print("Original shape:", df.shape)

    # -------- GeneID -> Symbol --------
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).map(
        lambda x: gene_map.get(x, x)
    )

    # -------- 截断样本列 --------
    dataset = "GSE101521" if "101521" in fname else "GSE136825"
    n_keep = KEEP_N[dataset]

    # 第一列是基因名，所以是 1 + n_keep
    df = df.iloc[:, : (1 + n_keep)]

    print("After processing:", df.shape)

    # -------- 覆盖保存 --------
    df.to_csv(path, sep="\t", index=False)
    print("Saved:", path)

print("\nAll files done.")

import pandas as pd

# 1. 读取 TPM 数据
tpm_file = "data/GSE101521_TPM_symbol.tsv"
df = pd.read_csv(tpm_file, sep='\t', index_col=0)

# 2. 指定你需要的 39 个基因
genes_of_interest = [
    "HES6","MIR12114","APC2","SCRT1","TNK2","CASKIN1","SNORD115-33","MIR665",
    "MIR3183","MIR6511B1","FASN","HCN2","PKD1","MIR4784","SNORD37","RNF208",
    "AATK","HAPLN4","PANX2","CIC","TNFRSF25","LOC107987266","AJM1","NAT8L",
    "TNRC18","SCAF1","MIR4505","ULK1","MIR1250","SNORD115-31","SNORD167",
    "PKD1P4-NPIPA8","MIR937","SNORD115-19","CAVIN2","SCRIB","SHANK3","PLEC","AGRN"
]

# 3. 筛选行
df_subset = df.loc[df.index.intersection(genes_of_interest)]

# 4. 保存到新文件
df_subset.to_csv("data/GSE101521_TPM_39genes.tsv", sep='\t')

print(f"✅ 已筛选 {df_subset.shape[0]} 个基因，保存到 data/GSE101521_TPM_39genes.tsv")

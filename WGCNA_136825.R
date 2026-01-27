# ====================== 1. 环境准备 ======================
library(WGCNA)
library(dplyr)
options(stringsAsFactors = FALSE)
enableWGCNAThreads(nThreads = 27) 

# ====================== 2. 配置 (针对 GSE136825) ======================
INPUT_FILE <- "/home/project/yihao/data/GSE136825.tsv"  # 请确认路径
OUT_DIR <- "BDY_WGCNA_results"
N_CASE <- 42    # Sinusitis
N_CTRL <- 33    # Control
dir.create(OUT_DIR, showWarnings = FALSE)

# ====================== 3. 数据读取与预处理 ======================
print("1. 读取鼻窦炎数据集...")
raw_df <- read.table(INPUT_FILE, header = TRUE, sep = "\t", check.names = FALSE)

# 基因名清洗并去重
clean_names <- function(x) {
  x <- toupper(as.character(x))
  x <- trimws(x)
  x <- gsub("\\..*$", "", x)
  return(x)
}
raw_df[,1] <- clean_names(raw_df[,1])

print("正在处理重复基因名...")
data_merged <- aggregate(. ~ raw_df[,1], data = raw_df[,-1], FUN = mean)
rownames(data_merged) <- data_merged[,1]
data_merged <- data_merged[, -1]

# 转置为 WGCNA 格式 (样本 x 基因)
datExpr0 <- as.data.frame(t(data_merged))

# 过滤低质量基因
gsg <- goodSamplesGenes(datExpr0, verbose = 3)
if (!gsg$allOK) {
    datExpr0 <- datExpr0[gsg$goodSamples, gsg$goodGenes]
}

# ====================== 4. 高方差基因筛选 (Top 8000) ======================
vars <- apply(datExpr0, 2, var)
datExpr <- datExpr0[, names(sort(vars, decreasing = TRUE))[1:min(8000, ncol(datExpr0))]]
print(paste("最终分析样本数:", nrow(datExpr), "基因数:", ncol(datExpr)))

# ====================== 5. 动态构建表型数据 (核心修复) ======================
# 按照你 Python 代码的逻辑：前 42 个是 Case，后 33 个是 Control
# 这里通过匹配过滤后的样本名来确保对应关系
all_sample_names <- rownames(datExpr0)
trait_map <- data.frame(
    SampleID = all_sample_names,
    # 根据 42/33 比例生成标签
    Sinusitis = c(rep(1, N_CASE), rep(0, N_CTRL)) 
)
rownames(trait_map) <- trait_map$SampleID

# 关键：根据 datExpr 现有的样本进行重排和过滤
datTraits <- trait_map[rownames(datExpr), "Sinusitis", drop=FALSE]

# ====================== 6. 软阈值选择 ======================
print("2. 筛选软阈值...")
powers <- c(1:10, seq(12, 20, by=2))
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

pdf(file.path(OUT_DIR, "SoftThreshold_GSE136825.pdf"), width = 9, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], type="n", main="Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], labels=powers, col="red")
abline(h=0.8, col="red")
plot(sft$fitIndices[,1], sft$fitIndices[,5], type="n", main="Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, col="red")
dev.off()

# 自动选点或手动设定
softPower <- sft$powerEstimate
if(is.na(softPower)) softPower <- 6 # 如果没跑出来，默认给个常用值
print(paste("推荐软阈值:", softPower))

# ====================== 7. 网络构建 ======================
print("3. 构建网络...")
net <- blockwiseModules(
  datExpr,
  power = softPower,
  TOMType = "unsigned",  # 样本数较多时 unsigned 比较稳健
  minModuleSize = 30,
  mergeCutHeight = 0.25,
  numericLabels = FALSE,
  maxBlockSize = 8000,
  verbose = 3
)

# ====================== 8. 相关性分析 ======================
MEs <- net$MEs
moduleTraitCor <- cor(MEs, datTraits, use = "p")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

# 热图保存
pdf(file.path(OUT_DIR, "Module_Trait_Heatmap.pdf"), width = 6, height = 10)
labeledHeatmap(
  Matrix = moduleTraitCor,
  xLabels = colnames(datTraits),
  yLabels = colnames(MEs),
  colorLabels = FALSE,
  colors = blueWhiteRed(50),
  textMatrix = paste(round(moduleTraitCor, 2), "\n(", signif(moduleTraitPvalue, 1), ")", sep = ""),
  main = "Module-trait relationships (GSE136825)"
)
dev.off()

# ====================== 9. 导出结果 ======================

print("正在导出完整 CSV 文件...")

# 1. 导出模块与表型的相关性系数 (Correlation)
write.csv(moduleTraitCor, 
          file.path(OUT_DIR, "Module_Trait_Correlation.csv"))

# 2. 导出模块与表型的显著性 P 值 (P-value)
write.csv(moduleTraitPvalue, 
          file.path(OUT_DIR, "Module_Trait_Pvalue.csv"))

# 3. 导出所有基因所属的模块颜色
all_modules <- data.frame(
    GeneSymbol = colnames(datExpr),
    Module = net$colors
)
write.csv(all_modules, 
          file.path(OUT_DIR, "All_Genes_With_Modules.csv"), 
          row.names = FALSE)

# 4. 导出最显著模块 (turquoise) 的基因列表
# 这里使用了你运行出来的 MEturquoise
bestME <- colnames(MEs)[which.min(moduleTraitPvalue[,1])]
bestColor <- substring(bestME, 3) # 去掉 "ME" 得到 "turquoise"

moduleGenes <- colnames(datExpr)[net$colors == bestColor]
write.csv(data.frame(GeneSymbol = moduleGenes), 
          file.path(OUT_DIR, paste0("Key_Module_", bestColor, "_Genes.csv")), 
          row.names = FALSE)

# 5. 额外导出：每个模块的基因数量统计 (方便查看模块大小)
gene_counts <- as.data.frame(table(net$colors))
colnames(gene_counts) <- c("ModuleColor", "GeneCount")
write.csv(gene_counts, file.path(OUT_DIR, "Module_Gene_Counts.csv"), row.names = FALSE)

print(paste("最显著关联模块:", bestME, "颜色:", bestColor))
print(paste("该模块包含基因数:", length(moduleGenes)))
print("🎉 所有 CSV 文件已补全，保存至：")
print(normalizePath(OUT_DIR))
# ====================== 1. 环境准备 ======================
library(WGCNA)
library(dplyr)
options(stringsAsFactors = FALSE)
enableWGCNAThreads(nThreads = 27)  # 并行计算

# ====================== 2. 输出目录 ======================
OUT_DIR <- "WGCNA_results_top8000"
dir.create(OUT_DIR, showWarnings = FALSE)

# ====================== 3. 数据读取与预处理 ======================
print("1. 读取表达矩阵...")
raw_df <- read.table(
  "/home/project/yihao/data/clean_59samples_symbol_unique.tsv",
  header = TRUE, sep = "\t", check.names = FALSE
)

# 基因名清洗函数
clean_names <- function(x) {
  x <- toupper(as.character(x))
  x <- trimws(x)
  x <- gsub("\\..*$", "", x)
  return(x)
}

raw_df[,1] <- clean_names(raw_df[,1])

# 重复基因取均值
data_merged <- aggregate(. ~ raw_df[,1], data = raw_df[,-1], FUN = mean)
colnames(data_merged)[1] <- "GeneID"
rownames(data_merged) <- data_merged$GeneID
data_merged <- data_merged[, -1]

# 转为 WGCNA 格式：样本 x 基因
datExpr0 <- as.data.frame(t(data_merged))

# 去除坏基因或坏样本
gsg <- goodSamplesGenes(datExpr0, verbose = 3)
datExpr0 <- datExpr0[, gsg$goodGenes]

# ====================== 4. 高方差基因筛选 ======================
print("2. 高方差基因筛选...")
vars <- apply(datExpr0, 2, var)
topN <- 8000
topGenes <- names(sort(vars, decreasing = TRUE))[1:topN]
datExpr <- datExpr0[, topGenes]

print(paste("最终进入 WGCNA 的基因数:", ncol(datExpr)))

write.csv(
  data.frame(GeneSymbol = topGenes),
  file.path(OUT_DIR, "Input_Top8000_Genes.csv"),
  row.names = FALSE
)

# ====================== 5. 表型数据 ======================
datTraits <- data.frame(
  MDD = c(rep(0, 29), rep(1, 30))
)
rownames(datTraits) <- rownames(datExpr)

# ====================== 6. 软阈值选择 ======================
print("3. 选择软阈值...")
powers <- 1:10
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

pdf(file.path(OUT_DIR, "SoftThreshold.pdf"), width = 9, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)", ylab="Scale Free Topology Model Fit, signed R^2",
     type="n", main="Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers, col="red")
abline(h=0.8, col="red")

plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)", ylab="Mean Connectivity",
     type="n", main="Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, col="red")
dev.off()

softPower <- 8   # 当前数据的合理值

# ====================== 7. 构建网络 ======================
print("4. 构建共表达网络...")
net <- blockwiseModules(
  datExpr,
  power = softPower,
  TOMType = "signed",
  minModuleSize = 30,
  reassignThreshold = 0,
  mergeCutHeight = 0.25,
  numericLabels = FALSE,  # 保留颜色名
  pamRespectsDendro = FALSE,
  saveTOMs = FALSE,
  maxBlockSize = 20000,
  verbose = 3
)

moduleColors <- net$colors   # 颜色名直接使用
MEs <- net$MEs

# ====================== 8. 模块-性状相关 ======================
print("5. 模块-表型相关分析...")
moduleTraitCor <- cor(MEs, datTraits, use = "p")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

write.csv(moduleTraitCor,
          file.path(OUT_DIR, "Module_Trait_Correlation.csv"))
write.csv(moduleTraitPvalue,
          file.path(OUT_DIR, "Module_Trait_Pvalue.csv"))

# 热图
pdf(file.path(OUT_DIR, "Module_Trait_Heatmap.pdf"), width = 6, height = 8)
labeledHeatmap(
  Matrix = moduleTraitCor,
  xLabels = "MDD",
  yLabels = colnames(MEs),
  colorLabels = FALSE,
  colors = blueWhiteRed(50),
  textMatrix = paste(
    round(moduleTraitCor, 2),
    "\n(",
    signif(moduleTraitPvalue, 1),
    ")",
    sep = ""
  ),
  main = "Module-trait relationships"
)
dev.off()

# ====================== 9. 提取最相关模块基因 ======================
bestME <- colnames(MEs)[which.max(abs(moduleTraitCor[, "MDD"]))]
bestColor <- substring(bestME, 3)  # 去掉 "ME" 前缀得到颜色名

print(paste("最相关模块:", bestME, "颜色:", bestColor))

moduleGenes <- colnames(datExpr)[moduleColors == bestColor]

write.csv(
  data.frame(GeneSymbol = moduleGenes),
  file.path(OUT_DIR, paste0("Key_Module_", bestColor, "_Genes.csv")),
  row.names = FALSE
)

# ====================== 10. 保存所有模块基因 ======================
print("6. 导出所有模块基因...")
all_modules <- data.frame(
  GeneSymbol = colnames(datExpr),
  Module = moduleColors
)

write.csv(
  all_modules,
  file.path(OUT_DIR, "All_Genes_With_Modules.csv"),
  row.names = FALSE
)

print("🎉 WGCNA 全流程完成，所有结果已保存至：")
print(normalizePath(OUT_DIR))

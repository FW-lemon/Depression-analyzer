🧠 Depression-analyzer: MDD & Sinusitis Transcriptomic Analysis

这是一个基于转录组数据的生物信息学分析 pipeline。该项目整合了 WGCNA (加权基因共表达网络分析)、DESeq2 差异表达分析 以及 多模型机器学习分类，旨在挖掘抑郁症 (MDD) 等疾病的关键生物标志物。

🚀 核心流程
该项目目前支持针对 MDD (GSE101521) 和 SINUSITIS (GSE136825) 的独立分析：

差异分析 (DEGs): 使用 DESeq2 识别病例组与对照组之间的差异表达基因。

网络分析 (WGCNA): 构建基因共表达网络，识别与临床表型显著相关的模块。

交集筛选: 提取 DEGs 与 WGCNA 核心模块的重叠基因。

机器学习分类:

Random Forest (特征重要性评估)

SVM (线性分类性能优化)

MLP Neural Network (非线性关系捕获)

LightGBM (梯度提升决策树高效预测)

📁 目录结构
Plaintext
.
├── ML.py                    # 核心机器学习脚本 (支持多模型对比)
├── ML-result/               # 训练结果 (ACC 曲线、特征重要性图、对比图)
├── data/                    # TPM 表达矩阵与原始数据
├── MDD_WGCNA_DEG_.../       # MDD 任务的 WGCNA 与 DEG 交集结果
├── SINUSITIS_WGCNA_.../     # 鼻窦炎任务的交集结果
├── environment.yml          # Conda 环境配置文件
└── requirements.txt         # Pip 依赖清单
🛠️ 环境配置
你可以通过以下命令快速复现该项目的运行环境：

Bash
# 克隆仓库
git clone https://github.com/FW-lemon/Depression-analyzer.git
cd Depression-analyzer

# 使用 Conda 创建环境
conda env create -f environment.yml
conda activate yihao
📊 结果展示
1. 模型性能对比
在 ML-result 目录下，你可以查看不同模型在交叉验证中的表现对比：

Model_Comparison_Accuracy.png: 展示 RF, SVM, NN, LightGBM 的 Accuracy 分布。

LearningCurve_[Model].png: 展示模型随样本量增加的收敛情况。

2. 特征基因排名
通过随机森林得出的特征重要性 (Feature Importance)，识别对疾病分类贡献最大的前 20 个基因：

Top_Feature_Importance.png

📝 如何运行
你可以通过修改 ML.py 中的 CONFIG 部分来切换分析任务：

Python
# 修改任务名即可切换
TASK_NAME = "MDD" 
# 或者
TASK_NAME = "SINUSITIS"
运行分析：

Bash
python ML.py
🤝 贡献与交流
如果你对分析流程有任何改进建议，欢迎提交 Pull Request 或开 Issue 讨论。
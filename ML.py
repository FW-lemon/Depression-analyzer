import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from lightgbm import LGBMClassifier

# ======================= CONFIG ==========================
# 你可以根据需要切换这两段配置

# --- 任务 2: SINUSITIS (如需运行请取消此段注释并注释上方) ---
TASK_NAME = "SINUSITIS"
GENE_LIST_FILE = "/home/project/yihao/SINUSITIS_WGCNA_DEG_overlap_results/overlap_WGCNA_deseq2_SINUSITIS_loose.csv"
TPM_FILE = "/home/project/yihao/data/GSE136825_TPM.tsv"
NUM_POSITIVE = 42
NUM_NEGATIVE = 33
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
OUT_ROOT = "ML-result"

"""
# --- 任务 1: MDD ---
TASK_NAME = "MDD"
GENE_LIST_FILE = "/home/project/yihao/MDD_WGCNA_DEG_overlap_results/overlap_WGCNA_deseq2_MDD_loose.csv"
TPM_FILE = "/home/project/yihao/data/GSE101521_TPM.tsv"
NUM_POSITIVE = 29   
NUM_NEGATIVE = 30   
LABEL_POSITIVE = 1   
LABEL_NEGATIVE = 0   
OUT_ROOT = "ML-result"

# --- 任务 2: SINUSITIS (如需运行请取消此段注释并注释上方) ---
TASK_NAME = "SINUSITIS"
GENE_LIST_FILE = "/home/project/yihao/SINUSITIS_WGCNA_DEG_overlap_results/overlap_WGCNA_deseq2_SINUSITIS_loose.csv"
TPM_FILE = "/home/project/yihao/data/GSE136825_TPM.tsv"
NUM_POSITIVE = 42
NUM_NEGATIVE = 33
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
OUT_ROOT = "ML-result"
"""
# =========================================================

def plot_learning_curve(estimator, X, y, out_path, title="Learning Curve"):
    """绘制学习曲线: 观察模型随训练样本增加时的 Accuracy 变化图"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color="#E84118", label="Training Accuracy")
    plt.plot(train_sizes, test_mean, 'o-', color="#44BD32", label="CV Accuracy")
    plt.title(f"{title}\n({TASK_NAME})", fontsize=14)
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    # 1. 创建输出目录
    out_dir = os.path.join(OUT_ROOT, TASK_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 2. 读取数据
    gene_df = pd.read_csv(GENE_LIST_FILE)
    gene_list = gene_df.iloc[:, 0].dropna().astype(str).unique().tolist()
    
    tpm_df = pd.read_csv(TPM_FILE, sep="\t", index_col=0)
    genes_used = sorted(set(gene_list) & set(tpm_df.index))
    
    if not genes_used:
        raise RuntimeError("❌ No gene matched between gene list and TPM")

    # 3. 构建 X (DataFrame格式以消除LGBM警告) 和 y
    X_raw = tpm_df.loc[genes_used].T
    y = np.array([LABEL_POSITIVE] * NUM_POSITIVE + [LABEL_NEGATIVE] * NUM_NEGATIVE)

    # 4. 标准化 (保持列名以解决警告)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=genes_used)

    # 5. 定义多模型
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, importance_type='split')
    }

    # 6. 训练与评估
    cv_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"[{TASK_NAME}] Starting Multi-Model Analysis...")

    for name, model in models.items():
        # 计算 5-Fold 交叉验证 Accuracy
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_results.append(pd.DataFrame({'Model': [name]*len(scores), 'Accuracy': scores}))
        print(f"-> {name}: Mean Acc = {scores.mean():.4f}")
        
        # 绘制并保存该模型的训练学习曲线
        plot_learning_curve(model, X_scaled, y, 
                            os.path.join(out_dir, f"LearningCurve_{name}.png"), 
                            f"Model Training Trend: {name}")

    # 7. 模型性能对比图 (修复 Seaborn palette 警告)
    res_df = pd.concat(cv_results)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.boxplot(x='Model', y='Accuracy', data=res_df, hue='Model', palette="Set3", legend=False)
    sns.stripplot(x='Model', y='Accuracy', data=res_df, color="black", alpha=0.5)
    plt.title(f"Model Accuracy Comparison ({TASK_NAME})", fontsize=15)
    plt.savefig(os.path.join(out_dir, "Model_Comparison_Accuracy.png"), dpi=300)
    plt.close()

    # 8. 特征重要性分析 (以 Random Forest 为例)
    rf = models["RandomForest"]
    rf.fit(X_scaled, y)
    rf_importance = pd.DataFrame({
        "Gene": genes_used,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Gene", data=rf_importance, hue='Gene', palette="viridis", legend=False)
    plt.title(f"Top 20 Important Genes (Random Forest - {TASK_NAME})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Top_Feature_Importance.png"), dpi=300)
    plt.close()

    print(f"✅ Analysis finished! Results are saved in: {out_dir}")

if __name__ == "__main__":
    main()
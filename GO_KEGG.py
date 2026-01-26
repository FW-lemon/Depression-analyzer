import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# ================= 1. 配置区域 =================
INPUT_FILE = "/home/project/yihao/WGCNA_DEG_overlap_results/overlap_WGCNA_deg_GSE101521_loose.csv"
OUTPUT_DIR = "/home/project/yihao/GO_KEGG"
COMBINED_FIGURE_NAME = "Final_Combined_Enrichment_Plot_v2.png"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import matplotlib
matplotlib.use('Agg')

# ================= 2. 数据处理函数 =================
def run_enrichment_and_save(genes, gene_set_name, gene_sets_list):
    print(f"🚀 正在分析并保存中间结果: {gene_set_name}")
    try:
        enr = gp.enrichr(gene_list=genes,
                         gene_sets=gene_sets_list,
                         organism='Human',
                         cutoff=1.0,
                         no_plot=True)
        res = enr.results
        if res.empty: 
            return pd.DataFrame()
        
        # --- 保存中间结果 ---
        csv_path = os.path.join(OUTPUT_DIR, f"Full_Results_{gene_set_name}.csv")
        res.to_csv(csv_path, index=False)
        print(f"💾 {gene_set_name} 完整数据已保存至: {csv_path}")
        
        # 筛选 P < 0.05 的前 15 条用于绘图
        plot_df = res[res['P-value'] < 0.05].copy()
        if plot_df.empty: 
            return pd.DataFrame()
        
        plot_df = plot_df.sort_values('P-value').head(15)
        plot_df['logP'] = -np.log10(plot_df['P-value'])
        plot_df['Gene_Count'] = plot_df['Overlap'].str.split('/').str[0].astype(int)
        plot_df['Term'] = plot_df['Term'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
        return plot_df
    except Exception as e:
        print(f"❌ {gene_set_name} 报错: {e}")
        return pd.DataFrame()

# ================= 3. 绘图函数 =================
def draw_dotplot_subplot(ax, plot_df, title, title_fs, label_fs, tick_fs, global_max_logp):
    if plot_df.empty:
        ax.text(0.5, 0.5, "No significant terms", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=title_fs, fontweight='bold')
        return None

    scatter = ax.scatter(x=plot_df['Combined Score'], 
                         y=plot_df['Term'], 
                         s=plot_df['Gene_Count'] * 70, 
                         c=plot_df['logP'], 
                         cmap='YlOrRd', 
                         edgecolors="0.2", 
                         linewidth=0.8,
                         vmin=0, vmax=global_max_logp)

    ax.set_xlabel('Combined Score', fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs, pad=15, fontweight='bold')
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)
    ax.invert_yaxis()
    return scatter

# ================= 4. 主流程 =================
if __name__ == "__main__":
    df_genes = pd.read_csv(INPUT_FILE)
    genes = df_genes['gene'].dropna().astype(str).str.strip().unique().tolist()
    
    # 数据库配置：增加了 Molecular Function (MF)
    db_configs = {
        "BP": ['GO_Biological_Process_2023'],
        "CC": ['GO_Cellular_Component_2023'],
        "MF": ['GO_Molecular_Function_2023'],
        "KEGG": ['KEGG_2021_Human']
    }
    
    # 运行分析并获取绘图数据
    dfs = {k: run_enrichment_and_save(genes, k, v) for k, v in db_configs.items()}
    
    # 获取全局最大 -log10P
    all_logp = [df['logP'].max() for df in dfs.values() if not df.empty]
    global_max_logp = max(all_logp) if all_logp else 5

    # 创建大图 (增加宽度以容纳4个图)
    fig = plt.figure(figsize=(26, 10))
    # 布局比例：BP, CC, MF, KEGG
    gs = GridSpec(nrows=1, ncols=4, width_ratios=[2, 2, 2, 3]) 

    # 绘制四个子图
    sc_list = []
    sc_list.append(draw_dotplot_subplot(fig.add_subplot(gs[0, 0]), dfs["BP"], 'A. Biological Process', 13, 11, 9, global_max_logp))
    sc_list.append(draw_dotplot_subplot(fig.add_subplot(gs[0, 1]), dfs["CC"], 'B. Cellular Component', 13, 11, 9, global_max_logp))
    sc_list.append(draw_dotplot_subplot(fig.add_subplot(gs[0, 2]), dfs["MF"], 'C. Molecular Function', 13, 11, 9, global_max_logp))
    sc_list.append(draw_dotplot_subplot(fig.add_subplot(gs[0, 3]), dfs["KEGG"], 'D. KEGG Pathways', 16, 13, 11, global_max_logp))

    # 添加统一颜色条
    valid_sc = next((sc for sc in reversed(sc_list) if sc is not None), None)
    if valid_sc:
        cbar_ax = fig.add_axes([0.92, 0.3, 0.012, 0.4]) 
        cbar = fig.colorbar(valid_sc, cax=cbar_ax)
        cbar.set_label('-log10(P-value)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.9, wspace=0.45) 
    
    # 保存大图
    save_path = os.path.join(OUTPUT_DIR, COMBINED_FIGURE_NAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✨ 全部完成！")
    print(f"📊 合并大图: {save_path}")
    print(f"📂 原始Excel数据也已保存在同目录下。")
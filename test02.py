import pandas as pd
import os

def count_and_merge(root_dir='.'):
    results = []

    merged_all_years = []  # 保存所有年份两份问卷都有的人的数据

    for subdir, _, files in os.walk(root_dir):
        # 找每年的 DPQ 和 KIQ 文件
        dpq_file = kiq_file = None
        for file in files:
            if 'DPQ' in file and file.endswith('_clean.csv'):
                dpq_file = os.path.join(subdir, file)
            if 'KIQ' in file and file.endswith('_clean.csv'):
                kiq_file = os.path.join(subdir, file)

        if not dpq_file or not kiq_file:
            continue  # 如果有一份缺失就跳过

        # 读取数据
        dpq_df = pd.read_csv(dpq_file)
        kiq_df = pd.read_csv(kiq_file)

        # 保留有效 SEQN（至少一列非空）
        dpq_valid = dpq_df.dropna(subset=[c for c in dpq_df.columns if c != 'SEQN'], how='all')
        kiq_valid = kiq_df.dropna(subset=[c for c in kiq_df.columns if c != 'SEQN'], how='all')

        # SEQN 集合
        dpq_seqs = set(dpq_valid['SEQN'])
        kiq_seqs = set(kiq_valid['SEQN'])

        # 两份都有的人
        both_seqs = dpq_seqs & kiq_seqs

        # 统计
        results.append({
            'year': os.path.basename(subdir),
            'dpq_count': len(dpq_seqs),
            'kiq_count': len(kiq_seqs),
            'both_count': len(both_seqs),
            'both_vs_dpq': len(both_seqs)/len(dpq_seqs) if dpq_seqs else 0,
            'both_vs_kiq': len(both_seqs)/len(kiq_seqs) if kiq_seqs else 0
        })

        # 合并数据
        dpq_both = dpq_valid[dpq_valid['SEQN'].isin(both_seqs)]
        kiq_both = kiq_valid[kiq_valid['SEQN'].isin(both_seqs)]

        merged = pd.merge(dpq_both, kiq_both, on='SEQN', how='inner')
        merged_all_years.append(merged)

    # 整体合并
    final_merged = pd.concat(merged_all_years, ignore_index=True)
    stats_df = pd.DataFrame(results)

    return stats_df, final_merged

# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    stats, merged_data = count_and_merge(root_dir='.')  # 指定你的数据根目录
    print("按年份统计：")
    print(stats)

    # 保存结果
    stats.to_csv('dpq_kiq_stats_by_year.csv', index=False)
    merged_data.to_csv('dpq_kiq_merged_all_years.csv', index=False)
    print("清洗并合并的表格已保存。")

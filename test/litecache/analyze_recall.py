import pandas as pd
import argparse

def analyze_token_prefetch_profiling(csv_path):
    # 1. 加载数据
    df = pd.read_csv(csv_path)
    
    print("="*60)
    print("🎯 Token 级别预取与 Overlap 精确度评估报告")
    print("="*60)

    # 过滤掉无效数据 (例如 layer 0 显然没有参与调度，selected_tokens = 0)
    valid_df = df[df['selected_tokens'] > 0]
    
    if valid_df.empty:
        print("未发现有效调度数据。")
        return

    # 2. 全局核心指标计算
    total_selected = valid_df['selected_tokens'].sum()
    total_hit = valid_df['hit_tokens'].sum()
    total_recalled = valid_df['recalled_tokens'].sum()
    
    global_hit_rate = total_hit / total_selected if total_selected > 0 else 0
    
    # 将 Bytes 转换为 MB
    total_prefetch_mb = valid_df['prefetch_h2d_bytes'].sum() / (1024**2)
    total_offload_mb = valid_df['offload_d2h_bytes'].sum() / (1024**2)
    est_prefetch_mb = valid_df['overlap_prefetch_h2d_bytes_est'].sum() / (1024**2)
    est_recall_mb = valid_df['overlap_recall_h2d_bytes_est'].sum() / (1024**2)

    print("\n📊 [全局统计]")
    print(f"总计需求 Tokens:     {total_selected:,}")
    print(f"总计命中 Tokens:     {total_hit:,}")
    print(f"全局真实命中率:      {global_hit_rate:.2%}")
    print(f"总 Prefetch 传输量:  {total_prefetch_mb:.2f} MB")
    print(f"总 Offload 传输量:   {total_offload_mb:.2f} MB")
    
    # 3. 预取算法重叠度评估 (Mean Precision & Recall)
    # 取均值来评估预测算法的平均表现
    avg_precision = valid_df['overlap_mean_precision'].mean()
    avg_recall = valid_df['overlap_mean_recall'].mean()
    
    print("\n🎯 [预测算法评估 (Overlap Metrics)]")
    print(f"平均预取准确率 (Precision): {avg_precision:.2%} (预取进来的有多少被真正用到了)")
    print(f"平均预取召回率 (Recall):    {avg_recall:.2%} (实际用到的有多少被成功预取了)")
    print(f"估计预取开销 (Est H2D):     {est_prefetch_mb:.2f} MB")
    print(f"估计动态召回开销 (Est H2D): {est_recall_mb:.2f} MB")

    # 4. Layer 级深入分析
    print("\n🔍 [Layer 级特征剖析 (按准确率 Precision 升序排列，寻找预测最差的层)]")
    layer_group = valid_df.groupby('layer').agg(
        avg_hit_rate=('hit_rate', 'mean'),
        avg_precision=('overlap_mean_precision', 'mean'),
        avg_recall=('overlap_mean_recall', 'mean'),
        total_prefetch_mb=('prefetch_h2d_bytes', lambda x: x.sum() / (1024**2))
    ).reset_index()

    # 找出预测表现最差的前 5 层
    worst_layers = layer_group.sort_values('avg_precision').head(5)
    
    # 格式化输出
    worst_layers['avg_hit_rate'] = worst_layers['avg_hit_rate'].apply(lambda x: f"{x:.2%}")
    worst_layers['avg_precision'] = worst_layers['avg_precision'].apply(lambda x: f"{x:.2%}")
    worst_layers['avg_recall'] = worst_layers['avg_recall'].apply(lambda x: f"{x:.2%}")
    worst_layers['total_prefetch_mb'] = worst_layers['total_prefetch_mb'].apply(lambda x: f"{x:.2f}")
    
    print(worst_layers.to_string(index=False))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Token-level KV Cache Prefetching data.")
    parser.add_argument("--file", type=str, default="token_data.csv", help="Path to the profiling CSV file.")
    args = parser.parse_args()
    
    analyze_token_prefetch_profiling(args.file)
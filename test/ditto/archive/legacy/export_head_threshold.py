import pandas as pd

# 读取数据
df = pd.read_csv('/speedup/logs-perf-from32k/ditto_head_thresholds_seq32k.csv')

# 提取核心列
threshold_data = df[['layer_idx', 'head_idx', 'reuse_threshold']]

# 转换为多维字典形式 {(layer_idx, head_idx): reuse_threshold}，方便在推理时按坐标 O(1) 查找
threshold_dict = df.set_index(['layer_idx', 'head_idx'])['reuse_threshold'].to_dict()

# 测试访问 Layer 0, Head 3 的阈值
print(f"L0H3 Threshold: {threshold_dict.get((0, 3))}")
# 可视化模块

## 模块说明

可视化模块负责生成相似度计算和聚类分析的可视化结果。该模块支持多种可视化方式，并提供了灵活的配置选项。

## 主要功能

1. 相似度可视化：
   - 相似度热力图：直观展示标题间的相似度分布
   - 相似度分布图：展示相似度值的统计分布
   - 支持多种相似度计算方法

2. 聚类可视化：
   - 聚类散点图：使用t-SNE降维展示聚类结果
   - 聚类大小分布图：展示各个簇的大小分布
   - 支持多种聚类方法

3. 批量处理：
   - 支持处理多个年份的数据
   - 自动保存所有可视化结果
   - 提供进度和错误日志

## 实现细节

1. 相似度热力图：
   - 使用seaborn的heatmap函数
   - 支持显示标题文本
   - 可调整图像大小和颜色方案

2. 相似度分布图：
   - 结合直方图和核密度估计
   - 自动计算最优分箱数
   - 添加网格线增强可读性

3. 聚类散点图：
   - 使用t-SNE进行降维
   - 支持显示标题标注
   - 添加颜色条显示簇的标识

4. 聚类大小分布图：
   - 使用条形图展示
   - 显示具体数值标签
   - 支持自定义颜色方案

## 输入数据

模块从 `results` 和 `data/processed` 目录读取以下文件：

1. 相似度矩阵：
   - `cosine_similarity_{method}_{year1}_{year2}.npz`
   - `edit_distance_similarity_{year1}_{year2}.npz`

2. 聚类结果：
   - `clusters_{method}_{year}.csv`

3. 向量数据：
   - `tfidf_vectors_{year}.npz`
   - `word2vec_vectors_{year}.npy`

## 输出结果

模块将可视化结果保存到 `results/visualizations` 目录：

1. 相似度可视化：
   - `similarity_heatmap_{method}_{year1}_{year2}.png`
   - `similarity_distribution_{method}_{year1}_{year2}.png`

2. 聚类可视化：
   - `cluster_scatter_{method}_{year}.png`
   - `cluster_sizes_{method}_{year}.png`

## 配置说明

在 `src/config.py` 中设置可视化参数：

```python
VISUALIZATION_CONFIG = {
    "heatmap": {
        "figsize": (12, 10),
        "cmap": "YlOrRd",
        "dpi": 300
    },
    "distribution": {
        "figsize": (10, 6),
        "bins": 50,
        "dpi": 300
    },
    "scatter": {
        "figsize": (12, 8),
        "random_state": 42,
        "dpi": 300
    },
    "sizes": {
        "figsize": (10, 6),
        "dpi": 300
    }
}
```

## 使用示例

```python
from src.visualization.visualizer import Visualizer

# 初始化可视化器
visualizer = Visualizer()

# 生成单个热力图
visualizer.plot_similarity_heatmap(
    year1=2020,
    year2=2021,
    method='tfidf',
    output_file='heatmap.png'
)

# 生成单个聚类散点图
visualizer.plot_cluster_scatter(
    year=2020,
    method='word2vec',
    output_file='scatter.png'
)

# 批量处理所有可视化任务
years = [2020, 2021, 2022, 2023]
success = visualizer.process_visualizations(years)
```

## 注意事项

1. 数据准备：
   - 确保已完成相似度计算和聚类分析
   - 检查输入文件的完整性
   - 验证数据格式的正确性

2. 性能优化：
   - 大规模数据可视化时注意内存使用
   - 适当调整图像大小和DPI
   - 使用批处理避免内存溢出

3. 可视化质量：
   - 调整配置参数以获得最佳效果
   - 确保图例和标签清晰可读
   - 选择合适的颜色方案

4. 存储管理：
   - 定期清理临时文件
   - 压缩历史图像文件
   - 监控磁盘空间使用

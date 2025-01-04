# 可视化模块

## 模块说明

可视化模块负责生成相似度计算和聚类分析的可视化结果。该模块支持多种可视化方式，并提供了灵活的配置选项。

### 主要功能

1. 相似度可视化：
   - 相似度热力图：直观展示标题间的相似度分布
   - 相似度分布图：展示相似度值的统计分布

2. 聚类可视化：
   - 聚类散点图：使用t-SNE降维展示聚类结果
   - 聚类大小分布图：展示各个簇的大小分布

3. 批量处理：
   - 支持处理多个年份的数据
   - 支持多种相似度计算方法
   - 自动保存所有可视化结果

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

可以通过配置文件调整以下参数：

1. 热力图配置：
   - 图像大小
   - 颜色方案
   - DPI设置

2. 分布图配置：
   - 图像大小
   - 分箱数量
   - DPI设置

3. 聚类可视化配置：
   - 图像大小
   - 颜色方案
   - 随机种子
   - DPI设置

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
years = [2020, 2021, 2022, 2023, 2024]
success = visualizer.process_visualizations(years)
```

## 注意事项

1. 确保已完成相似度计算和聚类分析
2. 大规模数据可视化时注意内存使用
3. 调整配置参数以获得最佳可视化效果
4. 建议使用高DPI设置以获得高质量图像
5. 处理大量数据时可能需要较长时间

# 聚类分析模块

## 模块说明

聚类分析模块负责对向量化后的论文标题进行聚类分析，发现标题之间的内在关联，并生成聚类结果和评估指标。

## 主要功能

1. 聚类分析：
   - K-means聚类算法
   - 自动确定最优聚类数
   - 支持多种向量表示方法

2. 聚类评估：
   - 轮廓系数评估
   - 簇内距离计算
   - 簇间距离分析

3. 结果分析：
   - 聚类结果统计
   - 簇大小分布
   - 聚类中心计算

## 实现细节

1. 聚类算法：
   - 使用scikit-learn的KMeans实现
   - 支持多次随机初始化
   - 自动选择最佳结果

2. 评估方法：
   - 计算轮廓系数
   - 计算簇内平均距离
   - 分析簇间距离分布

3. 结果处理：
   - 保存聚类标签
   - 计算聚类中心
   - 生成评估报告

## 输入数据格式

来源：
- `data/processed/tfidf_vectors_{year}.npz`
- `data/processed/word2vec_vectors_{year}.npy`
- `results/cosine_similarity_{method}_{year1}_{year2}.npz`
- `results/edit_distance_similarity_{year1}_{year2}.npz`

## 输出数据格式

聚类结果将保存在 `results` 目录下。

### 1. 聚类结果
文件名：`clusters_{method}_{year}.csv`

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| id | int | 论文唯一标识符 | 1 |
| title | string | 论文标题 | 基于深度学习的文本分类研究 |
| cluster_id | int | 聚类标签 | 0 |
| distance_to_center | float | 到聚类中心的距离 | 0.15 |

### 2. 聚类中心
文件名：`cluster_centers_{method}_{year}.npy`
- 使用 numpy.save 保存的矩阵
- 每行对应一个聚类中心的向量

### 3. 聚类评估指标
文件名：`clustering_metrics_{method}_{year}.json`

```json
{
    "n_clusters": int,
    "silhouette_score": float,
    "calinski_harabasz_score": float,
    "davies_bouldin_score": float,
    "cluster_sizes": {
        "0": int,
        "1": int,
        ...
    },
    "inertia": float
}
```

### 数据要求
- 聚类标签从0开始连续编号
- 评估指标必须包含至少三种不同的度量方法
- 可视化图表必须包含图例和轴标签
- 聚类结果必须包含到聚类中心的距离信息

## 配置说明

在 `src/config.py` 中设置聚类参数：

```python
CLUSTERING_CONFIG = {
    "kmeans": {
        "n_clusters": 10,     # 聚类数量
        "n_init": 10,         # 初始化次数
        "max_iter": 300,      # 最大迭代次数
        "random_state": 42    # 随机种子
    },
    "evaluation": {
        "min_clusters": 2,    # 最小聚类数
        "max_clusters": 20,   # 最大聚类数
        "step": 2            # 聚类数步长
    }
}
```

## 使用示例

```python
from src.clustering.clusterer import Clusterer

# 初始化聚类器
clusterer = Clusterer()

# 对单个年份数据进行聚类
clusterer.cluster_year(2024, method='tfidf')

# 批量处理多个年份
years = [2020, 2021, 2022, 2023]
clusterer.cluster_years(years)

# 评估最优聚类数
optimal_k = clusterer.find_optimal_k(vectors, k_range=range(2, 21))
```

## 注意事项

1. 数据准备：
   - 确保向量数据已正确生成
   - 检查向量维度的一致性
   - 预处理异常值和缺失值

2. 性能优化：
   - 合理设置聚类数量
   - 优化迭代次数
   - 使用批处理处理大规模数据

3. 结果评估：
   - 综合多个评估指标
   - 分析簇的质量
   - 验证聚类结果的合理性

4. 可视化：
   - 使用降维方法展示结果
   - 添加必要的图例说明
   - 保存高质量的可视化图表

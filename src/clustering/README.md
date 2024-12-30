# 聚类分析模块

## 输入数据格式
- 来源：
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

### 4. 聚类可视化
文件名：`cluster_visualization_{method}_{year}.png`
- 使用 t-SNE 降维后的聚类结果可视化
- 分辨率：1200x800
- DPI：300
- 格式：PNG

### 数据要求
- 聚类标签从0开始连续编号
- 评估指标必须包含至少三种不同的度量方法
- 可视化图表必须包含图例和轴标签
- 聚类结果必须包含到聚类中心的距离信息

# 相似度计算模块

## 输入数据格式
- 来源：
  - `data/processed/tfidf_vectors_{year}.npz`
  - `data/processed/word2vec_vectors_{year}.npy`
  - `data/processed/cleaned_titles_{year}.csv`（用于编辑距离计算）

## 输出数据格式

相似度计算结果将保存在 `results` 目录下。

### 1. 余弦相似度矩阵
文件名：`cosine_similarity_{method}_{year1}_{year2}.npz`
- method: 可选 'tfidf' 或 'word2vec'
- 使用 scipy.sparse.save_npz 保存的稀疏矩阵
- 矩阵大小：[n_samples_year1, n_samples_year2]
- 值域：[-1, 1]

### 2. 编辑距离相似度矩阵
文件名：`edit_distance_similarity_{year1}_{year2}.npz`
- 使用 scipy.sparse.save_npz 保存的稀疏矩阵
- 矩阵大小：[n_samples_year1, n_samples_year2]
- 值域：[0, 1]（已归一化的相似度，1表示完全相同）

### 3. 相似度计算元数据
文件名：`similarity_metadata_{year1}_{year2}.json`

```json
{
    "cosine_similarity_tfidf": {
        "shape": [n_samples_year1, n_samples_year2],
        "sparsity": float,
        "mean": float,
        "std": float
    },
    "cosine_similarity_word2vec": {
        "shape": [n_samples_year1, n_samples_year2],
        "sparsity": float,
        "mean": float,
        "std": float
    },
    "edit_distance_similarity": {
        "shape": [n_samples_year1, n_samples_year2],
        "sparsity": float,
        "mean": float,
        "std": float
    }
}
```

### 数据要求
- 相似度矩阵必须是对称的（当year1 = year2时）
- 稀疏矩阵只保存相似度大于阈值的值（默认阈值：0.5）
- 必须包含元数据文件，描述相似度矩阵的统计信息
- 编辑距离相似度需要归一化到[0,1]区间

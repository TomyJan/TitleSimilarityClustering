# 文本向量化模块

## 输入数据格式
- 来源：`data/processed/cleaned_titles_{year}.csv`
- 格式：参考预处理模块的输出格式规范

## 输出数据格式

向量化后的数据将保存在 `data/processed` 目录下。

### 1. TF-IDF 向量
文件名：`tfidf_vectors_{year}.npz`
- 使用 scipy.sparse.save_npz 保存的稀疏矩阵
- 每行对应一个标题的 TF-IDF 向量
- 保存对应的特征词典：`tfidf_vocabulary.json`

### 2. Word2Vec 向量
文件名：`word2vec_vectors_{year}.npy`
- 使用 numpy.save 保存的密集矩阵
- 每行对应一个标题的词向量（取所有词向量的平均值）
- 保存训练好的模型：`word2vec_model.bin`

### 3. 向量元数据
文件名：`vector_metadata_{year}.json`

```json
{
    "tfidf": {
        "shape": [n_samples, n_features],
        "vocabulary_size": int,
        "sparse": true
    },
    "word2vec": {
        "shape": [n_samples, vector_dim],
        "vocabulary_size": int,
        "vector_dim": int,
        "sparse": false
    }
}
```

### 数据要求
- 所有数值向量必须是标准化的（L2范数归一化）
- 向量维度在不同年份的数据之间必须保持一致
- 特征词典在所有年份间共享
- 必须包含元数据文件，描述向量的基本信息

# 文本向量化模块

## 输入数据格式
- 来源：`data/processed/cleaned_titles_{year}.csv`
- 格式：参考预处理模块的输出格式规范

## 输出数据格式

向量化后的数据将保存在 `data/processed` 目录下。

### 1. TF-IDF向量
文件名：`tfidf_vectors_{year}.npz`
- 使用scipy.sparse.save_npz保存稀疏矩阵
- 每行对应一个标题的TF-IDF向量
- 同时保存特征词列表：`tfidf_features.json`

### 2. Word2Vec向量
文件名：`word2vec_vectors_{year}.npy`
- 使用numpy.save保存密集矩阵
- 每行对应一个标题的Word2Vec向量(通过对词向量取平均得到)
- 同时保存词向量模型：`word2vec_model.bin`

### 3. 字符级编码向量
文件名：`char_vectors_{year}.npz`
- 使用scipy.sparse.save_npz保存稀疏矩阵
- 每行对应一个标题的字符级one-hot编码
- 同时保存字符映射表：`char_features.json`

### 数据要求
- 所有向量必须是二维数组，形状为 (n_samples, n_features)
- 向量化过程中的配置参数(如n_features)需要保存在对应的config文件中
- 对于每种向量化方法，需要保证所有年份的数据使用相同的特征空间

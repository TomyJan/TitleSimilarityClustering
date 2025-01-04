# 文本向量化模块

## 功能说明
本模块负责将预处理后的论文标题转换为向量表示，支持多种向量化方法。

## 输入数据格式
- 来源：`data/processed/cleaned_titles_{year}.csv`
- 格式：参考预处理模块的输出格式规范
- 必需字段：
  - id: 论文ID
  - title: 原始标题
  - cleaned_tokens: 预处理后的分词结果（空格分隔）
  - year: 论文年份

## 输出数据格式

向量化后的数据将保存在 `data/processed` 目录下。

### 1. TF-IDF向量
文件名：`tfidf_vectors_{year}.npz`
- 使用scipy.sparse.save_npz保存稀疏矩阵
- 每行对应一个标题的TF-IDF向量
- 同时保存特征词列表：`tfidf_features.json`
- 向量维度由配置文件中的 `max_features` 参数决定

### 2. Word2Vec向量
文件名：`word2vec_vectors_{year}.npy`
- 使用numpy.save保存密集矩阵
- 每行对应一个标题的Word2Vec向量(通过对词向量取平均得到)
- 同时保存词向量模型：`word2vec_model.bin`
- 向量维度由配置文件中的 `vector_size` 参数决定

### 3. 字符级编码向量
文件名：`char_vectors_{year}.npz`
- 使用scipy.sparse.save_npz保存稀疏矩阵
- 每行对应一个标题的字符级one-hot编码
- 同时保存字符映射表：`char_features.json`
- 向量维度等于字符表大小

### 数据要求
- 所有向量必须是二维数组，形状为 (n_samples, n_features)
- 向量化过程中的配置参数(如n_features)需要保存在对应的config文件中
- 对于每种向量化方法，需要保证所有年份的数据使用相同的特征空间

## 配置说明

在 `src/config.py` 中设置向量化参数：

```python
VECTORIZATION_CONFIG = {
    "tfidf": {
        "max_features": 5000,  # 最大特征数
        "min_df": 2,          # 最小文档频率
        "max_df": 0.95        # 最大文档频率
    },
    "word2vec": {
        "vector_size": 100,   # 词向量维度
        "window": 5,          # 上下文窗口大小
        "min_count": 2        # 最小词频
    }
}
```

## 使用说明

### 1. 直接运行
最简单的方式是运行项目根目录下的脚本：
```bash
python run_vectorization.py
```

### 2. 在代码中使用
```python
from src.vectorization.vectorizer import Vectorizer

# 初始化向量化器
vectorizer = Vectorizer()

# 处理单个文件
vectorizer.process_file("data/processed/cleaned_titles_2024.csv")

# 处理所有文件
vectorizer.process_all()
```

### 3. 单独使用各种向量化方法
```python
# TF-IDF向量化
tfidf_vectors = vectorizer.fit_tfidf(texts)
new_tfidf_vectors = vectorizer.transform_tfidf(new_texts)

# Word2Vec向量化
word2vec_vectors = vectorizer.fit_word2vec(tokenized_texts)
new_word2vec_vectors = vectorizer.transform_word2vec(new_tokenized_texts)

# 字符级向量化
char_vectors = vectorizer.fit_char_vectors(texts)
new_char_vectors = vectorizer.transform_char_vectors(new_texts)
```

## 注意事项

1. **内存使用**
   - TF-IDF和字符级向量使用稀疏矩阵存储，节省内存
   - Word2Vec向量使用密集矩阵存储，需要更多内存
   - 处理大规模数据时注意监控内存使用

2. **特征空间**
   - 所有年份的数据共享同一个特征空间
   - 更新特征空间后需要重新处理所有数据
   - 特征文件需要与向量文件一起保存

3. **错误处理**
   - 输入文件格式错误会抛出 `VectorizationError`
   - 向量化器未训练就调用transform会抛出异常
   - 确保输入数据的完整性和格式正确性

4. **性能优化**
   - 使用适当的 `max_features` 控制向量维度
   - 根据数据规模调整 `min_df` 和 `max_df`
   - 需要时可以使用降维技术减少向量维度

## 开发计划

1. **待实现功能**
   - 支持增量训练
   - 添加更多向量化方法
   - 实现向量降维功能

2. **优化方向**
   - 提高大规模数据处理效率
   - 优化内存使用
   - 增强错误处理机制

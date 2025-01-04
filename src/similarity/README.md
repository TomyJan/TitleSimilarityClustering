# 相似度计算模块

## 模块说明

相似度计算模块负责计算不同年份论文标题之间的相似度。该模块支持多种相似度计算方法，并提供了详细的统计信息。

### 主要功能

1. 支持多种相似度计算方法：
   - TF-IDF 向量的余弦相似度：基于词频-逆文档频率的文本相似度
   - Word2Vec 向量的余弦相似度：基于词向量的语义相似度
   - 编辑距离相似度：基于字符级别的文本相似度

2. 批量处理功能：
   - 支持同一年份内的标题相似度计算
   - 支持不同年份之间的标题相似度计算
   - 自动处理所有可用年份的数据

3. 结果统计与保存：
   - 计算相似度矩阵的基本统计信息（形状、稀疏度、均值、标准差）
   - 使用稀疏矩阵格式保存结果，节省存储空间
   - 生成详细的元数据文件

## 实现细节

1. 余弦相似度计算：
   - 对向量进行L2归一化
   - 处理零范数情况
   - Word2Vec相似度额外进行[0,1]区间映射

2. 编辑距离相似度计算：
   - 使用Levenshtein距离
   - 基于总字符长度的归一化
   - 相似度值映射到[0,1]区间

3. 相似度阈值：
   - 余弦相似度阈值：0.5（范围：[0, 1]）
   - 编辑距离相似度阈值：0.6（范围：[0, 1]）

## 输入数据

模块从 `data/processed` 目录读取以下文件：

1. TF-IDF 向量：
   - 文件格式：`tfidf_vectors_{year}.npz`
   - 数据类型：scipy.sparse.csr_matrix

2. Word2Vec 向量：
   - 文件格式：`word2vec_vectors_{year}.npy`
   - 数据类型：numpy.ndarray

3. 原始标题数据（用于编辑距离计算）：
   - 文件格式：`cleaned_titles_{year}.csv`
   - 必需列：`id`, `title`

## 输出结果

模块将计算结果保存到 `results` 目录：

1. 相似度矩阵：
   - 余弦相似度：`cosine_similarity_{method}_{year1}_{year2}.npz`
   - 编辑距离：`edit_distance_similarity_{year1}_{year2}.npz`
   - 格式：scipy.sparse.csr_matrix

2. 元数据文件：
   - 文件名：`similarity_metadata_{year1}_{year2}.json`
   - 包含内容：
     - 矩阵形状
     - 稀疏度
     - 均值
     - 标准差

## 性能优化

1. 使用稀疏矩阵存储：
   - 仅存储高于阈值的相似度值
   - 显著减少存储空间需求
   - 提高数据加载和处理速度

2. 向量化计算：
   - 使用 NumPy 和 SciPy 进行向量化计算
   - 避免使用循环，提高计算效率

3. 内存优化：
   - 分批处理大规模数据
   - 及时释放不需要的数据

## 使用示例

```python
from src.similarity.calculator import SimilarityCalculator

# 初始化计算器
calculator = SimilarityCalculator()

# 计算指定年份之间的相似度
similarity_matrix, stats = calculator.calculate_similarity(
    year1=2020,
    year2=2021,
    method='tfidf'  # 或 'word2vec', 'edit_distance'
)

# 批量处理多个年份
years = [2020, 2021, 2022, 2023, 2024]
success = calculator.process_years(years)
```

## 注意事项

1. 确保已运行向量化模块，生成所需的向量文件
2. 不同年份的向量维度必须一致
3. 处理大规模数据时注意内存使用
4. 建议定期清理结果目录，避免占用过多存储空间
5. 相似度阈值可以根据具体需求在配置文件中调整

# 标题相似度检测与聚类项目

本项目旨在实现论文标题的相似度分析与聚类，通过多种算法对比分析不同年份学生论文标题的相似性，并进行聚类分析。

## 项目目标

1. 实现论文标题的相似度计算
   - TF-IDF 余弦相似度
   - Word2Vec 余弦相似度
   - 编辑距离相似度
2. 对论文标题进行聚类分析
3. 比较不同相似度算法的效果
4. 优化代码实现以提高性能

## 项目架构

### 1. 数据处理
- [数据爬取模块](src/crawler/README.md)：使用 Python 爬虫获取论文标题数据
- [数据预处理模块](src/preprocessing/README.md)：使用 jieba 分词进行文本预处理
- 停用词处理：去除常见停用词，提高文本分析质量

### 2. 文本分析
- [文本向量化模块](src/vectorization/README.md)：实现多种文本向量化方法
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word2Vec
  - 字符级编码
- [相似度计算模块](src/similarity/README.md)：实现多种相似度算法
  - 余弦相似度：计算文本向量间的夹角余弦值
  - 编辑距离相似度：计算字符串转换的最小操作次数
  - 算法优劣对比分析
- [聚类分析模块](src/clustering/README.md)：使用 K-means 算法进行文本聚类
- [可视化模块](src/visualization/README.md)：生成相似度和聚类结果的可视化图表

## 算法比较

### TF-IDF 余弦相似度
- 优点：
  - 考虑词频和逆文档频率，能够反映词语的重要性
  - 计算效率高，适合大规模文本处理
  - 结果易于解释和理解
- 缺点：
  - 不考虑词序和语义关系
  - 对于短文本效果可能不够理想
  - 无法处理同义词和多义词

### Word2Vec 余弦相似度
- 优点：
  - 能够捕捉词语的语义关系
  - 可以处理同义词
  - 生成的向量具有良好的语义特性
- 缺点：
  - 需要大量训练数据
  - 计算资源消耗较大
  - 结果不如 TF-IDF 直观

### 编辑距离相似度
- 优点：
  - 直接比较字符串差异，结果直观
  - 不需要预训练或特征提取
  - 适合检测拼写错误和细微差异
- 缺点：
  - 计算复杂度较高
  - 不考虑语义信息
  - 对文本长度敏感

## 项目结构

```
TitleSimilarityClustering/
├── data/                   # 数据文件目录
│   ├── raw/               # 原始爬取数据
│   └── processed/         # 处理后的数据
├── src/                    # 源代码
│   ├── crawler/           # 数据爬取模块
│   ├── preprocessing/      # 数据预处理模块
│   ├── vectorization/      # 文本向量化模块
│   ├── similarity/         # 相似度计算模块
│   ├── clustering/         # 聚类分析模块
│   └── visualization/      # 可视化模块
├── results/                # 结果输出
│   └── visualizations/     # 可视化结果
└── requirements.txt        # 项目依赖
```

## 技术栈

- Python 3.8+
- pandas: 数据处理
- jieba: 中文分词
- numpy: 数值计算
- scikit-learn: 机器学习算法
- matplotlib & seaborn: 数据可视化
- gensim: Word2Vec模型
- scipy: 稀疏矩阵处理

## 使用说明

### 环境配置

1. Python 环境要求：
   ```bash
   Python 3.8+
   ```

2. 创建并激活虚拟环境：
   ```bash
   # 创建虚拟环境
   python -m venv venv
   
   # Windows 激活虚拟环境
   .\venv\Scripts\activate
   
   # Linux/Mac 激活虚拟环境
   source venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 运行流程

#### 方式一：一键运行
运行主流水线脚本，将依次执行所有处理步骤：
```bash
python scripts/run_pipeline.py
```

#### 方式二：逐步运行
如果需要分步执行或调试，可以按以下顺序运行各个步骤：

1. 数据获取：
   ```bash
   python scripts/run_crawler.py
   ```

2. 数据预处理：
   ```bash
   python scripts/run_preprocessor.py
   ```

3. 文本向量化：
   ```bash
   python scripts/run_vectorization.py
   ```

4. 相似度计算：
   ```bash
   python scripts/run_similarity.py
   ```

5. 聚类分析：
   ```bash
   python scripts/run_clustering.py
   ```

6. 可视化结果：
   ```bash
   python scripts/run_visualization.py
   ```

每个步骤的输出结果都会保存在相应的目录中。

## 分析结果

### 相似度分析
- 生成三种算法的相似度矩阵
- 提供相似度分布对比图
- 包含算法性能对比分析

### 聚类分析
- 基于不同相似度的聚类结果
- 聚类效果可视化
- 聚类统计信息

### 可视化报告
- 相似度热力图
- 聚类散点图
- 算法性能对比图
- 详细的分析报告

## 配置说明

所有配置参数都集中在 `src/config.py` 文件中：

1. 数据目录配置：
   - `RAW_DATA_DIR`：原始数据目录
   - `PROCESSED_DATA_DIR`：处理后数据目录
   - `RESULTS_DIR`：结果输出目录

2. 模块配置：
   - `CRAWLER_CONFIG`：爬虫配置
   - `PREPROCESSING_CONFIG`：预处理配置
   - `VECTORIZATION_CONFIG`：向量化配置
   - `SIMILARITY_CONFIG`：相似度计算配置
   - `CLUSTERING_CONFIG`：聚类配置
   - `VISUALIZATION_CONFIG`：可视化配置

详细配置说明请参考各模块文档。

## 输出结果

1. 数据文件：
   - 原始数据：`data/raw/thesis_titles_{year}.csv`
   - 预处理结果：`data/processed/cleaned_titles_{year}.csv`
   - 向量化结果：
     - `data/processed/tfidf_vectors_{year}.npz`
     - `data/processed/word2vec_vectors_{year}.npy`

2. 分析结果：
   - 相似度矩阵：`results/cosine_similarity_{method}_{year1}_{year2}.npz`
   - 聚类结果：`results/clusters_{method}_{year}.csv`
   - 可视化图表：`results/visualizations/`

## 注意事项

1. 运行顺序：必须按照预处理 -> 向量化 -> 相似度计算/聚类分析的顺序执行
2. 内存使用：处理大规模数据时注意内存占用
3. 存储空间：定期清理不需要的中间结果文件
4. 配置调整：可以根据具体需求调整配置参数
5. 日志查看：运行日志保存在 `logs` 目录下

## 开发计划

1. 功能优化：
   - 支持增量数据处理
   - 添加更多向量化方法
   - 优化聚类算法

2. 性能提升：
   - 优化内存使用
   - 提高处理速度
   - 支持并行计算

3. 可视化增强：
   - 添加交互式图表
   - 支持更多可视化方式
   - 优化图表样式

## 贡献指南

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MPL-2.0 许可证，详见 [LICENSE](LICENSE) 文件

本项目代码及文档完全使用 [Cursor](https://www.cursor.com/) 完成, 代码没有任何真人编写成份

# 标题相似度检测与聚类项目

本项目旨在实现论文标题的相似度分析与聚类，通过多种算法对比分析不同年级学生论文标题的相似性，并进行聚类分析。

## 项目目标

1. 实现论文标题的相似度计算
2. 对论文标题进行聚类分析
3. 比较不同相似度算法的效果
4. 优化代码实现以提高性能

## 技术实现路线

### 1. 数据处理
- CSV 文件处理：使用 Python 的 `pandas` 库进行 CSV 文件的读写操作
- 文本预处理：使用 `jieba` 分词库进行中文分词
- 停用词处理：去除常见停用词，提高文本分析质量

### 2. 文本向量化
实现多种文本向量化方法：
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word2Vec
- 字符级编码

### 3. 相似度计算
实现两种相似度算法：
- 余弦相似度：计算文本向量间的夹角余弦值
- 编辑距离相似度：计算将一个字符串转换成另一个所需的最小操作次数

### 4. 聚类分析
- 使用 K-means 算法进行文本聚类
- 分析不同年级论文标题的聚类结果

### 5. 性能优化
- 代码优化
- 算法效率提升
- 内存使用优化

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
│   └── clustering/         # 聚类分析模块
├── tests/                  # 测试文件
├── results/                # 结果输出
└── requirements.txt        # 项目依赖
```

## 技术栈

- Python 3.8+
- pandas: 数据处理
- jieba: 中文分词
- numpy: 数值计算
- scikit-learn: 机器学习算法
- matplotlib: 数据可视化
- gensim: Word2Vec模型
- scipy: 稀疏矩阵处理

## 实现步骤

1. **数据爬取**
    - 爬取论文标题数据

2. **数据预处理**
   - CSV 文件读取与处理
   - 文本清洗
   - 分词与停用词去除

3. **文本向量化**
   ```bash
   # 运行向量化处理
   python run_vectorization.py
   ```
   
   向量化处理会生成三种类型的向量表示：
   - TF-IDF向量 (`data/processed/tfidf_vectors_{year}.npz`)
   - Word2Vec向量 (`data/processed/word2vec_vectors_{year}.npy`)
   - 字符级向量 (`data/processed/char_vectors_{year}.npz`)
   
   同时会生成相应的特征文件：
   - TF-IDF特征词表 (`data/processed/tfidf_features.json`)
   - Word2Vec模型 (`data/processed/word2vec_model.bin`)
   - 字符映射表 (`data/processed/char_features.json`)

4. **相似度计算**
   - 实现余弦相似度算法
   - 实现编辑距离相似度算法
   - 计算不同年级论文标题间的相似度

5. **聚类分析**
   - K-means 聚类实现
   - 聚类结果分析与可视化

6. **性能优化**
   - 代码优化
   - 算法效率提升
   - 结果对比分析

## 预期成果

1. 完整的标题相似度分析系统
2. 不同相似度算法的对比分析报告
3. 聚类结果的可视化展示
4. 优化前后的性能对比报告

## 注意事项

1. 确保数据质量和预处理的准确性
2. 注意大规模数据处理时的性能问题
3. 保持代码的可维护性和可扩展性
4. 详细记录实验结果和优化过程

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

### 数据获取

运行爬虫获取论文数据：

1. 爬取指定时间范围的数据：
   ```bash
   python run_crawler.py
   ```
   爬取的时间长度在 run_crawler.py 中配置

爬虫运行说明：
- 数据将保存在 `data/raw/` 目录下，文件名格式为 `thesis_titles_{year}.csv`
- 每个年份的数据单独保存为一个 CSV 文件
- 数据包含以下字段：
  - id: 论文唯一标识符
  - title: 论文标题（已清洗）
  - publish_date: 发表日期（格式：YYYY-MM-DD）
  - author: 第一作者
  - major: 专业分类（固定为"信息科技"）

注意事项：
1. 默认爬取信息科技领域的论文
2. 数据格式要求：
   - 编码：UTF-8
   - 分隔符：逗号 (,)
   - 包含表头
   - 所有字段都不允许为空
   - 标题必须是规范的中文或英文，不包含特殊字符
3. 如遇到访问限制，可以：
   - 适当增加爬取间隔
   - 更换网络环境
   - 更新请求头信息

### 数据预处理

1. 运行预处理脚本：
   ```bash
   python run_preprocessor.py
   ```

预处理步骤说明：
1. 分词处理
   - 使用 jieba 进行中文分词
   - 输出文件：`data/processed/tokenized_titles_{year}.csv`
   - 包含字段：
     - id: 论文唯一标识符
     - title: 原始论文标题
     - tokens: 分词结果（空格分隔）
     - year: 论文年份

2. 停用词处理
   - 使用预定义的停用词表过滤无意义词语
   - 停用词表位置：`src/preprocessing/stopwords.txt`
   - 停用词来源：https://github.com/CharyHong/Stopwords/blob/main/stopwords_full.txt

3. 清洗结果
   - 输出文件：`data/processed/cleaned_titles_{year}.csv`
   - 包含字段：
     - id: 论文唯一标识符
     - title: 原始论文标题
     - cleaned_tokens: 去停用词后的分词结果（空格分隔）
     - year: 论文年份

注意事项：
1. 确保已安装必要的依赖：
   ```bash
   pip install jieba
   ```
2. 所有输出文件采用 UTF-8 编码
3. CSV 文件包含表头，使用逗号分隔
4. 分词结果中的词语使用单个空格分隔
5. 所有字段都不允许为空

### 文本向量化

1. 运行向量化脚本：
   ```bash
   python run_vectorization.py
   ```

   向量化处理会对2020-2024年的所有数据进行处理，生成三种不同的向量表示：
   - TF-IDF向量 (`data/processed/tfidf_vectors_{year}.npz`)
   - Word2Vec向量 (`data/processed/word2vec_vectors_{year}.npy`)
   - 字符级向量 (`data/processed/char_vectors_{year}.npz`)
   
   同时会生成相应的特征文件：
   - TF-IDF特征词表 (`data/processed/tfidf_features.json`)
   - Word2Vec模型 (`data/processed/word2vec_model.bin`)
   - 字符映射表 (`data/processed/char_features.json`)

   向量化原理：
   1. TF-IDF向量化
      - 计算词频(TF)和逆文档频率(IDF)
      - 使用稀疏矩阵存储高维向量
      - 通过配置控制特征词数量

   2. Word2Vec向量化
      - 训练词向量模型捕捉语义关系
      - 通过平均词向量得到文档向量
      - 使用密集矩阵存储低维向量

   3. 字符级向量化
      - 构建字符级的one-hot编码
      - 保持文本的字符级特征
      - 使用稀疏矩阵存储高维向量

   注意事项：
   1. 确保已完成数据预处理步骤
   2. 检查生成的向量文件大小是否合理
   3. 验证特征文件的完整性
   4. 注意内存使用，特别是处理大规模数据时

### 相似度计算

运行相似度计算脚本：
```bash
python run_similarity.py
```

相似度计算说明：
1. 支持三种计算方法：
   - TF-IDF向量的余弦相似度
   - Word2Vec向量的余弦相似度
   - 编辑距离相似度

2. 输出文件：
   - 相似度矩阵：`results/cosine_similarity_{method}_{year1}_{year2}.npz`
   - 编辑距离：`results/edit_distance_similarity_{year1}_{year2}.npz`
   - 统计信息：`results/similarity_metadata_{year1}_{year2}.json`

3. 注意事项：
   - 确保已完成向量化处理
   - 结果使用稀疏矩阵存储，节省空间
   - 自动处理所有年份组合
   - 可通过配置文件调整相似度阈值

# 数据预处理模块

## 模块说明

数据预处理模块负责对原始论文标题数据进行清洗、分词和标准化处理，为后续的向量化和相似度计算做准备。

## 主要功能

1. 文本清洗：
   - 去除特殊字符和标点符号
   - 统一格式（大小写、全半角等）
   - 去除多余空白字符

2. 中文分词：
   - 使用 jieba 进行中文分词
   - 自定义词典支持
   - 词性标注（可选）

3. 停用词处理：
   - 加载自定义停用词表
   - 过滤常见停用词
   - 去除低频词和高频词

## 实现细节

1. 文本清洗：
   - 使用正则表达式处理特殊字符
   - 统一编码为UTF-8
   - 规范化空白字符

2. 分词处理：
   - 配置 jieba 分词器
   - 加载自定义词典
   - 优化分词准确率

3. 停用词处理：
   - 维护停用词列表
   - 支持动态更新
   - 批量处理优化

## 输入数据格式

来源：`data/raw/thesis_titles_{year}.csv`
参考爬虫模块的输出格式规范。

## 输出数据格式

预处理后的数据将保存在 `data/processed` 目录下。

### 1. 分词结果
文件名：`tokenized_titles_{year}.csv`

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| id | int | 论文唯一标识符 | 1 |
| title | string | 原始论文标题 | 基于深度学习的文本分类研究 |
| tokens | string | 分词结果（空格分隔） | 基于 深度 学习 的 文本 分类 研究 |
| year | int | 论文年份 | 2014 |

### 2. 停用词表
文件名：`stopwords.txt`
- 每行一个停用词
- UTF-8 编码
- 不包含空行和重复词
- 停用词来源：https://github.com/CharyHong/Stopwords/blob/main/stopwords_full.txt

### 3. 清洗后的分词结果
文件名：`cleaned_titles_{year}.csv`

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| id | int | 论文唯一标识符 | 1 |
| title | string | 原始论文标题 | 基于深度学习的文本分类研究 |
| cleaned_tokens | string | 去停用词后的分词结果 | 深度 学习 文本 分类 研究 |
| year | int | 论文年份 | 2014 |

### 数据要求
- 编码：UTF-8
- 分隔符：逗号 (,)
- 包含表头
- 所有字段都不允许为空
- 分词结果中的词语间使用单个空格分隔

## 配置说明

在 `src/config.py` 中设置预处理参数：

```python
PREPROCESSING_CONFIG = {
    "min_title_length": 4,     # 最小标题长度
    "max_title_length": 100,   # 最大标题长度
    "min_token_length": 1,     # 最小词语长度
    "user_dict": "dict.txt",   # 自定义词典路径
    "stopwords": "stopwords.txt"  # 停用词表路径
}
```

## 使用示例

```python
from src.preprocessing.preprocessor import Preprocessor

# 初始化预处理器
preprocessor = Preprocessor()

# 处理单个文件
preprocessor.process_file("data/raw/thesis_titles_2024.csv")

# 处理所有文件
preprocessor.process_all()

# 单独使用分词功能
tokens = preprocessor.tokenize("基于深度学习的文本分类研究")
```

## 注意事项

1. 确保输入数据的编码格式正确
2. 定期更新停用词表和自定义词典
3. 注意特殊字符的处理
4. 保持分词结果的一致性
5. 定期检查预处理结果的质量

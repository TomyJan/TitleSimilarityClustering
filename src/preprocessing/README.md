# 数据预处理模块

## 输入数据格式
- 来源：`data/raw/thesis_titles_{year}.csv`
- 格式：参考爬虫模块的输出格式规范

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
- 停用词来源 https://github.com/CharyHong/Stopwords/blob/main/stopwords_full.txt

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

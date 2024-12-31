"""向量化模块配置"""

# 输入输出路径配置
INPUT_DIR = "data/processed"  # 预处理后的数据目录
OUTPUT_DIR = "data/processed"  # 向量化结果保存目录

# TF-IDF配置
TFIDF_CONFIG = {
    "max_features": 5000,  # 最大特征数
    "min_df": 2,  # 最小文档频率
    "max_df": 0.95,  # 最大文档频率
    "binary": False,  # 是否使用二值化
    "norm": "l2",  # 标准化方法
    "output_prefix": "tfidf_vectors_",  # 输出文件前缀
    "features_file": "tfidf_features.json"  # 特征词列表文件
}

# Word2Vec配置
WORD2VEC_CONFIG = {
    "vector_size": 100,  # 词向量维度
    "window": 5,  # 上下文窗口大小
    "min_count": 2,  # 最小词频
    "workers": 4,  # 训练线程数
    "sg": 1,  # 使用Skip-gram模型
    "epochs": 10,  # 训练轮数
    "output_prefix": "word2vec_vectors_",  # 输出文件前缀
    "model_file": "word2vec_model.bin"  # 模型保存文件
}

# 字符级编码配置
CHAR_CONFIG = {
    "max_features": 3000,  # 最大特征数
    "min_df": 2,  # 最小文档频率
    "max_df": 0.95,  # 最大文档频率
    "output_prefix": "char_vectors_",  # 输出文件前缀
    "features_file": "char_features.json"  # 字符映射表文件
}

# 通用配置
COMMON_CONFIG = {
    "random_state": 42,  # 随机数种子
    "n_jobs": -1  # 并行处理的作业数(-1表示使用所有CPU)
} 
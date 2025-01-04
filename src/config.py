"""项目统一配置文件"""
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录配置
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# 预处理模块配置
PREPROCESSING_CONFIG = {
    "stopwords_path": str(ROOT_DIR / "src" / "preprocessing" / "stopwords.txt"),
    "min_title_length": 2,  # 最小标题长度
    "max_title_length": 100,  # 最大标题长度
    "supported_languages": ["zh", "en"],  # 支持的语言
}

# 向量化模块配置
VECTORIZATION_CONFIG = {
    "tfidf": {
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95
    },
    "word2vec": {
        "vector_size": 100,
        "window": 5,
        "min_count": 2,
        "workers": 4
    }
}

# 相似度计算配置
SIMILARITY_CONFIG = {
    "methods": ["tfidf", "word2vec", "edit_distance"],
    "thresholds": {
        "cosine": 0.5,  # 降低余弦相似度阈值，以获得更多匹配
        "edit_distance": 0.6  # 保持编辑距离相似度阈值不变
    },
    "output_format": {
        "cosine": {
            "file_pattern": "cosine_similarity_{method}_{year1}_{year2}.npz",
            "value_range": [0, 1]  # 保持[0,1]范围
        },
        "edit_distance": {
            "file_pattern": "edit_distance_similarity_{year1}_{year2}.npz",
            "value_range": [0, 1]  # 归一化的编辑距离范围
        },
        "metadata": {
            "file_pattern": "similarity_metadata_{year1}_{year2}.json"
        }
    }
}

# 聚类配置
CLUSTERING_CONFIG = {
    "kmeans": {
        "n_clusters": 10,
        "random_state": 42
    },
    "output_format": {
        "clusters": {
            "file_pattern": "clusters_{method}_{year}.csv",
            "columns": ["id", "title", "cluster_id", "distance_to_center"]
        },
        "centers": {
            "file_pattern": "cluster_centers_{method}_{year}.npy"
        },
        "metrics": {
            "file_pattern": "clustering_metrics_{method}_{year}.json",
            "required_metrics": [
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score"
            ]
        },
        "visualization": {
            "file_pattern": "cluster_visualization_{method}_{year}.png",
            "resolution": [1200, 800],
            "dpi": 300
        }
    }
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'heatmap': {
        'figsize': (12, 10),
        'cmap': 'YlOrRd',
        'dpi': 300
    },
    'distribution': {
        'figsize': (10, 6),
        'bins': 50,
        'dpi': 300
    },
    'scatter': {
        'figsize': (12, 8),
        'random_state': 42,
        'dpi': 300
    },
    'sizes': {
        'figsize': (10, 6),
        'dpi': 300
    }
}

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(ROOT_DIR / "logs" / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

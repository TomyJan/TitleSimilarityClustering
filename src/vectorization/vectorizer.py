"""文本向量化实现"""
import os
import json
import logging
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from .config import (
    INPUT_DIR, OUTPUT_DIR,
    TFIDF_CONFIG, WORD2VEC_CONFIG, CHAR_CONFIG,
    COMMON_CONFIG
)

logger = logging.getLogger(__name__)

class BaseVectorizer(ABC):
    """向量化器基类"""
    
    def __init__(self, config: Dict):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """训练向量化模型
        
        Args:
            texts: 文本列表
        """
        pass
        
    @abstractmethod
    def transform(self, texts: List[str]) -> Union[np.ndarray, sparse.csr_matrix]:
        """将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量
        """
        pass
        
    def fit_transform(self, texts: List[str]) -> Union[np.ndarray, sparse.csr_matrix]:
        """训练并转换
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量
        """
        self.fit(texts)
        return self.transform(texts)
        
    @abstractmethod
    def save(self, output_dir: str) -> None:
        """保存模型和相关文件
        
        Args:
            output_dir: 输出目录
        """
        pass
        
    @abstractmethod
    def load(self, model_dir: str) -> None:
        """加载模型和相关文件
        
        Args:
            model_dir: 模型目录
        """
        pass

class TfidfVectorizer(BaseVectorizer):
    """TF-IDF向量化器"""
    
    def __init__(self, config: Dict = TFIDF_CONFIG):
        """初始化
        
        Args:
            config: TF-IDF配置
        """
        super().__init__(config)
        self.vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            binary=config['binary'],
            norm=config['norm']
        )
        
    def fit(self, texts: List[str]) -> None:
        """训练TF-IDF模型
        
        Args:
            texts: 文本列表
        """
        logger.info("开始训练TF-IDF模型...")
        self.vectorizer.fit(texts)
        logger.info(f"TF-IDF模型训练完成，特征维度: {len(self.vectorizer.get_feature_names_out())}")
        
    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """将文本转换为TF-IDF向量
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF向量矩阵
        """
        return self.vectorizer.transform(texts)
        
    def save(self, output_dir: str) -> None:
        """保存TF-IDF模型和特征词列表
        
        Args:
            output_dir: 输出目录
        """
        # 保存特征词列表
        features_path = os.path.join(output_dir, self.config['features_file'])
        features = self.vectorizer.get_feature_names_out().tolist()
        os.makedirs(output_dir, exist_ok=True)
        
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        logger.info(f"特征词列表已保存至: {features_path}")
        
    def load(self, model_dir: str) -> None:
        """加载TF-IDF模型和特征词列表
        
        Args:
            model_dir: 模型目录
        """
        # 加载特征词列表
        features_path = os.path.join(model_dir, self.config['features_file'])
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特征词列表文件不存在: {features_path}")
            
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
            
        # 重新构建向量化器
        self.vectorizer = TfidfVectorizer(
            vocabulary=features,
            max_features=self.config['max_features'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            binary=self.config['binary'],
            norm=self.config['norm']
        )
        logger.info(f"已加载特征词列表，特征维度: {len(features)}")
        
    def save_vectors(self, vectors: sparse.csr_matrix, year: int, output_dir: str) -> None:
        """保存向量化结果
        
        Args:
            vectors: 向量矩阵
            year: 年份
            output_dir: 输出目录
        """
        output_path = os.path.join(
            output_dir,
            f"{self.config['output_prefix']}{year}.npz"
        )
        os.makedirs(output_dir, exist_ok=True)
        sparse.save_npz(output_path, vectors)
        logger.info(f"向量已保存至: {output_path}")
        
    def process_file(self, input_path: str, output_dir: str) -> bool:
        """处理单个文件
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            # 读取数据
            df = pd.read_csv(input_path)
            if 'cleaned_tokens' not in df.columns:
                logger.error(f"输入文件缺少cleaned_tokens列: {input_path}")
                return False
                
            # 获取年份
            year = df['year'].iloc[0]
            
            # 向量化
            vectors = self.transform(df['cleaned_tokens'])
            
            # 保存结果
            self.save_vectors(vectors, year, output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            return False
            
    def process_all(self, input_dir: str, output_dir: str) -> bool:
        """处理所有文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            # 首先收集所有文本进行训练
            all_texts = []
            for filename in os.listdir(input_dir):
                if filename.startswith('cleaned_titles_') and filename.endswith('.csv'):
                    df = pd.read_csv(os.path.join(input_dir, filename))
                    all_texts.extend(df['cleaned_tokens'])
                    
            # 训练模型
            self.fit(all_texts)
            
            # 保存模型
            self.save(output_dir)
            
            # 处理每个文件
            success = True
            for filename in os.listdir(input_dir):
                if filename.startswith('cleaned_titles_') and filename.endswith('.csv'):
                    input_path = os.path.join(input_dir, filename)
                    logger.info(f"正在处理文件: {filename}")
                    if not self.process_file(input_path, output_dir):
                        success = False
                        
            return success
            
        except Exception as e:
            logger.error(f"批量处理文件时出错: {str(e)}")
            return False 
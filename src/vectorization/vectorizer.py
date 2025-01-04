#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy import sparse

logger = logging.getLogger(__name__)

class TitleVectorizer:
    """论文标题向量化类"""
    
    def __init__(self):
        """初始化向量化器"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95
        )
        self.word2vec_model = None
        self.char_vectorizer = None
        
    def _load_titles(self, year: int) -> pd.DataFrame:
        """加载标题数据
        
        Args:
            year: 年份
            
        Returns:
            包含标题的DataFrame
        """
        try:
            file_path = os.path.join("data", "preprocessed", f"cleaned_titles_{year}.csv")
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['id', 'title', 'cleaned_tokens']):
                raise ValueError(f"标题文件缺少必需的列: {file_path}")
            return df
        except Exception as e:
            logger.error(f"加载标题文件时出错: {str(e)}")
            raise
            
    def _save_vectors(self, vectors: np.ndarray, year: int, method: str) -> None:
        """保存向量化结果
        
        Args:
            vectors: 向量矩阵
            year: 年份
            method: 向量化方法
        """
        try:
            output_dir = os.path.join("data", "vectorized")
            os.makedirs(output_dir, exist_ok=True)
            
            if isinstance(vectors, sparse.spmatrix):
                output_path = os.path.join(output_dir, f"{method}_vectors_{year}.npz")
                sparse.save_npz(output_path, vectors)
            else:
                output_path = os.path.join(output_dir, f"{method}_vectors_{year}.npy")
                np.save(output_path, vectors)
                
            logger.info(f"向量化结果已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"保存向量化结果时出错: {str(e)}")
            raise
            
    def fit_tfidf(self, years: List[int]) -> None:
        """训练TF-IDF向量化器
        
        Args:
            years: 年份列表
        """
        try:
            # 收集所有年份的文本
            texts = []
            for year in years:
                df = self._load_titles(year)
                texts.extend(df['cleaned_tokens'].tolist())
                
            # 训练向量化器
            self.tfidf_vectorizer.fit(texts)
            logger.info(f"TF-IDF向量化器训练完成，特征数: {len(self.tfidf_vectorizer.get_feature_names_out())}")
            
        except Exception as e:
            logger.error(f"训练TF-IDF向量化器时出错: {str(e)}")
            raise
            
    def transform_tfidf(self, year: int) -> None:
        """使用TF-IDF方法向量化指定年份的标题
        
        Args:
            year: 年份
        """
        try:
            df = self._load_titles(year)
            vectors = self.tfidf_vectorizer.transform(df['cleaned_tokens'])
            self._save_vectors(vectors, year, 'tfidf')
            
        except Exception as e:
            logger.error(f"TF-IDF向量化时出错: {str(e)}")
            raise
            
    def fit_word2vec(self, years: List[int]) -> None:
        """训练Word2Vec模型
        
        Args:
            years: 年份列表
        """
        try:
            # 收集所有年份的分词结果
            sentences = []
            for year in years:
                df = self._load_titles(year)
                sentences.extend([tokens.split() for tokens in df['cleaned_tokens']])
                
            # 训练模型
            self.word2vec_model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4
            )
            logger.info(f"Word2Vec模型训练完成，词表大小: {len(self.word2vec_model.wv.key_to_index)}")
            
        except Exception as e:
            logger.error(f"训练Word2Vec模型时出错: {str(e)}")
            raise
            
    def transform_word2vec(self, year: int) -> None:
        """使用Word2Vec方法向量化指定年份的标题
        
        Args:
            year: 年份
        """
        try:
            df = self._load_titles(year)
            
            # 计算每个标题的平均词向量
            vectors = []
            for tokens in df['cleaned_tokens']:
                token_list = tokens.split()
                token_vectors = []
                for token in token_list:
                    if token in self.word2vec_model.wv:
                        token_vectors.append(self.word2vec_model.wv[token])
                if token_vectors:
                    vectors.append(np.mean(token_vectors, axis=0))
                else:
                    vectors.append(np.zeros(self.word2vec_model.vector_size))
                    
            vectors = np.array(vectors)
            self._save_vectors(vectors, year, 'word2vec')
            
        except Exception as e:
            logger.error(f"Word2Vec向量化时出错: {str(e)}")
            raise
            
    def vectorize_all(self, years: List[int], output_dir: str) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            # 训练向量化器和模型
            logger.info("开始训练向量化器和模型...")
            self.fit_tfidf(years)
            self.fit_word2vec(years)
            
            # 转换每个年份的数据
            for year in years:
                logger.info(f"正在处理 {year} 年的数据...")
                self.transform_tfidf(year)
                self.transform_word2vec(year)
                
            return True
            
        except Exception as e:
            logger.error(f"处理年份数据时出错: {str(e)}")
            return False

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class TitleSimilarityCalculator:
    """论文标题相似度计算类"""
    
    def __init__(self):
        """初始化相似度计算器"""
        pass
        
    def _load_vectors(self, year: int, method: str) -> np.ndarray:
        """加载向量化结果
        
        Args:
            year: 年份
            method: 向量化方法
            
        Returns:
            向量矩阵
        """
        try:
            input_dir = os.path.join("data", "vectorized")
            if method == 'tfidf':
                input_path = os.path.join(input_dir, f"tfidf_vectors_{year}.npz")
                return sparse.load_npz(input_path)
            else:  # word2vec
                input_path = os.path.join(input_dir, f"word2vec_vectors_{year}.npy")
                return np.load(input_path)
        except Exception as e:
            logger.error(f"加载向量化结果时出错: {str(e)}")
            raise
            
    def _save_similarity_matrix(self, matrix: np.ndarray, year: int, method: str) -> None:
        """保存相似度矩阵
        
        Args:
            matrix: 相似度矩阵
            year: 年份
            method: 向量化方法
        """
        try:
            output_dir = os.path.join("data", "similarity")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"similarity_{method}_{year}.npy")
            np.save(output_path, matrix)
            logger.info(f"相似度矩阵已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"保存相似度矩阵时出错: {str(e)}")
            raise
            
    def calculate_similarity(self, year: int, method: str) -> None:
        """计算指定年份和方法的相似度矩阵
        
        Args:
            year: 年份
            method: 向量化方法
        """
        try:
            # 加载向量
            vectors = self._load_vectors(year, method)
            
            # 计算相似度矩阵
            if isinstance(vectors, sparse.spmatrix):
                similarity_matrix = cosine_similarity(vectors)
            else:
                similarity_matrix = 1 - cdist(vectors, vectors, metric='cosine')
                
            # 保存结果
            self._save_similarity_matrix(similarity_matrix, year, method)
            
        except Exception as e:
            logger.error(f"计算相似度矩阵时出错: {str(e)}")
            raise
            
    def calculate_all(self, years: List[int], output_dir: str) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            methods = ['tfidf', 'word2vec']
            
            # 计算每个年份和方法的相似度矩阵
            for year in years:
                logger.info(f"正在处理 {year} 年的数据...")
                for method in methods:
                    logger.info(f"使用 {method} 方法计算相似度...")
                    self.calculate_similarity(year, method)
                    
            return True
            
        except Exception as e:
            logger.error(f"处理年份数据时出错: {str(e)}")
            return False

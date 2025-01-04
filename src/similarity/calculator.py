#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
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
            
    def _load_titles(self, year: int) -> pd.DataFrame:
        """加载原始标题数据，用于编辑距离计算
        
        Args:
            year: 年份
            
        Returns:
            包含标题的DataFrame
        """
        try:
            file_path = os.path.join("data", "preprocessed", f"cleaned_titles_{year}.csv")
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"加载标题数据时出错: {str(e)}")
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

    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """计算两个字符串之间的编辑距离
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            
        Returns:
            编辑距离
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # 动态规划计算编辑距离
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )
        
        return dp[m][n]

    def _calculate_edit_distance_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串之间的编辑距离相似度
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            
        Returns:
            相似度值 [0,1]，1表示完全相同，0表示完全不同
        """
        edit_distance = self._calculate_edit_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        return 1 - (edit_distance / max_len)

    def calculate_similarity(self, year: int, method: str) -> None:
        """计算指定年份和方法的相似度矩阵
        
        Args:
            year: 年份
            method: 相似度计算方法 ('tfidf', 'word2vec' 或 'edit_distance')
        """
        try:
            if method == 'edit_distance':
                # 加载原始标题数据
                df = self._load_titles(year)
                titles = df['title'].tolist()
                n = len(titles)
                
                # 计算编辑距离相似度矩阵
                similarity_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i, n):
                        sim = self._calculate_edit_distance_similarity(titles[i], titles[j])
                        similarity_matrix[i][j] = sim
                        similarity_matrix[j][i] = sim
                        
            else:
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
            methods = ['tfidf', 'word2vec', 'edit_distance']
            
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

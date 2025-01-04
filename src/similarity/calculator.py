"""相似度计算模块"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from Levenshtein import distance as levenshtein_distance
from src.config import SIMILARITY_CONFIG, PROCESSED_DATA_DIR, RESULTS_DIR

class SimilarityError(Exception):
    """相似度计算错误的基类"""
    pass

class SimilarityCalculator:
    """相似度计算类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化计算器
        
        Args:
            config: 相似度计算配置，如果为None则使用默认配置
        """
        self.config = config or SIMILARITY_CONFIG
        self.logger = logging.getLogger(__name__)
        
    def _load_vectors(self, year: int, method: str) -> np.ndarray:
        """加载向量数据
        
        Args:
            year: 年份
            method: 向量化方法，可选 'tfidf' 或 'word2vec'
            
        Returns:
            向量矩阵
            
        Raises:
            SimilarityError: 如果文件不存在或格式错误
        """
        try:
            if method == 'tfidf':
                file_path = os.path.join(PROCESSED_DATA_DIR, f"tfidf_vectors_{year}.npz")
                vectors = sparse.load_npz(file_path)
                return vectors
            elif method == 'word2vec':
                file_path = os.path.join(PROCESSED_DATA_DIR, f"word2vec_vectors_{year}.npy")
                vectors = np.load(file_path)
                return vectors
            else:
                raise SimilarityError(f"不支持的向量化方法: {method}")
        except Exception as e:
            raise SimilarityError(f"加载向量文件时出错: {str(e)}")
            
    def _load_titles(self, year: int) -> pd.DataFrame:
        """加载标题数据
        
        Args:
            year: 年份
            
        Returns:
            包含标题的DataFrame
            
        Raises:
            SimilarityError: 如果文件不存在或格式错误
        """
        try:
            file_path = os.path.join(PROCESSED_DATA_DIR, f"cleaned_titles_{year}.csv")
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['id', 'title']):
                raise SimilarityError(f"标题文件缺少必需的列: {file_path}")
            return df
        except Exception as e:
            raise SimilarityError(f"加载标题文件时出错: {str(e)}")
            
    def calculate_cosine_similarity(
        self,
        vectors1: np.ndarray,
        vectors2: np.ndarray,
        threshold: float
    ) -> sparse.csr_matrix:
        """计算余弦相似度
        
        Args:
            vectors1: 第一组向量
            vectors2: 第二组向量
            threshold: 相似度阈值
            
        Returns:
            稀疏相似度矩阵
        """
        # 如果是稀疏矩阵，转换为密集矩阵
        if sparse.issparse(vectors1):
            vectors1 = vectors1.toarray()
        if sparse.issparse(vectors2):
            vectors2 = vectors2.toarray()
            
        # 计算余弦相似度
        norms1 = np.linalg.norm(vectors1, axis=1)
        norms2 = np.linalg.norm(vectors2, axis=1)
        
        # 处理零范数
        norms1[norms1 == 0] = 1
        norms2[norms2 == 0] = 1
        
        # 归一化向量
        vectors1_normalized = vectors1 / norms1[:, np.newaxis]
        vectors2_normalized = vectors2 / norms2[:, np.newaxis]
        
        # 计算相似度矩阵
        similarity = np.dot(vectors1_normalized, vectors2_normalized.T)
        
        # 对于Word2Vec向量，进行额外的归一化
        if vectors1.shape[1] == 300:  # Word2Vec向量维度通常为300
            similarity = (similarity + 1) / 2  # 将[-1,1]映射到[0,1]
        
        # 应用阈值，转换为稀疏矩阵
        similarity[similarity < threshold] = 0
        return sparse.csr_matrix(similarity)
        
    def calculate_edit_distance_similarity(
        self,
        titles1: List[str],
        titles2: List[str],
        threshold: float
    ) -> sparse.csr_matrix:
        """计算编辑距离相似度
        
        Args:
            titles1: 第一组标题
            titles2: 第二组标题
            threshold: 相似度阈值
            
        Returns:
            稀疏相似度矩阵
        """
        n1, n2 = len(titles1), len(titles2)
        rows, cols, data = [], [], []
        
        for i, title1 in enumerate(titles1):
            for j, title2 in enumerate(titles2):
                # 计算编辑距离
                edit_dist = levenshtein_distance(title1, title2)
                # 使用更合理的归一化方式
                total_len = len(title1) + len(title2)
                if total_len == 0:
                    similarity = 1.0
                else:
                    # 使用两倍编辑距离除以总长度，确保相似度在[0,1]区间
                    similarity = 1.0 - (2.0 * edit_dist) / total_len
                    
                # 应用阈值
                if similarity >= threshold:
                    rows.append(i)
                    cols.append(j)
                    data.append(similarity)
                    
        return sparse.csr_matrix((data, (rows, cols)), shape=(n1, n2))
        
    def calculate_similarity(
        self,
        year1: int,
        year2: int,
        method: str = 'tfidf'
    ) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
        """计算两个年份数据之间的相似度
        
        Args:
            year1: 第一个年份
            year2: 第二个年份
            method: 相似度计算方法，可选 'tfidf', 'word2vec' 或 'edit_distance'
            
        Returns:
            相似度矩阵和元数据
        """
        try:
            if method in ['tfidf', 'word2vec']:
                # 加载向量数据
                vectors1 = self._load_vectors(year1, method)
                vectors2 = self._load_vectors(year2, method)
                
                # 计算余弦相似度
                threshold = self.config['thresholds']['cosine']
                similarity_matrix = self.calculate_cosine_similarity(
                    vectors1, vectors2, threshold
                )
                
            elif method == 'edit_distance':
                # 加载标题数据
                df1 = self._load_titles(year1)
                df2 = self._load_titles(year2)
                
                # 计算编辑距离相似度
                threshold = self.config['thresholds']['edit_distance']
                similarity_matrix = self.calculate_edit_distance_similarity(
                    df1['title'].tolist(),
                    df2['title'].tolist(),
                    threshold
                )
                
            else:
                raise SimilarityError(f"不支持的相似度计算方法: {method}")
                
            # 计算统计信息
            stats = {
                'shape': list(similarity_matrix.shape),
                'sparsity': similarity_matrix.nnz / (similarity_matrix.shape[0] * similarity_matrix.shape[1]),
                'mean': float(similarity_matrix.data.mean()) if similarity_matrix.nnz > 0 else 0.0,
                'std': float(similarity_matrix.data.std()) if similarity_matrix.nnz > 0 else 0.0
            }
            
            return similarity_matrix, stats
            
        except Exception as e:
            raise SimilarityError(f"计算相似度时出错: {str(e)}")
            
    def save_results(
        self,
        similarity_matrix: sparse.csr_matrix,
        year1: int,
        year2: int,
        method: str,
        stats: Dict[str, Any]
    ) -> None:
        """保存计算结果
        
        Args:
            similarity_matrix: 相似度矩阵
            year1: 第一个年份
            year2: 第二个年份
            method: 计算方法
            stats: 统计信息
        """
        try:
            # 创建输出目录
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            # 保存相似度矩阵
            if method in ['tfidf', 'word2vec']:
                matrix_file = os.path.join(
                    RESULTS_DIR,
                    f"cosine_similarity_{method}_{year1}_{year2}.npz"
                )
            else:
                matrix_file = os.path.join(
                    RESULTS_DIR,
                    f"edit_distance_similarity_{year1}_{year2}.npz"
                )
            sparse.save_npz(matrix_file, similarity_matrix)
            
            # 保存或更新元数据
            metadata_file = os.path.join(
                RESULTS_DIR,
                f"similarity_metadata_{year1}_{year2}.json"
            )
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                
            if method in ['tfidf', 'word2vec']:
                metadata[f"cosine_similarity_{method}"] = stats
            else:
                metadata["edit_distance_similarity"] = stats
                
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            raise SimilarityError(f"保存结果时出错: {str(e)}")
            
    def process_years(self, years: List[int], methods: Optional[List[str]] = None) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            methods: 计算方法列表，如果为None则使用配置中的所有方法
            
        Returns:
            处理是否成功
        """
        try:
            if methods is None:
                methods = self.config['methods']
                
            for year1 in years:
                for year2 in years:
                    if year2 < year1:
                        continue
                        
                    self.logger.info(f"正在计算 {year1} 和 {year2} 年的相似度...")
                    
                    for method in methods:
                        self.logger.info(f"使用方法: {method}")
                        similarity_matrix, stats = self.calculate_similarity(
                            year1, year2, method
                        )
                        self.save_results(
                            similarity_matrix,
                            year1,
                            year2,
                            method,
                            stats
                        )
                        
            return True
            
        except Exception as e:
            self.logger.error(f"处理年份数据时出错: {str(e)}")
            raise

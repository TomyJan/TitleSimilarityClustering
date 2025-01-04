"""聚类分析模块"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import matplotlib.pyplot as plt
from src.config import CLUSTERING_CONFIG, PROCESSED_DATA_DIR, RESULTS_DIR
from sklearn.cluster import DBSCAN

class ClusteringError(Exception):
    """聚类分析错误的基类"""
    pass

class TitleClusterer:
    """聚类分析类"""
    
    def __init__(self):
        """初始化聚类器"""
        self.logger = logging.getLogger(__name__)
        # 修改聚类参数
        self.n_clusters = 5  # 聚类数量
        self.eps = 0.3  # DBSCAN的邻域半径
        self.min_samples = 3  # DBSCAN的最小样本数
        
    def _load_vectors(self, year: int, method: str) -> np.ndarray:
        """加载向量数据
        
        Args:
            year: 年份
            method: 向量化方法，可选 'tfidf' 或 'word2vec'
            
        Returns:
            向量矩阵
            
        Raises:
            ClusteringError: 如果文件不存在或格式错误
        """
        try:
            if method == 'tfidf':
                file_path = os.path.join("data", "vectorized", f"tfidf_vectors_{year}.npz")
                vectors = sparse.load_npz(file_path)
                return vectors.toarray()
            elif method == 'word2vec':
                file_path = os.path.join("data", "vectorized", f"word2vec_vectors_{year}.npy")
                vectors = np.load(file_path)
                return vectors
            else:
                raise ClusteringError(f"不支持的向量化方法: {method}")
        except Exception as e:
            raise ClusteringError(f"加载向量文件时出错: {str(e)}")
            
    def _load_titles(self, year: int) -> pd.DataFrame:
        """加载标题数据
        
        Args:
            year: 年份
            
        Returns:
            包含标题的DataFrame
            
        Raises:
            ClusteringError: 如果文件不存在或格式错误
        """
        try:
            file_path = os.path.join("data", "preprocessed", f"cleaned_titles_{year}.csv")
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['id', 'title']):
                raise ClusteringError(f"标题文件缺少必需的列: {file_path}")
            return df
        except Exception as e:
            raise ClusteringError(f"加载标题文件时出错: {str(e)}")
            
    def calculate_metrics(
        self,
        vectors: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """计算聚类评估指标
        
        Args:
            vectors: 向量矩阵
            labels: 聚类标签
            
        Returns:
            评估指标字典
        """
        metrics = {}
        try:
            metrics['silhouette_score'] = float(silhouette_score(vectors, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(vectors, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(vectors, labels))
        except Exception as e:
            self.logger.warning(f"计算评估指标时出错: {str(e)}")
            metrics = {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': -1
            }
        return metrics
        
    def visualize_clusters(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        output_file: str
    ) -> None:
        """可视化聚类结果
        
        Args:
            vectors: 向量矩阵
            labels: 聚类标签
            output_file: 输出文件路径
        """
        try:
            # 使用PCA降维到2维进行可视化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            
            # 创建图形
            plt.figure(
                figsize=(
                    self.config['output_format']['visualization']['resolution'][0] / 100,
                    self.config['output_format']['visualization']['resolution'][1] / 100
                )
            )
            
            # 绘制散点图
            scatter = plt.scatter(
                vectors_2d[:, 0],
                vectors_2d[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6
            )
            
            # 添加图例
            plt.colorbar(scatter)
            
            # 添加标题和轴标签
            plt.title('Cluster Visualization (PCA)')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            
            # 保存图形
            plt.savefig(
                output_file,
                dpi=self.config['output_format']['visualization']['dpi'],
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"可视化聚类结果时出错: {str(e)}")
            
    def cluster(
        self,
        year: int,
        method: str = 'tfidf'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """对指定年份的数据进行聚类
        
        Args:
            year: 年份
            method: 向量化方法，可选 'tfidf' 或 'word2vec'
            
        Returns:
            聚类标签、聚类中心和评估指标
        """
        try:
            # 加载向量数据
            vectors = self._load_vectors(year, method)
            
            # 创建并训练聚类器
            kmeans = KMeans(
                n_clusters=self.config['kmeans']['n_clusters'],
                random_state=self.config['kmeans']['random_state']
            )
            labels = kmeans.fit_predict(vectors)
            
            # 计算评估指标
            metrics = self.calculate_metrics(vectors, labels)
            
            return labels, kmeans.cluster_centers_, metrics
            
        except Exception as e:
            raise ClusteringError(f"聚类分析时出错: {str(e)}")
            
    def save_results(
        self,
        year: int,
        method: str,
        labels: np.ndarray,
        centers: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """保存聚类结果
        
        Args:
            year: 年份
            method: 向量化方法
            labels: 聚类标签
            centers: 聚类中心
            metrics: 评估指标
        """
        try:
            # 创建输出目录
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            # 加载标题数据
            df = self._load_titles(year)
            
            # 添加聚类结果
            df['cluster_id'] = labels
            
            # 计算到聚类中心的距离
            vectors = self._load_vectors(year, method)
            distances = []
            for i, vector in enumerate(vectors):
                cluster_id = labels[i]
                center = centers[cluster_id]
                distance = np.linalg.norm(vector - center)
                distances.append(distance)
            df['distance_to_center'] = distances
            
            # 保存聚类结果
            output_file = os.path.join(
                RESULTS_DIR,
                self.config['output_format']['clusters']['file_pattern'].format(
                    method=method,
                    year=year
                )
            )
            df.to_csv(output_file, index=False)
            
            # 保存聚类中心
            centers_file = os.path.join(
                RESULTS_DIR,
                self.config['output_format']['centers']['file_pattern'].format(
                    method=method,
                    year=year
                )
            )
            np.save(centers_file, centers)
            
            # 保存评估指标
            metrics_file = os.path.join(
                RESULTS_DIR,
                self.config['output_format']['metrics']['file_pattern'].format(
                    method=method,
                    year=year
                )
            )
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
                
            # 生成可视化结果
            vis_file = os.path.join(
                RESULTS_DIR,
                self.config['output_format']['visualization']['file_pattern'].format(
                    method=method,
                    year=year
                )
            )
            self.visualize_clusters(vectors, labels, vis_file)
            
        except Exception as e:
            raise ClusteringError(f"保存结果时出错: {str(e)}")
            
    def process_years(
        self,
        years: List[int],
        methods: Optional[List[str]] = None
    ) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            methods: 向量化方法列表，如果为None则使用配置中的所有方法
            
        Returns:
            处理是否成功
        """
        try:
            if methods is None:
                methods = ['tfidf', 'word2vec']
                
            for year in years:
                self.logger.info(f"正在处理 {year} 年的数据...")
                
                for method in methods:
                    self.logger.info(f"使用方法: {method}")
                    labels, centers, metrics = self.cluster(year, method)
                    self.save_results(year, method, labels, centers, metrics)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"处理年份数据时出错: {str(e)}")
            raise 
            
    def cluster_all(self, years: List[int], output_dir: str) -> bool:
        """对所有年份的数据进行聚类
        
        Args:
            years: 要处理的年份列表
            output_dir: 输出目录
            
        Returns:
            bool: 是否成功
        """
        try:
            # 创建聚类结果目录
            clustering_dir = os.path.join('data', 'clustering')
            os.makedirs(clustering_dir, exist_ok=True)
            
            for year in years:
                # 加载TF-IDF向量
                tfidf_vectors = sparse.load_npz(os.path.join('data', 'vectorized', f'tfidf_vectors_{year}.npz'))
                # 加载Word2Vec向量
                word2vec_vectors = np.load(os.path.join('data', 'vectorized', f'word2vec_vectors_{year}.npy'))
                
                # 对TF-IDF向量进行聚类
                tfidf_labels = self._cluster_vectors(tfidf_vectors.toarray())
                np.save(os.path.join(clustering_dir, f'cluster_labels_tfidf_{year}.npy'), tfidf_labels)
                
                # 对Word2Vec向量进行聚类
                word2vec_labels = self._cluster_vectors(word2vec_vectors)
                np.save(os.path.join(clustering_dir, f'cluster_labels_word2vec_{year}.npy'), word2vec_labels)
                
                self.logger.info(f"{year}年聚类完成")
            
            return True
        except Exception as e:
            self.logger.error(f"聚类过程出错: {str(e)}")
            return False 
            
    def _cluster_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """对向量进行聚类
        
        Args:
            vectors: 要聚类的向量矩阵
            
        Returns:
            np.ndarray: 聚类标签
        """
        try:
            # 使用DBSCAN进行聚类
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(vectors)
            
            # 如果所有样本都被判定为噪声点，则使用KMeans
            if len(np.unique(labels)) <= 1:
                # 根据数据量动态调整聚类数
                n_samples = len(vectors)
                self.n_clusters = min(self.n_clusters, n_samples // 5)  # 确保每个簇至少有5个样本
                self.n_clusters = max(2, self.n_clusters)  # 确保至少有2个簇
                
                clustering = KMeans(n_clusters=self.n_clusters, random_state=42)
                labels = clustering.fit_predict(vectors)
            
            # 确保标签数量与向量数量相同
            if len(labels) != len(vectors):
                self.logger.error(f"标签数量({len(labels)})与向量数量({len(vectors)})不匹配")
                return np.zeros(len(vectors))
            
            return labels
        except Exception as e:
            self.logger.error(f"聚类过程出错: {str(e)}")
            return np.zeros(len(vectors)) 
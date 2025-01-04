import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Optional
import json
import logging
from scipy import sparse
from src.config import VISUALIZATION_CONFIG, PROCESSED_DATA_DIR, RESULTS_DIR
import matplotlib as mpl

# 配置字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

class VisualizationError(Exception):
    """可视化过程中的异常类"""
    pass

class Visualizer:
    """论文标题相似度和聚类结果的可视化类"""
    
    def __init__(self, results_dir: str = RESULTS_DIR, 
                 processed_dir: str = PROCESSED_DATA_DIR,
                 output_dir: str = 'results/visualizations',
                 config: Dict = VISUALIZATION_CONFIG):
        """
        初始化可视化器
        
        Args:
            results_dir: 结果文件目录
            processed_dir: 处理后数据目录
            output_dir: 可视化输出目录
            config: 可视化配置
        """
        self.results_dir = results_dir
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.config = config
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置日志
        self.logger = logging.getLogger(__name__)
        
    def _load_similarity_matrix(self, year1: int, year2: int, method: str) -> np.ndarray:
        """
        加载相似度矩阵
        
        Args:
            year1: 第一年
            year2: 第二年
            method: 相似度计算方法 ('tfidf', 'word2vec', 'edit_distance')
            
        Returns:
            相似度矩阵
        """
        try:
            if method in ['tfidf', 'word2vec']:
                filename = f'cosine_similarity_{method}_{year1}_{year2}.npz'
            else:
                filename = f'edit_distance_similarity_{year1}_{year2}.npz'
                
            filepath = os.path.join(self.results_dir, filename)
            return sparse.load_npz(filepath).toarray()
        except Exception as e:
            raise VisualizationError(f'加载相似度矩阵失败: {str(e)}')
            
    def _load_titles(self, year: int) -> List[str]:
        """
        加载指定年份的标题数据
        
        Args:
            year: 年份
            
        Returns:
            标题列表
        """
        try:
            filename = f'cleaned_titles_{year}.csv'
            filepath = os.path.join(self.processed_dir, filename)
            df = pd.read_csv(filepath)
            return df['title'].tolist()
        except Exception as e:
            raise VisualizationError(f'加载标题数据失败: {str(e)}')
            
    def plot_similarity_heatmap(self, year1: int, year2: int, method: str,
                              output_file: Optional[str] = None) -> None:
        """
        绘制相似度热力图
        
        Args:
            year1: 第一年
            year2: 第二年
            method: 相似度计算方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            similarity_matrix = self._load_similarity_matrix(year1, year2, method)
            titles1 = self._load_titles(year1)
            titles2 = self._load_titles(year2)
            
            # 获取配置
            config = self.config['heatmap']
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            sns.heatmap(similarity_matrix, cmap=config['cmap'], xticklabels=False, yticklabels=False)
            plt.title(f'标题相似度热力图 ({method}, {year1}-{year2})')
            plt.xlabel(f'{year2}年标题')
            plt.ylabel(f'{year1}年标题')
            
            # 保存图像
            if output_file is None:
                output_file = f'similarity_heatmap_{method}_{year1}_{year2}.png'
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f'已生成热力图: {output_path}')
        except Exception as e:
            raise VisualizationError(f'生成热力图失败: {str(e)}')
            
    def plot_similarity_distribution(self, year1: int, year2: int, method: str,
                                   output_file: Optional[str] = None) -> None:
        """
        绘制相似度分布图
        
        Args:
            year1: 第一年
            year2: 第二年
            method: 相似度计算方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            similarity_matrix = self._load_similarity_matrix(year1, year2, method)
            
            # 获取配置
            config = self.config['distribution']
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            sns.histplot(similarity_matrix.flatten(), bins=config['bins'], kde=True)
            plt.title(f'标题相似度分布 ({method}, {year1}-{year2})')
            plt.xlabel('相似度值')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            if output_file is None:
                output_file = f'similarity_distribution_{method}_{year1}_{year2}.png'
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f'已生成分布图: {output_path}')
        except Exception as e:
            raise VisualizationError(f'生成分布图失败: {str(e)}')
            
    def plot_cluster_scatter(self, year: int, method: str,
                           output_file: Optional[str] = None) -> None:
        """
        绘制聚类散点图
        
        Args:
            year: 年份
            method: 向量化方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            if method == 'tfidf':
                vectors = sparse.load_npz(os.path.join(self.processed_dir, f'tfidf_vectors_{year}.npz')).toarray()
            else:
                vectors = np.load(os.path.join(self.processed_dir, f'word2vec_vectors_{year}.npy'))
                
            clusters_df = pd.read_csv(os.path.join(self.results_dir, f'clusters_{method}_{year}.csv'))
            labels = clusters_df['cluster_id'].values
            
            # 获取配置
            config = self.config['scatter']
            
            # 降维
            tsne = TSNE(n_components=2, random_state=config['random_state'])
            vectors_2d = tsne.fit_transform(vectors)
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='tab20')
            plt.colorbar(scatter)
            plt.title(f'聚类结果散点图 ({method}, {year})')
            plt.xlabel('t-SNE维度1')
            plt.ylabel('t-SNE维度2')
            
            # 保存图像
            if output_file is None:
                output_file = f'cluster_scatter_{method}_{year}.png'
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f'已生成散点图: {output_path}')
        except Exception as e:
            raise VisualizationError(f'生成散点图失败: {str(e)}')
            
    def plot_cluster_sizes(self, year: int, method: str,
                          output_file: Optional[str] = None) -> None:
        """
        绘制聚类大小分布图
        
        Args:
            year: 年份
            method: 向量化方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            clusters_df = pd.read_csv(os.path.join(self.results_dir, f'clusters_{method}_{year}.csv'))
            cluster_sizes = clusters_df['cluster_id'].value_counts().sort_index()
            
            # 获取配置
            config = self.config['sizes']
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            cluster_sizes.plot(kind='bar')
            plt.title(f'聚类大小分布 ({method}, {year})')
            plt.xlabel('簇编号')
            plt.ylabel('标题数量')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(cluster_sizes):
                plt.text(i, v, str(v), ha='center', va='bottom')
            
            # 保存图像
            if output_file is None:
                output_file = f'cluster_sizes_{method}_{year}.png'
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f'已生成大小分布图: {output_path}')
        except Exception as e:
            raise VisualizationError(f'生成大小分布图失败: {str(e)}')
            
    def process_visualizations(self, years: List[int]) -> bool:
        """
        批量处理所有可视化任务
        
        Args:
            years: 要处理的年份列表
            
        Returns:
            是否全部成功
        """
        success = True
        methods = ['tfidf', 'word2vec', 'edit_distance']
        
        try:
            # 生成相似度相关的可视化
            for i in range(len(years)-1):
                year1, year2 = years[i], years[i+1]
                for method in methods:
                    self.plot_similarity_heatmap(year1, year2, method)
                    self.plot_similarity_distribution(year1, year2, method)
            
            # 生成聚类相关的可视化
            for year in years[:-1]:  # 最后一年没有聚类结果
                for method in ['tfidf', 'word2vec']:  # 编辑距离方法没有聚类结果
                    try:
                        self.plot_cluster_scatter(year, method)
                        self.plot_cluster_sizes(year, method)
                    except Exception as e:
                        self.logger.error(f'生成{year}年{method}方法的聚类可视化失败: {str(e)}')
                        success = False
                    
        except Exception as e:
            self.logger.error(f'批量处理可视化任务失败: {str(e)}')
            success = False
            
        return success

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
from sklearn.decomposition import PCA

# 配置字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

class VisualizationError(Exception):
    """可视化过程中的异常类"""
    pass

class TitleVisualizer:
    """论文标题相似度和聚类结果的可视化类"""
    
    def __init__(self, results_dir: str = RESULTS_DIR, 
                 processed_dir: str = PROCESSED_DATA_DIR,
                 output_dir: str = 'results/visualization',
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
        
    def _load_similarity_matrix(self, year: int, method: str) -> np.ndarray:
        """
        加载相似度矩阵
        
        Args:
            year: 年份
            method: 相似度计算方法 ('tfidf', 'word2vec')
            
        Returns:
            相似度矩阵
        """
        try:
            filename = f'similarity_{method}_{year}.npy'
            filepath = os.path.join('data', 'similarity', filename)
            return np.load(filepath)
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
            
    def _load_vectors(self, year: int, method: str) -> np.ndarray:
        """加载向量数据
        
        Args:
            year: 年份
            method: 向量化方法，可选 'tfidf' 或 'word2vec'
            
        Returns:
            向量矩阵
            
        Raises:
            VisualizationError: 如果文件不存在或格式错误
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
                raise VisualizationError(f"不支持的向量化方法: {method}")
        except Exception as e:
            raise VisualizationError(f"加载向量文件时出错: {str(e)}")
            
    def plot_similarity_heatmap(self, year: int, method: str,
                              output_file: Optional[str] = None) -> None:
        """
        绘制相似度热力图
        
        Args:
            year: 年份
            method: 相似度计算方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            similarity_matrix = self._load_similarity_matrix(year, method)
            titles = self._load_titles(year)
            
            # 获取配置
            config = self.config['heatmap']
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            sns.heatmap(similarity_matrix, cmap=config['cmap'], xticklabels=False, yticklabels=False)
            plt.title(f'标题相似度热力图 ({method}, {year})')
            plt.xlabel('标题')
            plt.ylabel('标题')
            
            # 保存图像
            if output_file is None:
                output_file = f'similarity_heatmap_{method}_{year}.png'
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f'已生成热力图: {output_path}')
        except Exception as e:
            raise VisualizationError(f'生成热力图失败: {str(e)}')
            
    def plot_similarity_distribution(self, year: int, method: str,
                                   output_file: Optional[str] = None) -> None:
        """
        绘制相似度分布图
        
        Args:
            year: 年份
            method: 相似度计算方法
            output_file: 输出文件名
        """
        try:
            # 加载数据
            similarity_matrix = self._load_similarity_matrix(year, method)
            
            # 获取配置
            config = self.config['distribution']
            
            # 创建图像
            plt.figure(figsize=config['figsize'])
            sns.histplot(similarity_matrix.flatten(), bins=config['bins'], kde=True)
            plt.title(f'标题相似度分布 ({method}, {year})')
            plt.xlabel('相似度值')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            if output_file is None:
                output_file = f'similarity_distribution_{method}_{year}.png'
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
                vectors = sparse.load_npz(os.path.join("data", "vectorized", f'tfidf_vectors_{year}.npz')).toarray()
            else:
                vectors = np.load(os.path.join("data", "vectorized", f'word2vec_vectors_{year}.npy'))
                
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
        methods = ['tfidf', 'word2vec']
        
        try:
            # 生成相似度相关的可视化
            for year in years:
                for method in methods:
                    try:
                        self.plot_similarity_heatmap(year, method)
                        self.plot_similarity_distribution(year, method)
                    except Exception as e:
                        self.logger.error(f'生成{year}年{method}方法的相似度可视化失败: {str(e)}')
                        success = False
            
            # 生成聚类相关的可视化
            for year in years:
                for method in methods:
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
            
    def plot_similarity_heatmaps(self, years: List[int], output_dir: str = 'results/visualization') -> bool:
        """生成所有年份的相似度矩阵热图
        
        Args:
            years: 年份列表
            output_dir: 输出目录
            
        Returns:
            是否全部成功
        """
        success = True
        methods = ['tfidf', 'word2vec']
        
        try:
            for year in years:
                for method in methods:
                    try:
                        # 加载相似度矩阵
                        similarity_matrix = np.load(os.path.join('data', 'similarity', f'similarity_{method}_{year}.npy'))
                        
                        # 生成热图
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(similarity_matrix, cmap='YlOrRd')
                        plt.title(f'{year}年论文标题相似度矩阵 ({method})')
                        
                        # 保存图片
                        output_path = os.path.join(output_dir, f'similarity_heatmap_{method}_{year}.png')
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        plt.savefig(output_path)
                        plt.close()
                        
                        self.logger.info(f'已生成热图: {output_path}')
                    except Exception as e:
                        self.logger.error(f'生成{year}年{method}方法的热图失败: {str(e)}')
                        success = False
                        
        except Exception as e:
            self.logger.error(f'生成热图失败: {str(e)}')
            success = False
            
        return success
        
    def plot_clustering_results(self, years: List[int], output_dir: str = 'results/visualization') -> bool:
        """生成所有年份的聚类结果可视化
        
        Args:
            years: 年份列表
            output_dir: 输出目录
            
        Returns:
            是否全部成功
        """
        success = True
        methods = ['tfidf', 'word2vec']
        
        try:
            for year in years:
                for method in methods:
                    try:
                        # 加载向量和聚类结果
                        if method == 'tfidf':
                            vectors = sparse.load_npz(os.path.join('data', 'vectorized', f'tfidf_vectors_{year}.npz')).toarray()
                        else:
                            vectors = np.load(os.path.join('data', 'vectorized', f'word2vec_vectors_{year}.npy'))
                            
                        clusters_df = pd.read_csv(os.path.join('results', f'clusters_{method}_{year}.csv'))
                        labels = clusters_df['cluster_id'].values
                        
                        # 使用PCA降维到2维
                        pca = PCA(n_components=2)
                        vectors_2d = pca.fit_transform(vectors)
                        
                        # 创建散点图
                        plt.figure(figsize=(10, 8))
                        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='viridis')
                        plt.colorbar(scatter)
                        plt.title(f'{year}年论文标题聚类结果 ({method})')
                        plt.xlabel('第一主成分')
                        plt.ylabel('第二主成分')
                        
                        # 保存图片
                        output_path = os.path.join(output_dir, f'cluster_scatter_{method}_{year}.png')
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        plt.savefig(output_path)
                        plt.close()
                        
                        self.logger.info(f'已生成散点图: {output_path}')
                    except Exception as e:
                        self.logger.error(f'生成{year}年{method}方法的散点图失败: {str(e)}')
                        success = False
                        
        except Exception as e:
            self.logger.error(f'生成散点图失败: {str(e)}')
            success = False
            
        return success

    def generate_analysis_report(self, years: List[int], output_dir: str) -> bool:
        """生成可视化分析报告
        
        Args:
            years: 要分析的年份列表
            output_dir: 输出目录
            
        Returns:
            bool: 是否成功生成报告
        """
        try:
            report_path = os.path.join(output_dir, "visualization_analysis.md")
            
            with open(report_path, "w", encoding="utf-8") as f:
                # 写入报告头部
                f.write("# 论文标题相似度与聚类分析报告\n\n")
                f.write(f"分析时间范围：{min(years)}-{max(years)}\n\n")
                
                # 相似度矩阵分析
                f.write("## 1. 相似度矩阵分析\n\n")
                f.write("### 1.1 整体相似度分布\n")
                # 计算并写入相似度统计信息
                similarity_stats = self._analyze_similarity_distribution(years)
                f.write(f"- 平均相似度：{similarity_stats['mean']:.4f}\n")
                f.write(f"- 最大相似度：{similarity_stats['max']:.4f}\n")
                f.write(f"- 最小相似度：{similarity_stats['min']:.4f}\n")
                f.write(f"- 标准差：{similarity_stats['std']:.4f}\n\n")
                
                # 聚类结果分析
                f.write("## 2. 聚类结果分析\n\n")
                f.write("### 2.1 聚类统计\n")
                # 计算并写入聚类统计信息
                cluster_stats = self._analyze_clustering_results(years)
                for year in years:
                    f.write(f"\n#### {year}年聚类结果\n")
                    f.write(f"- 聚类数量：{cluster_stats[year]['n_clusters']}\n")
                    f.write(f"- 最大簇大小：{cluster_stats[year]['max_cluster_size']}\n")
                    f.write(f"- 最小簇大小：{cluster_stats[year]['min_cluster_size']}\n")
                    f.write(f"- 平均簇大小：{cluster_stats[year]['avg_cluster_size']:.2f}\n")
                    
                # 研究趋势分析
                f.write("\n## 3. 研究趋势分析\n\n")
                trends = self._analyze_research_trends(years)
                f.write("### 3.1 热点主题演变\n")
                for year, topics in trends.items():
                    f.write(f"\n#### {year}年热点主题\n")
                    for topic in topics:
                        f.write(f"- {topic}\n")
                
                # 建议与结论
                f.write("\n## 4. 建议与结论\n\n")
                f.write("### 4.1 研究方向建议\n")
                f.write("- 根据聚类结果，建议关注以下新兴研究方向：\n")
                f.write("  1. [待补充具体建议]\n")
                f.write("  2. [待补充具体建议]\n\n")
                
                f.write("### 4.2 总体结论\n")
                f.write("- [待补充总体结论]\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成分析报告时出错: {str(e)}")
            return False

    def _analyze_similarity_distribution(self, years: List[int]) -> Dict:
        """分析相似度分布。"""
        all_similarities = []
        for year in years:
            for method in ['tfidf', 'word2vec']:
                similarity_matrix = np.load(f'data/similarity/similarity_{method}_{year}.npy')
                # 获取上三角矩阵的值（不包括对角线）
                upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                all_similarities.extend(upper_triangle)
        
        all_similarities = np.array(all_similarities)
        stats = {
            'mean': np.mean(all_similarities),
            'max': np.max(all_similarities),
            'min': np.min(all_similarities),
            'std': np.std(all_similarities)
        }
        return stats

    def _analyze_clustering_results(self, years: List[int]) -> Dict:
        """分析聚类结果
        
        Args:
            years: 要分析的年份列表
            
        Returns:
            Dict: 聚类统计信息
        """
        try:
            results = {}
            methods = ['tfidf', 'word2vec']
            
            for year in years:
                year_results = {'n_clusters': 0, 'max_cluster_size': 0, 
                              'min_cluster_size': float('inf'), 'avg_cluster_size': 0.0}
                              
                for method in methods:
                    try:
                        # 读取聚类结果
                        clusters_df = pd.read_csv(os.path.join('results', f'clusters_{method}_{year}.csv'))
                        cluster_sizes = clusters_df['cluster_id'].value_counts()
                        
                        # 更新统计信息
                        year_results['n_clusters'] = max(year_results['n_clusters'], 
                                                       len(cluster_sizes))
                        year_results['max_cluster_size'] = max(year_results['max_cluster_size'], 
                                                             cluster_sizes.max())
                        year_results['min_cluster_size'] = min(year_results['min_cluster_size'], 
                                                             cluster_sizes.min())
                        year_results['avg_cluster_size'] = max(year_results['avg_cluster_size'], 
                                                             float(cluster_sizes.mean()))
                    except Exception as e:
                        self.logger.warning(f"无法分析{year}年{method}方法的聚类结果: {str(e)}")
                        continue
                
                if year_results['min_cluster_size'] == float('inf'):
                    year_results['min_cluster_size'] = 0
                    
                results[year] = year_results
                
            return results
        except Exception as e:
            self.logger.error(f"分析聚类结果时出错: {str(e)}")
            return {year: {
                'n_clusters': 0,
                'max_cluster_size': 0,
                'min_cluster_size': 0,
                'avg_cluster_size': 0.0
            } for year in years}

    def _analyze_research_trends(self, years: List[int]) -> Dict:
        """分析研究趋势
        
        Args:
            years: 要分析的年份列表
            
        Returns:
            Dict: 研究趋势信息
        """
        try:
            trends = {}
            methods = ['tfidf', 'word2vec']
            
            for year in years:
                year_topics = set()
                for method in methods:
                    try:
                        # 读取聚类结果
                        clusters_path = os.path.join('results', f'clusters_{method}_{year}.csv')
                        if not os.path.exists(clusters_path):
                            self.logger.warning(f"找不到聚类结果文件: {clusters_path}")
                            continue
                            
                        clusters_df = pd.read_csv(clusters_path)
                        
                        # 确保数据框有必要的列
                        if 'cluster_id' not in clusters_df.columns or 'cleaned_tokens' not in clusters_df.columns:
                            self.logger.warning(f"{year}年{method}方法的聚类结果缺少必要的列")
                            continue
                        
                        # 获取每个簇的代表性主题
                        for cluster_id in clusters_df['cluster_id'].unique():
                            cluster_titles = clusters_df[clusters_df['cluster_id'] == cluster_id]['cleaned_tokens']
                            if len(cluster_titles) >= 5:  # 只考虑大于等于5个标题的簇
                                # 使用第一个标题作为主题
                                year_topics.add(cluster_titles.iloc[0])
                                
                    except Exception as e:
                        self.logger.warning(f"无法分析{year}年{method}方法的研究趋势: {str(e)}")
                        continue
                
                # 如果没有找到任何主题，使用默认值
                if not year_topics:
                    year_topics = {'暂无足够数据进行分析'}
                    
                trends[year] = list(year_topics)[:10]  # 每年最多保留10个主题
                
            return trends
        except Exception as e:
            self.logger.error(f"分析研究趋势时出错: {str(e)}")
            return {year: ['数据不足'] for year in years}

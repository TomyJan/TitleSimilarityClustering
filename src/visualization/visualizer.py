import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from sklearn.decomposition import PCA
from scipy import sparse
import matplotlib as mpl

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['MiSans', 'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir: str):
        """初始化可视化器。

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _load_vectors(self, year: int, method: str) -> np.ndarray:
        """加载向量数据。

        Args:
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            np.ndarray: 向量矩阵
        """
        try:
            if method == 'tfidf':
                file_path = os.path.join('data', 'vectorized', f'tfidf_vectors_{year}.npz')
                vectors = sparse.load_npz(file_path)
                return vectors.toarray()
            else:  # word2vec
                file_path = os.path.join('data', 'vectorized', f'word2vec_vectors_{year}.npy')
                return np.load(file_path)
        except Exception as e:
            logger.error(f'加载{year}年{method}方法的向量时出错: {e}')
            return np.array([])

    def _load_labels(self, year: int, method: str) -> np.ndarray:
        """加载聚类标签。

        Args:
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            np.ndarray: 聚类标签
        """
        try:
            # 首先尝试从results目录加载
            label_path = os.path.join('results', f'cluster_labels_{method}_{year}.npy')
            if not os.path.exists(label_path):
                # 如果不存在，尝试从data/clustering目录加载
                label_path = os.path.join('data', 'clustering', f'cluster_labels_{method}_{year}.npy')
            return np.load(label_path)
        except Exception as e:
            logger.error(f'加载{year}年{method}方法的聚类标签时出错: {e}')
            return np.array([])

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray, year: int, method: str) -> bool:
        """绘制相似度矩阵热图。

        Args:
            similarity_matrix: 相似度矩阵
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            bool: 是否成功生成热图
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, cmap='YlOrRd')
            plt.title(f'{year}年论文标题相似度矩阵 ({method}方法)')
            
            output_path = os.path.join(self.output_dir, f'similarity_heatmap_{method}_{year}.png')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f'已生成热图: {output_path}')
            return True
        except Exception as e:
            logger.error(f'生成热图时出错: {e}')
            return False

    def plot_cluster_scatter(self, vectors: np.ndarray, labels: np.ndarray, year: int, method: str) -> bool:
        """绘制聚类结果散点图。

        Args:
            vectors: 降维后的向量
            labels: 聚类标签
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            bool: 是否成功生成散点图
        """
        try:
            # 确保向量和标签的维度匹配
            if len(vectors) != len(labels):
                logger.error(f'{year}年{method}方法的向量数量({len(vectors)})与标签数量({len(labels)})不匹配')
                return False

            # 使用PCA降维到2维用于可视化
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(f'{year}年论文标题聚类结果 ({method}方法)')
            
            output_path = os.path.join(self.output_dir, f'cluster_scatter_{method}_{year}.png')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f'已生成散点图: {output_path}')
            return True
        except Exception as e:
            logger.error(f'生成散点图时出错: {e}')
            return False

    def _analyze_similarity_distribution(self, year: int, method: str) -> Dict:
        """分析相似度分布。

        Args:
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            Dict: 包含相似度分布统计信息的字典
        """
        try:
            similarity_matrix = np.load(os.path.join('data', 'similarity', f'similarity_{method}_{year}.npy'))
            # 获取上三角矩阵的值（不包括对角线）
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            return {
                'mean': np.mean(upper_triangle),
                'std': np.std(upper_triangle),
                'max': np.max(upper_triangle),
                'min': np.min(upper_triangle)
            }
        except Exception as e:
            logger.warning(f'无法分析{year}年{method}方法的相似度分布: {e}')
            return {}

    def _analyze_clustering_results(self, year: int, method: str) -> Dict:
        """分析聚类结果。

        Args:
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            Dict: 包含聚类结果统计信息的字典
        """
        try:
            labels = np.load(os.path.join('data', 'clustering', f'cluster_labels_{method}_{year}.npy'))
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            
            return {
                'num_clusters': len(unique_labels),
                'max_cluster_size': np.max(cluster_sizes),
                'min_cluster_size': np.min(cluster_sizes),
                'avg_cluster_size': np.mean(cluster_sizes)
            }
        except Exception as e:
            logger.warning(f'无法分析{year}年{method}方法的聚类结果: {e}')
            return {}

    def _analyze_research_trends(self, year: int, method: str) -> List[str]:
        """分析研究趋势。

        Args:
            year: 年份
            method: 使用的方法 (tfidf 或 word2vec)

        Returns:
            List[str]: 主要研究主题列表
        """
        try:
            # 从preprocessed目录读取清洗后的文件
            df = pd.read_csv(os.path.join('data', 'preprocessed', f'cleaned_titles_{year}.csv'))
            # 修改文件路径以匹配clusterer.py中的保存路径
            labels = np.load(os.path.join('data', 'clustering', f'cluster_labels_{method}_{year}.npy'))
            
            # 获取每个聚类中的标题
            unique_labels = np.unique(labels)
            trends = []
            for label in unique_labels:
                cluster_titles = df.iloc[labels == label]['title'].tolist()
                # 这里可以添加更复杂的主题提取逻辑
                if cluster_titles:
                    trends.append(f"聚类 {label}: {len(cluster_titles)}篇论文")
            
            return trends
        except Exception as e:
            logger.warning(f'无法分析{year}年{method}方法的研究趋势: {e}')
            return []

    def generate_analysis_report(self, years: List[int], output_dir: str) -> bool:
        """生成分析报告。

        Args:
            years: 年份列表
            output_dir: 输出目录

        Returns:
            bool: 是否成功生成报告
        """
        try:
            report_content = []
            report_content.append("# 论文标题相似度分析报告\n")

            methods = ['tfidf', 'word2vec']
            
            for year in years:
                report_content.append(f"\n## {year}年分析结果\n")
                
                for method in methods:
                    report_content.append(f"\n### {method.upper()}方法\n")
                    
                    # 加载向量和标签
                    vectors = self._load_vectors(year, method)
                    labels = self._load_labels(year, method)
                    
                    if len(vectors) > 0 and len(labels) > 0:
                        # 生成相似度热图
                        similarity_matrix = np.load(os.path.join('data', 'similarity', f'similarity_{method}_{year}.npy'))
                        self.plot_similarity_heatmap(similarity_matrix, year, method)
                        
                        # 生成聚类散点图
                        if len(vectors) == len(labels):
                            self.plot_cluster_scatter(vectors, labels, year, method)
                        else:
                            logger.error(f'{year}年{method}方法的向量数量({len(vectors)})与标签数量({len(labels)})不匹配')
                    
                    # 相似度分布分析
                    sim_stats = self._analyze_similarity_distribution(year, method)
                    if sim_stats:
                        report_content.append("\n#### 相似度分布\n")
                        report_content.append(f"- 平均相似度: {sim_stats['mean']:.4f}")
                        report_content.append(f"- 标准差: {sim_stats['std']:.4f}")
                        report_content.append(f"- 最大相似度: {sim_stats['max']:.4f}")
                        report_content.append(f"- 最小相似度: {sim_stats['min']:.4f}\n")
                    
                    # 聚类分析
                    cluster_stats = self._analyze_clustering_results(year, method)
                    if cluster_stats:
                        report_content.append("\n#### 聚类统计\n")
                        report_content.append(f"- 聚类数量: {cluster_stats['num_clusters']}")
                        report_content.append(f"- 最大聚类大小: {cluster_stats['max_cluster_size']}")
                        report_content.append(f"- 最小聚类大小: {cluster_stats['min_cluster_size']}")
                        report_content.append(f"- 平均聚类大小: {cluster_stats['avg_cluster_size']:.2f}\n")
                    
                    # 研究趋势分析
                    trends = self._analyze_research_trends(year, method)
                    if trends:
                        report_content.append("\n#### 研究趋势\n")
                        for trend in trends:
                            report_content.append(f"- {trend}\n")
            
            # 保存报告
            report_path = os.path.join(output_dir, 'analysis_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            logger.info(f'分析报告已保存至: {report_path}')
            return True
            
        except Exception as e:
            logger.error(f'生成分析报告时出错: {str(e)}')
            return False

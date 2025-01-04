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
import time

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

    def _load_similarity_matrix(self, year: int, method: str) -> np.ndarray:
        """加载相似度矩阵。

        Args:
            year: 年份
            method: 使用的方法

        Returns:
            np.ndarray: 相似度矩阵
        """
        try:
            matrix_path = os.path.join('data', 'similarity', f'similarity_{method}_{year}.npy')
            return np.load(matrix_path)
        except Exception as e:
            logger.error(f'加载{year}年{method}方法的相似度矩阵时出错: {e}')
            return np.array([])

    def plot_similarity_comparison(self, year: int) -> bool:
        """绘制不同方法的相似度分布对比图。

        Args:
            year: 年份

        Returns:
            bool: 是否成功生成对比图
        """
        try:
            methods = ['tfidf', 'word2vec', 'edit_distance']
            method_names = {'tfidf': 'TF-IDF', 'word2vec': 'Word2Vec', 'edit_distance': '编辑距离'}
            plt.figure(figsize=(12, 6))

            for method in methods:
                matrix = self._load_similarity_matrix(year, method)
                if matrix.size > 0:
                    # 获取上三角矩阵的值（不包括对角线）
                    similarities = matrix[np.triu_indices_from(matrix, k=1)]
                    sns.kdeplot(similarities, label=method_names[method])

            plt.title(f'{year}年不同相似度算法的分布对比')
            plt.xlabel('相似度值')
            plt.ylabel('密度')
            plt.legend()
            
            output_path = os.path.join(self.output_dir, f'similarity_comparison_{year}.png')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f'已生成相似度对比图: {output_path}')
            return True
        except Exception as e:
            logger.error(f'生成相似度对比图时出错: {e}')
            return False

    def _analyze_algorithm_performance(self, year: int) -> Dict:
        """分析不同算法的性能。

        Args:
            year: 年份

        Returns:
            Dict: 包含性能分析结果的字典
        """
        try:
            methods = ['tfidf', 'word2vec', 'edit_distance']
            performance = {}
            
            for method in methods:
                start_time = time.time()
                matrix = self._load_similarity_matrix(year, method)
                if matrix.size > 0:
                    # 计算性能指标
                    performance[method] = {
                        'time': time.time() - start_time,
                        'memory': matrix.nbytes / (1024 * 1024),  # MB
                        'sparsity': 1.0 - (np.count_nonzero(matrix) / matrix.size)
                    }
            
            return performance
        except Exception as e:
            logger.error(f'分析算法性能时出错: {e}')
            return {}

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

            methods = ['tfidf', 'word2vec', 'edit_distance']
            method_names = {'tfidf': 'TF-IDF', 'word2vec': 'Word2Vec', 'edit_distance': '编辑距离'}
            
            # 添加算法比较部分
            report_content.append("\n## 算法比较分析\n")
            report_content.append("\n### 算法特点比较\n")
            report_content.append("\n#### TF-IDF 算法\n")
            report_content.append("- 优点：")
            report_content.append("  - 考虑词频和逆文档频率，能够反映词语的重要性")
            report_content.append("  - 计算效率高，适合大规模文本处理")
            report_content.append("  - 结果易于解释和理解")
            report_content.append("- 缺点：")
            report_content.append("  - 不考虑词序和语义关系")
            report_content.append("  - 对于短文本效果可能不够理想")
            report_content.append("  - 无法处理同义词和多义词\n")

            report_content.append("\n#### Word2Vec 算法\n")
            report_content.append("- 优点：")
            report_content.append("  - 能够捕捉词语的语义关系")
            report_content.append("  - 可以处理同义词")
            report_content.append("  - 生成的向量具有良好的语义特性")
            report_content.append("- 缺点：")
            report_content.append("  - 需要大量训练数据")
            report_content.append("  - 计算资源消耗较大")
            report_content.append("  - 结果不如TF-IDF直观\n")

            report_content.append("\n#### 编辑距离算法\n")
            report_content.append("- 优点：")
            report_content.append("  - 直接比较字符串差异，结果直观")
            report_content.append("  - 不需要预训练或特征提取")
            report_content.append("  - 适合检测拼写错误和细微差异")
            report_content.append("- 缺点：")
            report_content.append("  - 计算复杂度较高")
            report_content.append("  - 不考虑语义信息")
            report_content.append("  - 对文本长度敏感\n")
            
            for year in years:
                report_content.append(f"\n## {year}年分析结果\n")
                
                # 生成相似度分布对比图
                self.plot_similarity_comparison(year)
                report_content.append(f"\n### 相似度分布对比\n")
                report_content.append(f"![相似度分布对比](similarity_comparison_{year}.png)\n")
                
                # 添加性能分析
                performance = self._analyze_algorithm_performance(year)
                if performance:
                    report_content.append("\n### 算法性能对比\n")
                    report_content.append("| 算法 | 处理时间(秒) | 内存占用(MB) | 稀疏度 |")
                    report_content.append("|------|------------|------------|--------|")
                    for method in methods:
                        if method in performance:
                            perf = performance[method]
                            report_content.append(
                                f"| {method_names[method]} | {perf['time']:.4f} | {perf['memory']:.2f} | {perf['sparsity']:.4f} |"
                            )
                
                for method in methods:
                    report_content.append(f"\n### {method_names[method]}方法\n")
                    
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

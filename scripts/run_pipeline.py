#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.crawler.crawler import CNKICrawler
from src.preprocessing.preprocessor import TitlePreprocessor
from src.vectorization.vectorizer import TitleVectorizer
from src.similarity.calculator import TitleSimilarityCalculator
from src.clustering.clusterer import TitleClusterer
from src.visualization.visualizer import Visualizer

def check_data_exists(year: int) -> bool:
    """检查指定年份的数据是否存在
    
    Args:
        year: 要检查的年份
        
    Returns:
        bool: 数据是否存在
    """
    try:
        # 检查原始数据
        raw_file = os.path.join('data', 'raw', f'thesis_titles_{year}.csv')
        if not os.path.exists(raw_file):
            return False
            
        # 检查预处理数据
        cleaned_file = os.path.join('data', 'preprocessed', f'cleaned_titles_{year}.csv')
        tokenized_file = os.path.join('data', 'preprocessed', f'tokenized_titles_{year}.csv')
        if not (os.path.exists(cleaned_file) and os.path.exists(tokenized_file)):
            return False
            
        # 检查向量化数据
        tfidf_file = os.path.join('data', 'vectorized', f'tfidf_vectors_{year}.npz')
        word2vec_file = os.path.join('data', 'vectorized', f'word2vec_vectors_{year}.npy')
        if not (os.path.exists(tfidf_file) and os.path.exists(word2vec_file)):
            return False
            
        return True
    except Exception as e:
        logger.error(f"检查数据时出错: {e}")
        return False

def main(years: Optional[List[int]] = None):
    """主函数
    
    Args:
        years: 要处理的年份列表，如果为None则使用默认值
    """
    if years is None:
        years = list(range(2020, 2025))
        
    try:
        # 创建输出目录
        output_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查数据并在需要时爬取
        crawler = CNKICrawler()
        missing_years = [year for year in years if not check_data_exists(year)]
        if missing_years:
            logger.info(f"需要爬取以下年份的数据: {missing_years}")
            success = crawler.crawl_all(missing_years)
            if not success:
                logger.error("爬取数据失败")
                sys.exit(1)
        
        # 预处理
        logger.info("开始预处理...")
        preprocessor = TitlePreprocessor()
        success = preprocessor.process_all(years, output_dir)
        if not success:
            logger.error("预处理失败")
            sys.exit(1)
        logger.info("预处理完成")
        
        # 向量化
        logger.info("开始向量化...")
        vectorizer = TitleVectorizer()
        success = vectorizer.vectorize_all(years, output_dir)
        if not success:
            logger.error("向量化失败")
            sys.exit(1)
        logger.info("向量化完成")
        
        # 计算相似度
        logger.info("开始计算相似度...")
        calculator = TitleSimilarityCalculator()
        success = calculator.calculate_all(years, output_dir)
        if not success:
            logger.error("相似度计算失败")
            sys.exit(1)
        logger.info("相似度计算完成")
        
        # 聚类分析
        logger.info("开始聚类分析...")
        clusterer = TitleClusterer()
        success = clusterer.cluster_all(years, output_dir)
        if not success:
            logger.error("聚类分析失败")
            sys.exit(1)
        logger.info("聚类分析完成")
        
        # 可视化
        logger.info("开始生成可视化结果...")
        visualizer = Visualizer(output_dir)
        success = visualizer.generate_analysis_report(years, output_dir)
        if not success:
            logger.error("生成可视化结果失败")
            sys.exit(1)
        logger.info("可视化结果生成完成")
        
        logger.info(f"所有处理完成，结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"处理过程出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
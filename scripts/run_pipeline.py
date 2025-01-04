#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import subprocess
import traceback
from datetime import datetime
from typing import List, Tuple
from src.preprocessing.preprocessor import TitlePreprocessor
from src.vectorization.vectorizer import TitleVectorizer
from src.similarity.calculator import TitleSimilarityCalculator
from src.clustering.clusterer import TitleClusterer
from src.visualization.visualizer import TitleVisualizer
from src.config import YEAR_RANGE
from src.crawler.crawler import CNKICrawler

def setup_logging() -> None:
    """配置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_step(script_path: str, step_name: str) -> bool:
    """执行单个处理步骤
    
    Args:
        script_path: 脚本路径
        step_name: 步骤名称
        
    Returns:
        bool: 是否执行成功
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始执行 {step_name}")
    
    if not os.path.exists(script_path):
        logger.error(f"找不到脚本文件: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        logger.info(f"{step_name} 执行完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} 执行失败")
        logger.error(f"错误码: {e.returncode}")
        logger.error(f"输出: {e.output}")
        if e.stderr:
            logger.error(f"错误信息: {e.stderr}")
        return False
        
    except Exception as e:
        logger.error(f"{step_name} 执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_data_exists(year: int) -> bool:
    """检查指定年份的数据是否已存在
    
    Args:
        year: 要检查的年份
        
    Returns:
        bool: 数据是否存在
    """
    data_file = os.path.join("data", "raw", f"thesis_titles_{year}.csv")
    return os.path.exists(data_file)

def main() -> None:
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始执行处理流水线")
    
    try:
        # 获取年份范围
        years = list(range(YEAR_RANGE['start_year'], YEAR_RANGE['end_year'] + 1))
        
        # 检查数据并执行爬虫
        missing_years = [year for year in years if not check_data_exists(year)]
        if missing_years:
            logger.info(f"发现缺失数据的年份: {missing_years}")
            logger.info("开始执行 爬虫")
            crawler = CNKICrawler()
            success = crawler.crawl_all(missing_years)
            if not success:
                logger.error("爬虫 执行失败")
                sys.exit(1)
            logger.info("爬虫 执行完成")
        else:
            logger.info("所有年份的数据已存在，跳过爬虫步骤")
        
        # 确保预处理输出目录存在
        output_dir = os.path.join("data", "preprocessed")
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行预处理
        logger.info("开始执行 预处理")
        preprocessor = TitlePreprocessor()
        success = preprocessor.process_all(years, output_dir)
        if not success:
            logger.error("预处理 执行失败")
            sys.exit(1)
        logger.info("预处理 执行完成")
        
        # 执行向量化
        logger.info("开始执行 向量化")
        vectorizer = TitleVectorizer()
        output_dir = os.path.join("data", "vectorized")
        os.makedirs(output_dir, exist_ok=True)
        success = vectorizer.vectorize_all(years, output_dir)
        if not success:
            logger.error("向量化 执行失败")
            sys.exit(1)
        logger.info("向量化 执行完成")
        
        # 执行相似度计算
        logger.info("开始执行 相似度计算")
        calculator = TitleSimilarityCalculator()
        output_dir = os.path.join("data", "similarity")
        os.makedirs(output_dir, exist_ok=True)
        success = calculator.calculate_all(years, output_dir)
        if not success:
            logger.error("相似度计算 执行失败")
            sys.exit(1)
        logger.info("相似度计算 执行完成")
        
        # 执行聚类
        logger.info("开始执行 聚类")
        clusterer = TitleClusterer()
        output_dir = os.path.join("results")
        os.makedirs(output_dir, exist_ok=True)
        success = clusterer.cluster_all(years, output_dir)
        if not success:
            logger.error("聚类 执行失败")
            sys.exit(1)
        logger.info("聚类 执行完成")
        
        # 执行可视化
        logger.info("开始执行 可视化")
        visualizer = TitleVisualizer()
        output_dir = os.path.join("results", "visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成相似度矩阵热图
        logger.info("生成相似度矩阵热图")
        success = visualizer.plot_similarity_heatmaps(years, output_dir)
        if not success:
            logger.error("生成相似度矩阵热图失败")
            sys.exit(1)
        
        # 生成聚类结果可视化
        logger.info("生成聚类结果可视化")
        success = visualizer.plot_clustering_results(years, output_dir)
        if not success:
            logger.error("生成聚类结果可视化失败")
            sys.exit(1)
        
        logger.info("可视化 执行完成")
        logger.info("流水线执行完成")
        
    except Exception as e:
        logger.error(f"流水线执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
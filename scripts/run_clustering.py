#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import traceback
from datetime import datetime
from src.clustering.clusterer import TitleClusterer
from src.config import CLUSTERING_CONFIG

def setup_logging() -> None:
    """配置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"clustering_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main() -> None:
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始聚类分析")
    
    try:
        # 初始化聚类器
        clusterer = TitleClusterer()
        
        # 获取要处理的年份
        years = list(range(2020, 2024))  # 2020-2023
        logger.info(f"将处理以下年份: {years}")
        
        # 确保输出目录存在
        output_dir = os.path.join(project_root, "data", "clusters")
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行聚类
        success = clusterer.cluster_all(years, output_dir)
        if not success:
            logger.error("聚类过程中出现错误")
            sys.exit(1)
            
        logger.info("聚类分析完成")
        
    except Exception as e:
        logger.error(f"聚类分析出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
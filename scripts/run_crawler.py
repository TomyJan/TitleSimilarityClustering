#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
from datetime import datetime
from src.crawler.crawler import CNKICrawler
from src.config import YEAR_RANGE

def setup_logging() -> None:
    """配置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"crawler_{timestamp}.log")
    
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
    # 配置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始爬取论文标题")
    
    try:
        # 初始化爬虫
        crawler = CNKICrawler()
        
        # 获取年份范围
        years = list(range(YEAR_RANGE['start_year'], YEAR_RANGE['end_year'] + 1))
        logger.info(f"将爬取以下年份的数据: {years}")
        
        # 爬取数据
        success = crawler.crawl_all(years)
        
        if success:
            logger.info("数据爬取完成")
        else:
            logger.error("数据爬取过程中出现错误")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"爬虫运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
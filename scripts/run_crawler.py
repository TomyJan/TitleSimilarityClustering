#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime
from src.crawler.crawler import Crawler
from src.config import CRAWLER_CONFIG

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
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始爬取论文数据")
    
    try:
        # 初始化爬虫
        crawler = Crawler()
        
        # 获取要处理的年份
        years = list(range(2020, 2024))  # 2020-2023
        logger.info(f"将处理以下年份: {years}")
        
        # 执行爬取
        success = crawler.crawl_years(years)
        if not success:
            logger.error("爬取过程中出现错误")
            sys.exit(1)
            
        logger.info("数据爬取完成")
        
    except Exception as e:
        logger.error(f"爬虫运行出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
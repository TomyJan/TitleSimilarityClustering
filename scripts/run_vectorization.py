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
from src.vectorization.vectorizer import TitleVectorizer
from src.config import YEAR_RANGE

def setup_logging() -> None:
    """配置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vectorization_{timestamp}.log")
    
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
    logger.info("开始文本向量化")
    
    try:
        # 初始化向量化器
        vectorizer = TitleVectorizer()
        
        # 获取年份范围
        years = list(range(YEAR_RANGE['start_year'], YEAR_RANGE['end_year'] + 1))
        logger.info(f"将处理以下年份: {years}")
        
        # 确保输出目录存在
        output_dir = os.path.join("data", "vectorized")
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行向量化
        success = vectorizer.vectorize_all(years, output_dir)
        if not success:
            logger.error("文本向量化过程中出现错误")
            sys.exit(1)
            
        logger.info("文本向量化完成")
        
    except Exception as e:
        logger.error(f"向量化过程出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
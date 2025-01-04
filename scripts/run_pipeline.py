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

def main() -> None:
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始执行处理流水线")
    
    # 定义处理步骤
    steps: List[Tuple[str, str]] = [
        ("scripts/run_preprocessor.py", "预处理"),
        ("scripts/run_vectorization.py", "向量化"),
        ("scripts/run_similarity.py", "相似度计算"),
        ("scripts/run_clustering.py", "聚类分析"),
        ("scripts/run_visualization.py", "可视化生成")
    ]
    
    try:
        # 按顺序执行每个步骤
        for script_path, step_name in steps:
            success = run_step(script_path, step_name)
            if not success:
                logger.error(f"{step_name} 失败，终止流水线")
                sys.exit(1)
        
        logger.info("所有处理步骤执行完成")
        
    except Exception as e:
        logger.error(f"流水线执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
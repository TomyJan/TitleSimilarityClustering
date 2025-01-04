"""运行向量化处理"""
import logging
from src.vectorization.vectorizer import Vectorizer
from src.config import LOGGING_CONFIG
import logging.config

def main():
    # 配置日志
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化向量化器
        vectorizer = Vectorizer()
        
        # 处理所有文件
        logger.info("开始处理所有文件...")
        vectorizer.process_all()
        logger.info("所有文件处理完成")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
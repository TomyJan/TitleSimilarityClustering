"""运行聚类分析的脚本"""
import os
import logging
from datetime import datetime
from src.clustering.clusterer import Clusterer
from src.config import CLUSTERING_CONFIG, PROCESSED_DATA_DIR

def setup_logging():
    """配置日志"""
    os.makedirs('logs', exist_ok=True)
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] %(name)s: %(message)s'
    )
    
    # 配置文件处理器
    file_handler = logging.FileHandler(
        os.path.join('logs', f'clustering_{datetime.now():%Y%m%d_%H%M%S}.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
def get_available_years():
    """获取可用的年份数据"""
    years = set()
    for file in os.listdir(PROCESSED_DATA_DIR):
        if file.startswith('cleaned_titles_') and file.endswith('.csv'):
            try:
                year = int(file.split('_')[-1].split('.')[0])
                years.add(year)
            except ValueError:
                continue
    return sorted(list(years))
    
def main():
    """主函数"""
    # 配置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 获取可用年份
        years = get_available_years()
        if not years:
            logger.error("未找到任何年份的数据")
            return False
            
        logger.info(f"找到以下年份的数据: {years}")
        
        # 初始化聚类器
        clusterer = Clusterer()
        
        # 处理所有年份
        success = clusterer.process_years(years)
        
        if success:
            logger.info("聚类分析完成")
            return True
        else:
            logger.error("聚类分析失败")
            return False
            
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        return False
        
if __name__ == '__main__':
    main() 
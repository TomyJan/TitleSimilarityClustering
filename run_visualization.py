import os
import logging
from src.visualization.visualizer import Visualizer

def setup_logging():
    """配置日志"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 配置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化可视化器
        visualizer = Visualizer()
        
        # 定义要处理的年份
        years = [2020, 2021, 2022, 2023, 2024]
        
        # 批量处理所有可视化任务
        logger.info('开始生成可视化结果...')
        success = visualizer.process_visualizations(years)
        
        if success:
            logger.info('所有可视化任务已完成')
        else:
            logger.error('部分可视化任务失败')
            
    except Exception as e:
        logger.error(f'可视化过程出错: {str(e)}')
        raise

if __name__ == '__main__':
    main()

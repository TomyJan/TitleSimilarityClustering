"""运行标题预处理脚本"""
import logging
from src.preprocessing.preprocessor import TitlePreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 创建预处理器实例
    preprocessor = TitlePreprocessor()
    
    # 处理数据
    success = preprocessor.process_all(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    
    if success:
        print("预处理完成!")
    else:
        print("预处理过程中出现错误!")

if __name__ == "__main__":
    main() 
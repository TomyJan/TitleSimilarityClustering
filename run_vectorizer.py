"""运行文本向量化脚本"""
import logging
from src.vectorization.vectorizer import TfidfVectorizer
from src.vectorization.config import INPUT_DIR, OUTPUT_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    
    # 处理数据
    success = vectorizer.process_all(INPUT_DIR, OUTPUT_DIR)
    
    if success:
        print("向量化完成!")
    else:
        print("向量化过程中出现错误!")

if __name__ == "__main__":
    main() 
"""运行知网论文爬虫"""
import logging
import argparse
from datetime import datetime
from src.crawler.crawler import CNKICrawler

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_date(date_str: str) -> str:
    """解析日期字符串"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的日期格式: {date_str}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='知网论文爬虫')
    parser.add_argument('--start-date', type=parse_date, 
                       default=f"{datetime.now().year}-01-01",
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date,
                       default=f"{datetime.now().year}-12-31",
                       help='结束日期 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # 配置日志
    setup_logging()
    
    # 创建爬虫实例并开始爬取
    crawler = CNKICrawler()
    success = crawler.crawl_papers(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if success:
        print("爬取完成!")
    else:
        print("爬取失败!")
        exit(1)

if __name__ == "__main__":
    main()

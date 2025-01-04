"""论文标题预处理模块"""
import os
import logging
import pandas as pd
import jieba
from typing import List, Optional

logger = logging.getLogger(__name__)

class TitlePreprocessor:
    """论文标题预处理类"""
    
    def __init__(self, stopwords_path: str = "src/preprocessing/stopwords.txt"):
        """初始化预处理器
        
        Args:
            stopwords_path: 停用词表路径
        """
        self.stopwords_path = stopwords_path
        self.stopwords = set()
        self._load_stopwords()
        
    def _load_stopwords(self) -> None:
        """加载停用词表"""
        try:
            if os.path.exists(self.stopwords_path):
                with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f if line.strip())
                logger.info(f"成功加载 {len(self.stopwords)} 个停用词")
            else:
                logger.warning(f"停用词表文件不存在: {self.stopwords_path}")
        except Exception as e:
            logger.error(f"加载停用词表时出错: {str(e)}")
            
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            分词结果列表
        """
        return list(jieba.cut(text.strip()))
        
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """去除停用词
        
        Args:
            tokens: 分词结果列表
            
        Returns:
            去除停用词后的分词列表
        """
        return [token for token in tokens if token not in self.stopwords]
        
    def process_file(self, input_path: str, output_dir: str) -> bool:
        """处理单个文件
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            # 读取数据
            df = pd.read_csv(input_path)
            if 'title' not in df.columns:
                logger.error(f"输入文件缺少title列: {input_path}")
                return False
                
            # 获取年份
            year = int(df['publish_date'].iloc[0][:4])
            
            # 准备基础数据
            result_df = pd.DataFrame({
                'id': df['id'],
                'title': df['title'],
                'year': year
            })
            
            # 分词处理
            logger.info("正在进行分词处理...")
            result_df['tokens'] = df['title'].apply(lambda x: ' '.join(self._tokenize(x)))
            
            # 保存分词结果
            tokenized_path = os.path.join(output_dir, f"tokenized_titles_{year}.csv")
            os.makedirs(os.path.dirname(tokenized_path), exist_ok=True)
            result_df[['id', 'title', 'tokens', 'year']].to_csv(
                tokenized_path, index=False, encoding='utf-8'
            )
            logger.info(f"分词结果已保存至: {tokenized_path}")
            
            # 去停用词处理
            logger.info("正在进行去停用词处理...")
            result_df['cleaned_tokens'] = result_df['tokens'].apply(
                lambda x: ' '.join(self._remove_stopwords(x.split()))
            )
            
            # 保存清洗结果
            cleaned_path = os.path.join(output_dir, f"cleaned_titles_{year}.csv")
            result_df[['id', 'title', 'cleaned_tokens', 'year']].to_csv(
                cleaned_path, index=False, encoding='utf-8'
            )
            logger.info(f"清洗结果已保存至: {cleaned_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            return False
            
    def process_all(self, years: List[int], output_dir: str) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        try:
            success = True
            for year in years:
                input_path = os.path.join("data", "raw", f"thesis_titles_{year}.csv")
                logger.info(f"正在处理 {year} 年的数据...")
                if not self.process_file(input_path, output_dir):
                    success = False
            return success
        except Exception as e:
            logger.error(f"批量处理文件时出错: {str(e)}")
            return False 
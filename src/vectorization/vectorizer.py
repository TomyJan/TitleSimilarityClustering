"""向量化模块"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from src.config import VECTORIZATION_CONFIG, PROCESSED_DATA_DIR

class VectorizationError(Exception):
    """向量化错误的基类"""
    pass

class Vectorizer:
    """向量化类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化向量化器
        
        Args:
            config: 向量化配置，如果为None则使用默认配置
        """
        self.config = config or VECTORIZATION_CONFIG
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.char_features = None
        
    def _load_titles(self, year: int) -> pd.DataFrame:
        """加载标题数据
        
        Args:
            year: 年份
            
        Returns:
            包含标题的DataFrame
            
        Raises:
            VectorizationError: 如果文件不存在或格式错误
        """
        try:
            file_path = os.path.join(PROCESSED_DATA_DIR, f"cleaned_titles_{year}.csv")
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['id', 'title']):
                raise VectorizationError(f"标题文件缺少必需的列: {file_path}")
            return df
        except Exception as e:
            raise VectorizationError(f"加载标题文件时出错: {str(e)}")
            
    def _load_tokens(self, year: int) -> pd.DataFrame:
        """加载分词数据
        
        Args:
            year: 年份
            
        Returns:
            包含分词结果的DataFrame
            
        Raises:
            VectorizationError: 如果文件不存在或格式错误
        """
        try:
            file_path = os.path.join(PROCESSED_DATA_DIR, f"tokenized_titles_{year}.csv")
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['id', 'tokens']):
                raise VectorizationError(f"分词文件缺少必需的列: {file_path}")
            # 将字符串形式的tokens转换为列表
            df['tokens'] = df['tokens'].str.split()
            return df
        except Exception as e:
            raise VectorizationError(f"加载分词文件时出错: {str(e)}")
            
    def _get_all_tokens(self, years: List[int]) -> List[List[str]]:
        """获取所有年份的分词结果
        
        Args:
            years: 年份列表
            
        Returns:
            所有分词结果的列表
        """
        all_tokens = []
        for year in years:
            df = self._load_tokens(year)
            all_tokens.extend(df['tokens'].tolist())
        return all_tokens
        
    def fit_tfidf(self, years: List[int]) -> None:
        """训练TF-IDF向量化器
        
        Args:
            years: 年份列表
        """
        try:
            # 获取所有年份的分词结果
            all_tokens = self._get_all_tokens(years)
            
            # 将分词列表转换为文本
            texts = [' '.join(tokens) for tokens in all_tokens]
            
            # 创建并训练TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                min_df=self.config['tfidf']['min_df'],
                max_df=self.config['tfidf']['max_df']
            )
            self.tfidf_vectorizer.fit(texts)
            
            # 保存特征名称
            features_file = os.path.join(PROCESSED_DATA_DIR, 'tfidf_features.json')
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'features': list(self.tfidf_vectorizer.get_feature_names_out()),
                        'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_)
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
                
        except Exception as e:
            raise VectorizationError(f"训练TF-IDF向量化器时出错: {str(e)}")
            
    def fit_word2vec(self, years: List[int]) -> None:
        """训练Word2Vec模型
        
        Args:
            years: 年份列表
        """
        try:
            # 获取所有年份的分词结果
            all_tokens = self._get_all_tokens(years)
            
            # 训练Word2Vec模型
            self.word2vec_model = Word2Vec(
                sentences=all_tokens,
                vector_size=self.config['word2vec']['vector_size'],
                window=self.config['word2vec']['window'],
                min_count=self.config['word2vec']['min_count'],
                workers=self.config['word2vec']['workers']
            )
            
            # 保存模型
            model_file = os.path.join(PROCESSED_DATA_DIR, 'word2vec_model.bin')
            self.word2vec_model.save(model_file)
            
        except Exception as e:
            raise VectorizationError(f"训练Word2Vec模型时出错: {str(e)}")
            
    def fit_char_vectors(self, years: List[int]) -> None:
        """创建字符级特征
        
        Args:
            years: 年份列表
        """
        try:
            # 获取所有年份的标题
            all_titles = []
            for year in years:
                df = self._load_titles(year)
                all_titles.extend(df['title'].tolist())
                
            # 获取所有唯一字符
            chars = sorted(list(set(''.join(all_titles))))
            self.char_features = {char: i for i, char in enumerate(chars)}
            
            # 保存特征
            features_file = os.path.join(PROCESSED_DATA_DIR, 'char_features.json')
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'features': self.char_features,
                        'vocabulary_size': len(chars)
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
                
        except Exception as e:
            raise VectorizationError(f"创建字符级特征时出错: {str(e)}")
            
    def transform_tfidf(self, year: int) -> sparse.csr_matrix:
        """将指定年份的数据转换为TF-IDF向量
        
        Args:
            year: 年份
            
        Returns:
            TF-IDF向量矩阵
        """
        try:
            if self.tfidf_vectorizer is None:
                raise VectorizationError("TF-IDF向量化器未训练")
                
            # 加载分词数据
            df = self._load_tokens(year)
            texts = [' '.join(tokens) for tokens in df['tokens']]
            
            # 转换为TF-IDF向量
            vectors = self.tfidf_vectorizer.transform(texts)
            
            # 保存向量
            output_file = os.path.join(PROCESSED_DATA_DIR, f"tfidf_vectors_{year}.npz")
            sparse.save_npz(output_file, vectors)
            
            return vectors
            
        except Exception as e:
            raise VectorizationError(f"转换TF-IDF向量时出错: {str(e)}")
            
    def transform_word2vec(self, year: int) -> np.ndarray:
        """将指定年份的数据转换为Word2Vec向量
        
        Args:
            year: 年份
            
        Returns:
            Word2Vec向量矩阵
        """
        try:
            if self.word2vec_model is None:
                raise VectorizationError("Word2Vec模型未训练")
                
            # 加载分词数据
            df = self._load_tokens(year)
            
            # 计算每个标题的平均词向量
            vectors = []
            for tokens in df['tokens']:
                token_vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        token_vectors.append(self.word2vec_model.wv[token])
                if token_vectors:
                    vectors.append(np.mean(token_vectors, axis=0))
                else:
                    vectors.append(np.zeros(self.config['word2vec']['vector_size']))
                    
            vectors = np.array(vectors)
            
            # 保存向量
            output_file = os.path.join(PROCESSED_DATA_DIR, f"word2vec_vectors_{year}.npy")
            np.save(output_file, vectors)
            
            return vectors
            
        except Exception as e:
            raise VectorizationError(f"转换Word2Vec向量时出错: {str(e)}")
            
    def transform_char_vectors(self, year: int) -> sparse.csr_matrix:
        """将指定年份的数据转换为字符级向量
        
        Args:
            year: 年份
            
        Returns:
            字符级向量矩阵
        """
        try:
            if self.char_features is None:
                raise VectorizationError("字符级特征未创建")
                
            # 加载标题数据
            df = self._load_titles(year)
            
            # 创建稀疏矩阵
            rows, cols, data = [], [], []
            for i, title in enumerate(df['title']):
                for char in title:
                    if char in self.char_features:
                        rows.append(i)
                        cols.append(self.char_features[char])
                        data.append(1)
                        
            vectors = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(df), len(self.char_features))
            )
            
            # 保存向量
            output_file = os.path.join(PROCESSED_DATA_DIR, f"char_vectors_{year}.npz")
            sparse.save_npz(output_file, vectors)
            
            return vectors
            
        except Exception as e:
            raise VectorizationError(f"转换字符级向量时出错: {str(e)}")
            
    def process_years(self, years: List[int]) -> bool:
        """处理多个年份的数据
        
        Args:
            years: 年份列表
            
        Returns:
            处理是否成功
        """
        try:
            # 训练向量化器和模型
            self.logger.info("开始训练向量化器和模型...")
            self.fit_tfidf(years)
            self.fit_word2vec(years)
            self.fit_char_vectors(years)
            
            # 转换每个年份的数据
            for year in years:
                self.logger.info(f"正在处理 {year} 年的数据...")
                self.transform_tfidf(year)
                self.transform_word2vec(year)
                self.transform_char_vectors(year)
                
            return True
            
        except Exception as e:
            self.logger.error(f"处理年份数据时出错: {str(e)}")
            raise

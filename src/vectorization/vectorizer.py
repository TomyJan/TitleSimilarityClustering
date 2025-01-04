"""文本向量化模块"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
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
    """文本向量化类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化向量化器
        
        Args:
            config: 向量化配置，如果为None则使用默认配置
        """
        self.config = config or VECTORIZATION_CONFIG
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.char_to_id = None
        
    def _save_features(self, features: List[str], filename: str) -> None:
        """保存特征列表
        
        Args:
            features: 特征列表
            filename: 保存的文件名
        """
        with open(os.path.join(PROCESSED_DATA_DIR, filename), 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
            
    def _load_features(self, filename: str) -> List[str]:
        """加载特征列表
        
        Args:
            filename: 特征文件名
            
        Returns:
            特征列表
        """
        with open(os.path.join(PROCESSED_DATA_DIR, filename), 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def fit_tfidf(self, texts: List[str]) -> sparse.csr_matrix:
        """训练TF-IDF向量化器并转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF向量矩阵
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf']['max_features'],
            min_df=self.config['tfidf']['min_df'],
            max_df=self.config['tfidf']['max_df']
        )
        vectors = self.tfidf_vectorizer.fit_transform(texts)
        self._save_features(
            self.tfidf_vectorizer.get_feature_names_out().tolist(),
            'tfidf_features.json'
        )
        return vectors
        
    def transform_tfidf(self, texts: List[str]) -> sparse.csr_matrix:
        """使用已训练的TF-IDF向量化器转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF向量矩阵
            
        Raises:
            VectorizationError: 如果向量化器未训练
        """
        if self.tfidf_vectorizer is None:
            raise VectorizationError("TF-IDF向量化器未训练")
        return self.tfidf_vectorizer.transform(texts)
        
    def fit_word2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """训练Word2Vec模型并转换文本
        
        Args:
            tokenized_texts: 分词后的文本列表
            
        Returns:
            Word2Vec向量矩阵
        """
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.config['word2vec']['vector_size'],
            window=self.config['word2vec']['window'],
            min_count=self.config['word2vec']['min_count']
        )
        self.word2vec_model.save(os.path.join(PROCESSED_DATA_DIR, 'word2vec_model.bin'))
        
        # 计算每个文本的平均词向量
        vectors = np.zeros((len(tokenized_texts), self.config['word2vec']['vector_size']))
        for i, tokens in enumerate(tokenized_texts):
            token_vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    token_vectors.append(self.word2vec_model.wv[token])
            if token_vectors:
                vectors[i] = np.mean(token_vectors, axis=0)
        return vectors
        
    def transform_word2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """使用已训练的Word2Vec模型转换文本
        
        Args:
            tokenized_texts: 分词后的文本列表
            
        Returns:
            Word2Vec向量矩阵
            
        Raises:
            VectorizationError: 如果模型未训练
        """
        if self.word2vec_model is None:
            raise VectorizationError("Word2Vec模型未训练")
            
        vectors = np.zeros((len(tokenized_texts), self.config['word2vec']['vector_size']))
        for i, tokens in enumerate(tokenized_texts):
            token_vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    token_vectors.append(self.word2vec_model.wv[token])
            if token_vectors:
                vectors[i] = np.mean(token_vectors, axis=0)
        return vectors
        
    def fit_char_vectors(self, texts: List[str]) -> sparse.csr_matrix:
        """训练字符级向量化器并转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            字符级one-hot编码矩阵
        """
        # 构建字符映射
        chars = set()
        for text in texts:
            chars.update(text)
        self.char_to_id = {char: i for i, char in enumerate(sorted(chars))}
        self._save_features(
            list(self.char_to_id.keys()),
            'char_features.json'
        )
        
        # 转换为one-hot编码
        rows, cols, data = [], [], []
        for i, text in enumerate(texts):
            for char in text:
                if char in self.char_to_id:
                    rows.append(i)
                    cols.append(self.char_to_id[char])
                    data.append(1)
        
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(texts), len(self.char_to_id))
        )
        
    def transform_char_vectors(self, texts: List[str]) -> sparse.csr_matrix:
        """使用已训练的字符级向量化器转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            字符级one-hot编码矩阵
            
        Raises:
            VectorizationError: 如果向量化器未训练
        """
        if self.char_to_id is None:
            raise VectorizationError("字符级向量化器未训练")
            
        rows, cols, data = [], [], []
        for i, text in enumerate(texts):
            for char in text:
                if char in self.char_to_id:
                    rows.append(i)
                    cols.append(self.char_to_id[char])
                    data.append(1)
        
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(texts), len(self.char_to_id))
        )
        
    def process_file(self, input_path: str) -> bool:
        """处理单个文件
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            处理是否成功
        """
        try:
            # 读取数据
            df = pd.read_csv(input_path)
            if 'cleaned_tokens' not in df.columns:
                raise VectorizationError(f"输入文件缺少cleaned_tokens列: {input_path}")
                
            year = df['year'].iloc[0]
            texts = df['cleaned_tokens'].tolist()
            tokenized_texts = [text.split() for text in texts]
            
            # TF-IDF向量化
            self.logger.info("正在进行TF-IDF向量化...")
            tfidf_vectors = self.fit_tfidf(texts)
            sparse.save_npz(
                os.path.join(PROCESSED_DATA_DIR, f"tfidf_vectors_{year}.npz"),
                tfidf_vectors
            )
            
            # Word2Vec向量化
            self.logger.info("正在进行Word2Vec向量化...")
            word2vec_vectors = self.fit_word2vec(tokenized_texts)
            np.save(
                os.path.join(PROCESSED_DATA_DIR, f"word2vec_vectors_{year}.npy"),
                word2vec_vectors
            )
            
            # 字符级向量化
            self.logger.info("正在进行字符级向量化...")
            char_vectors = self.fit_char_vectors(df['title'].tolist())
            sparse.save_npz(
                os.path.join(PROCESSED_DATA_DIR, f"char_vectors_{year}.npz"),
                char_vectors
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件时出错: {str(e)}")
            raise
            
    def process_all(self, input_dir: str = str(PROCESSED_DATA_DIR)) -> bool:
        """处理目录下的所有文件
        
        Args:
            input_dir: 输入目录
            
        Returns:
            处理是否成功
        """
        try:
            success = True
            for filename in os.listdir(input_dir):
                if filename.startswith('cleaned_titles_') and filename.endswith('.csv'):
                    input_path = os.path.join(input_dir, filename)
                    self.logger.info(f"正在处理文件: {filename}")
                    if not self.process_file(input_path):
                        success = False
            return success
        except Exception as e:
            self.logger.error(f"批量处理文件时出错: {str(e)}")
            raise 
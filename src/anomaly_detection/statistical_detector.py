"""
统计异常检测器
基于文本统计特征和历史数据分布来检测异常，包括文本长度、词汇新颖性、情感强度等
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
import re

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class StatisticalAnomalyDetector:
    """
    统计异常检测器
    
    基于历史数据的统计分布来检测文本异常，包括：
    1. 文本长度异常 - 基于历史长度分布
    2. 词汇新颖性异常 - 新词汇占比过高  
    3. 情感强度异常 - 情感倾向超出正常范围
    4. 数值预测异常 - 数值预测偏离历史分布
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 配置参数
        text_processor: 文本处理器
        historical_data: 历史统计数据
        is_trained: 是否已训练
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化统计异常检测器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        self.file_manager = get_file_manager()
        
        # 历史统计数据
        self.historical_data = {
            'text_lengths': [],
            'vocabulary': set(),
            'word_frequencies': defaultdict(int),
            'sentiment_scores': [],
            'numerical_predictions': [],
            'report_count': 0,
            'last_update': None
        }
        
        # 统计模型
        self.length_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        self.sentiment_stats = {'mean': 0, 'std': 0}
        self.prediction_stats = {'mean': 0, 'std': 0}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.scaler = StandardScaler()
        
        self.is_trained = False
        
        # 加载历史数据
        self._load_historical_data()
        
        logger.info("统计异常检测器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('statistical_anomaly', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'text_length_threshold': 2.0,      # 文本长度Z-score阈值
            'vocabulary_novelty_threshold': 0.15,  # 新词汇占比阈值
            'sentiment_intensity_threshold': 2.5,  # 情感强度Z-score阈值
            'numerical_deviation_threshold': 0.95, # 数值偏离分位数阈值
            'min_historical_samples': 50,      # 最少历史样本数
            'update_frequency_days': 7,        # 更新频率（天）
            'max_vocabulary_size': 50000,      # 最大词汇表大小
            'sentiment_keywords': {
                'positive': ['利好', '上涨', '增长', '优秀', '强劲', '超预期'],
                'negative': ['利空', '下跌', '下降', '风险', '压力', '低于预期'],
                'neutral': ['稳定', '维持', '持平', '观望', '中性']
            }
        }
    
    def _load_historical_data(self):
        """加载历史统计数据"""
        try:
            data_path = Path("data/models/statistical_history.pkl")
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                self.historical_data.update(loaded_data)
                
                # 重建统计模型
                if len(self.historical_data['text_lengths']) >= self.config['min_historical_samples']:
                    self._build_statistical_models()
                    logger.info(f"已加载 {self.historical_data['report_count']} 份历史数据")
                
        except Exception as e:
            logger.warning(f"历史数据加载失败: {e}")
    
    def _save_historical_data(self):
        """保存历史统计数据"""
        try:
            data_path = Path("data/models/statistical_history.pkl")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 限制词汇表大小
            if len(self.historical_data['vocabulary']) > self.config['max_vocabulary_size']:
                # 保留高频词汇
                word_freq = self.historical_data['word_frequencies']
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                top_words = top_words[:self.config['max_vocabulary_size']]
                self.historical_data['vocabulary'] = {word for word, _ in top_words}
            
            with open(data_path, 'wb') as f:
                pickle.dump(self.historical_data, f)
                
            logger.debug("历史数据已保存")
            
        except Exception as e:
            logger.error(f"历史数据保存失败: {e}")
    
    def update_historical_data(self, texts: List[str], update_vocabulary: bool = True):
        """
        更新历史统计数据
        
        Args:
            texts: 文本列表
            update_vocabulary: 是否更新词汇表
        """
        try:
            for text in texts:
                # 更新文本长度
                self.historical_data['text_lengths'].append(len(text))
                
                # 更新词汇表和词频
                if update_vocabulary:
                    words = self.text_processor.segment_text(text)
                    for word in words:
                        self.historical_data['vocabulary'].add(word)
                        self.historical_data['word_frequencies'][word] += 1
                
                # 更新情感分数
                sentiment_score = self._calculate_sentiment_score(text)
                self.historical_data['sentiment_scores'].append(sentiment_score)
                
                # 提取数值预测
                numerical_values = self._extract_numerical_predictions(text)
                self.historical_data['numerical_predictions'].extend(numerical_values)
                
                self.historical_data['report_count'] += 1
            
            # 更新时间戳
            self.historical_data['last_update'] = datetime.now()
            
            # 重建统计模型
            if len(self.historical_data['text_lengths']) >= self.config['min_historical_samples']:
                self._build_statistical_models()
            
            # 保存数据
            self._save_historical_data()
            
            logger.info(f"已更新 {len(texts)} 份文档的历史统计数据")
            
        except Exception as e:
            logger.error(f"历史数据更新失败: {e}")
            raise
    
    def _build_statistical_models(self):
        """构建统计模型"""
        try:
            # 文本长度统计
            lengths = np.array(self.historical_data['text_lengths'])
            self.length_stats = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'percentiles': np.percentile(lengths, [5, 25, 50, 75, 95])
            }
            
            # 情感强度统计
            sentiments = np.array(self.historical_data['sentiment_scores'])
            self.sentiment_stats = {
                'mean': np.mean(sentiments),
                'std': np.std(sentiments),
                'percentiles': np.percentile(sentiments, [5, 25, 50, 75, 95])
            }
            
            # 数值预测统计
            if self.historical_data['numerical_predictions']:
                predictions = np.array(self.historical_data['numerical_predictions'])
                self.prediction_stats = {
                    'mean': np.mean(predictions),
                    'std': np.std(predictions),
                    'percentiles': np.percentile(predictions, [5, 25, 50, 75, 95])
                }
            
            self.is_trained = True
            logger.info("统计模型构建完成")
            
        except Exception as e:
            logger.error(f"统计模型构建失败: {e}")
            raise
    
    def detect_statistical_anomalies(self, text: str) -> Dict[str, Any]:
        """
        检测统计异常
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 异常检测结果
            
        Raises:
            RuntimeError: 检测器未训练
        """
        if not self.is_trained:
            if len(self.historical_data['text_lengths']) < self.config['min_historical_samples']:
                raise RuntimeError(f"历史样本不足，需要至少 {self.config['min_historical_samples']} 个样本")
            else:
                self._build_statistical_models()
        
        try:
            results = {
                'text_length_anomaly': self._detect_text_length_anomaly(text),
                'vocabulary_novelty_anomaly': self._detect_vocabulary_novelty(text),
                'sentiment_intensity_anomaly': self._detect_sentiment_anomaly(text),
                'numerical_prediction_anomaly': self._detect_numerical_anomaly(text),
                'overall_score': 0.0,
                'anomaly_level': 'NORMAL',
                'details': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 计算综合异常分数
            anomaly_scores = [
                results['text_length_anomaly']['score'],
                results['vocabulary_novelty_anomaly']['score'],
                results['sentiment_intensity_anomaly']['score'],
                results['numerical_prediction_anomaly']['score']
            ]
            
            # 加权平均（可配置权重）
            weights = [0.2, 0.3, 0.3, 0.2]  # 默认权重
            results['overall_score'] = np.average(anomaly_scores, weights=weights)
            
            # 确定异常等级
            results['anomaly_level'] = self._determine_anomaly_level(results['overall_score'])
            
            # 详细信息
            results['details'] = {
                'sample_count': self.historical_data['report_count'],
                'last_update': self.historical_data['last_update'].isoformat() if self.historical_data['last_update'] else None,
                'text_length': len(text),
                'word_count': len(self.text_processor.segment_text(text)),
                'unique_words': len(set(self.text_processor.segment_text(text)))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"统计异常检测失败: {e}")
            raise
    
    def _detect_text_length_anomaly(self, text: str) -> Dict[str, Any]:
        """检测文本长度异常"""
        text_length = len(text)
        
        # 计算Z-score
        z_score = abs(text_length - self.length_stats['mean']) / max(self.length_stats['std'], 1e-6)
        
        # 判断异常
        is_anomaly = z_score > self.config['text_length_threshold']
        
        # 异常分数 (0-1)
        anomaly_score = min(z_score / self.config['text_length_threshold'], 1.0)
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'z_score': z_score,
            'current_length': text_length,
            'historical_mean': self.length_stats['mean'],
            'historical_std': self.length_stats['std'],
            'description': f"文本长度{text_length}，Z-score={z_score:.2f}"
        }
    
    def _detect_vocabulary_novelty(self, text: str) -> Dict[str, Any]:
        """检测词汇新颖性异常"""
        words = self.text_processor.segment_text(text)
        unique_words = set(words)
        
        # 计算新词汇比例
        new_words = unique_words - self.historical_data['vocabulary']
        novelty_ratio = len(new_words) / max(len(unique_words), 1)
        
        # 判断异常
        is_anomaly = novelty_ratio > self.config['vocabulary_novelty_threshold']
        
        # 异常分数
        anomaly_score = min(novelty_ratio / self.config['vocabulary_novelty_threshold'], 1.0)
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'novelty_ratio': novelty_ratio,
            'new_words_count': len(new_words),
            'total_unique_words': len(unique_words),
            'new_words': list(new_words)[:10],  # 只显示前10个新词
            'description': f"新词汇占比{novelty_ratio:.2%}，发现{len(new_words)}个新词"
        }
    
    def _detect_sentiment_anomaly(self, text: str) -> Dict[str, Any]:
        """检测情感强度异常"""
        sentiment_score = self._calculate_sentiment_score(text)
        
        # 计算Z-score
        z_score = abs(sentiment_score - self.sentiment_stats['mean']) / max(self.sentiment_stats['std'], 1e-6)
        
        # 判断异常
        is_anomaly = z_score > self.config['sentiment_intensity_threshold']
        
        # 异常分数
        anomaly_score = min(z_score / self.config['sentiment_intensity_threshold'], 1.0)
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'z_score': z_score,
            'sentiment_score': sentiment_score,
            'historical_mean': self.sentiment_stats['mean'],
            'historical_std': self.sentiment_stats['std'],
            'description': f"情感强度{sentiment_score:.2f}，Z-score={z_score:.2f}"
        }
    
    def _detect_numerical_anomaly(self, text: str) -> Dict[str, Any]:
        """检测数值预测异常"""
        numerical_values = self._extract_numerical_predictions(text)
        
        if not numerical_values or not self.historical_data['numerical_predictions']:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无数值预测数据"
            }
        
        # 计算与历史分布的偏离
        historical_percentiles = self.prediction_stats.get('percentiles', [0, 25, 50, 75, 100])
        
        anomaly_scores = []
        anomalous_values = []
        
        for value in numerical_values:
            # 检查是否超出历史分位数范围
            lower_bound = historical_percentiles[0]  # 5%分位数
            upper_bound = historical_percentiles[4]  # 95%分位数
            
            if value < lower_bound or value > upper_bound:
                # 计算偏离程度
                if value < lower_bound:
                    deviation = (lower_bound - value) / max(abs(lower_bound), 1e-6)
                else:
                    deviation = (value - upper_bound) / max(abs(upper_bound), 1e-6)
                
                anomaly_scores.append(min(deviation, 1.0))
                anomalous_values.append(value)
        
        if anomaly_scores:
            overall_score = max(anomaly_scores)
            is_anomaly = overall_score > 0.5
        else:
            overall_score = 0.0
            is_anomaly = False
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'numerical_values': numerical_values,
            'anomalous_values': anomalous_values,
            'historical_range': [historical_percentiles[0], historical_percentiles[4]],
            'description': f"发现{len(numerical_values)}个数值，{len(anomalous_values)}个异常"
        }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        计算情感分数
        
        Returns:
            float: 情感分数 (-1到1，负数表示消极，正数表示积极)
        """
        words = self.text_processor.segment_text(text)
        
        positive_words = self.config['sentiment_keywords']['positive']
        negative_words = self.config['sentiment_keywords']['negative']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # 计算情感分数
        sentiment_score = (positive_count - negative_count) / total_words
        
        return sentiment_score
    
    def _extract_numerical_predictions(self, text: str) -> List[float]:
        """
        提取数值预测
        
        Returns:
            List[float]: 数值预测列表
        """
        numerical_values = []
        
        # 匹配百分比、倍数、价格等数值
        patterns = [
            r'(\d+\.?\d*)\s*%',  # 百分比
            r'(\d+\.?\d*)\s*倍',  # 倍数
            r'(\d+\.?\d*)\s*元',  # 价格
            r'(\d+\.?\d*)\s*亿',  # 金额
            r'(\d+\.?\d*)\s*万',  # 金额
            r'增长\s*(\d+\.?\d*)',  # 增长率
            r'下降\s*(\d+\.?\d*)',  # 下降率
            r'目标价\s*(\d+\.?\d*)'  # 目标价
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    value = float(match.group(1))
                    numerical_values.append(value)
                except ValueError:
                    continue
        
        return numerical_values
    
    def _determine_anomaly_level(self, score: float) -> str:
        """
        确定异常等级
        
        Args:
            score: 异常分数 (0-1)
            
        Returns:
            str: 异常等级
        """
        if score >= 0.8:
            return 'CRITICAL'
        elif score >= 0.6:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        elif score >= 0.2:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Returns:
            Dict[str, Any]: 统计摘要
        """
        return {
            'report_count': self.historical_data['report_count'],
            'vocabulary_size': len(self.historical_data['vocabulary']),
            'last_update': self.historical_data['last_update'].isoformat() if self.historical_data['last_update'] else None,
            'text_length_stats': self.length_stats,
            'sentiment_stats': self.sentiment_stats,
            'prediction_stats': self.prediction_stats,
            'is_trained': self.is_trained,
            'config': self.config
        }
    
    def reset_historical_data(self):
        """重置历史数据"""
        self.historical_data = {
            'text_lengths': [],
            'vocabulary': set(),
            'word_frequencies': defaultdict(int),
            'sentiment_scores': [],
            'numerical_predictions': [],
            'report_count': 0,
            'last_update': None
        }
        
        self.is_trained = False
        logger.info("历史数据已重置")


# 全局检测器实例
_global_statistical_detector = None


def get_statistical_detector() -> StatisticalAnomalyDetector:
    """
    获取全局统计异常检测器实例
    
    Returns:
        StatisticalAnomalyDetector: 检测器实例
    """
    global _global_statistical_detector
    
    if _global_statistical_detector is None:
        _global_statistical_detector = StatisticalAnomalyDetector()
    
    return _global_statistical_detector


if __name__ == "__main__":
    # 使用示例
    detector = StatisticalAnomalyDetector()
    
    # 模拟历史数据
    sample_texts = [
        "近期A股市场表现稳定，预计未来增长5%左右，目标价25元。",
        "公司业绩超预期，营收增长15%，利润率提升3个百分点。",
        "市场风险加大，建议谨慎投资，预期下降2%。"
    ] * 20  # 扩充样本
    
    # 更新历史数据
    detector.update_historical_data(sample_texts)
    
    # 检测异常
    test_text = "这是一个极其异常的超长文本，包含了大量前所未见的新词汇，情感极度激烈，预测暴涨1000%！！！"
    result = detector.detect_statistical_anomalies(test_text)
    
    print("统计异常检测结果:")
    print(f"整体异常分数: {result['overall_score']:.3f}")
    print(f"异常等级: {result['anomaly_level']}")
    print(f"文本长度异常: {result['text_length_anomaly']['is_anomaly']}")
    print(f"词汇新颖性异常: {result['vocabulary_novelty_anomaly']['is_anomaly']}")
    print(f"情感强度异常: {result['sentiment_intensity_anomaly']['is_anomaly']}")
    print(f"数值预测异常: {result['numerical_prediction_anomaly']['is_anomaly']}") 
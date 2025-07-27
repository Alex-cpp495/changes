"""
特征提取器
从研报数据中提取多维度特征，包括文本特征、数值特征、时间特征、元数据特征等
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging
import math

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor

logger = get_logger(__name__)


class FeatureExtractor:
    """
    特征提取器
    
    从研报数据中提取多维度特征：
    1. 文本特征 - 长度、复杂度、情感、关键词等
    2. 数值特征 - 预测数据、财务指标、比率等
    3. 时间特征 - 发布时间、市场时间、周期性等
    4. 元数据特征 - 作者、机构、股票等结构化信息
    5. 语义特征 - 主题、实体、关系等深度特征
    6. 行为特征 - 发布模式、频率、变化等
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 特征提取配置
        text_processor: 文本处理器
        feature_cache: 特征缓存
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化特征提取器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        
        # 特征缓存（用于增量特征计算）
        self.feature_cache = {
            'author_history': defaultdict(list),
            'stock_history': defaultdict(list),
            'time_features': {},
            'vocabulary_stats': Counter()
        }
        
        # 预编译正则表达式
        self.patterns = self._compile_patterns()
        
        # 统计信息
        self.extraction_stats = {
            'total_extracted': 0,
            'feature_dimensions': 0,
            'processing_time': 0.0
        }
        
        logger.info("特征提取器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('feature_extraction', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'text_features': {
                'extract_length_features': True,
                'extract_complexity_features': True,
                'extract_sentiment_features': True,
                'extract_keyword_features': True,
                'extract_readability_features': True,
                'max_keywords': 20,
                'min_keyword_freq': 2
            },
            'numerical_features': {
                'extract_prediction_features': True,
                'extract_financial_ratios': True,
                'extract_statistical_features': True,
                'normalize_numbers': True
            },
            'temporal_features': {
                'extract_time_features': True,
                'extract_market_timing': True,
                'extract_cyclical_features': True,
                'timezone': 'Asia/Shanghai'
            },
            'metadata_features': {
                'extract_author_features': True,
                'extract_institution_features': True,
                'extract_stock_features': True,
                'extract_category_features': True
            },
            'semantic_features': {
                'extract_topic_features': True,
                'extract_entity_features': True,
                'extract_dependency_features': False,  # 需要额外模型
                'topic_num': 10
            },
            'behavioral_features': {
                'extract_publishing_patterns': True,
                'extract_frequency_features': True,
                'extract_change_features': True,
                'history_window_days': 90
            },
            'feature_engineering': {
                'create_interaction_features': True,
                'create_ratio_features': True,
                'create_polynomial_features': False,
                'polynomial_degree': 2,
                'normalize_features': True
            }
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译正则表达式模式"""
        patterns = {}
        
        # 数字和单位模式
        patterns['number_with_unit'] = re.compile(r'\d+(?:\.\d+)?[万亿千百十元美金%‰]+')
        patterns['percentage'] = re.compile(r'\d+(?:\.\d+)?%')
        patterns['price'] = re.compile(r'\d+(?:\.\d+)?元')
        patterns['currency'] = re.compile(r'\d+(?:\.\d+)?[万亿千百十]?[元美金]')
        
        # 时间模式
        patterns['date'] = re.compile(r'\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?')
        patterns['quarter'] = re.compile(r'\d{4}年[第]?[一二三四1234]季度')
        patterns['year'] = re.compile(r'\d{4}年')
        
        # 股票和公司模式
        patterns['stock_code'] = re.compile(r'[0-9]{6}(?:\.[A-Z]{2})?')
        patterns['company_name'] = re.compile(r'[\u4e00-\u9fff]{2,10}(?:股份)?(?:有限)?公司')
        
        # 评级和预测模式
        patterns['rating'] = re.compile(r'[买卖增减持]{1,2}[入出持]?|中性|推荐')
        patterns['target_price'] = re.compile(r'目标价\s*[:：]?\s*(\d+(?:\.\d+)?)')
        patterns['growth_rate'] = re.compile(r'增长\s*(\d+(?:\.\d+)?)%')
        
        # 情感和态度模式
        patterns['positive_words'] = re.compile(r'利好|上涨|增长|优秀|强劲|看好|推荐|买入')
        patterns['negative_words'] = re.compile(r'利空|下跌|下降|风险|压力|看空|卖出|减持')
        patterns['uncertain_words'] = re.compile(r'可能|或许|预计|估计|大概|不确定')
        
        return patterns
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取完整特征集
        
        Args:
            data: 研报数据
            
        Returns:
            Dict[str, Any]: 提取的特征
        """
        start_time = datetime.now()
        
        try:
            features = {
                'text_features': {},
                'numerical_features': {},
                'temporal_features': {},
                'metadata_features': {},
                'semantic_features': {},
                'behavioral_features': {},
                'engineered_features': {},
                'feature_metadata': {
                    'extraction_time': None,
                    'feature_count': 0,
                    'data_source': data.get('source', 'unknown')
                }
            }
            
            # 1. 文本特征提取
            if self.config['text_features']['extract_length_features']:
                features['text_features'].update(self._extract_text_features(data))
            
            # 2. 数值特征提取
            if self.config['numerical_features']['extract_prediction_features']:
                features['numerical_features'].update(self._extract_numerical_features(data))
            
            # 3. 时间特征提取
            if self.config['temporal_features']['extract_time_features']:
                features['temporal_features'].update(self._extract_temporal_features(data))
            
            # 4. 元数据特征提取
            if self.config['metadata_features']['extract_author_features']:
                features['metadata_features'].update(self._extract_metadata_features(data))
            
            # 5. 语义特征提取
            if self.config['semantic_features']['extract_topic_features']:
                features['semantic_features'].update(self._extract_semantic_features(data))
            
            # 6. 行为特征提取
            if self.config['behavioral_features']['extract_publishing_patterns']:
                features['behavioral_features'].update(self._extract_behavioral_features(data))
            
            # 7. 特征工程
            if self.config['feature_engineering']['create_interaction_features']:
                features['engineered_features'].update(self._engineer_features(features))
            
            # 计算总特征数
            total_features = sum(len(category) for category in features.values() if isinstance(category, dict))
            
            # 更新元数据
            processing_time = (datetime.now() - start_time).total_seconds()
            features['feature_metadata'].update({
                'extraction_time': processing_time,
                'feature_count': total_features,
                'timestamp': datetime.now().isoformat()
            })
            
            # 更新统计信息
            self.extraction_stats['total_extracted'] += 1
            self.extraction_stats['feature_dimensions'] = total_features
            self.extraction_stats['processing_time'] += processing_time
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return {
                'error': str(e),
                'feature_metadata': {
                    'extraction_time': (datetime.now() - start_time).total_seconds(),
                    'feature_count': 0,
                    'error': str(e)
                }
            }
    
    def _extract_text_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取文本特征"""
        features = {}
        content = data.get('content', '')
        
        if not content:
            return features
        
        # 基础长度特征
        features['char_count'] = len(content)
        features['word_count'] = len(self.text_processor.segment_text(content))
        features['sentence_count'] = len(re.split(r'[。！？]', content))
        features['paragraph_count'] = len([p for p in content.split('\n') if p.strip()])
        
        # 复杂度特征
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['char_count'] / features['sentence_count']
            features['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
            features['avg_words_per_sentence'] = 0
        
        # 词汇多样性
        words = self.text_processor.segment_text(content)
        unique_words = set(words)
        features['vocabulary_diversity'] = len(unique_words) / max(len(words), 1)
        features['unique_word_count'] = len(unique_words)
        
        # 数字和符号统计
        features['number_count'] = len(self.patterns['number_with_unit'].findall(content))
        features['percentage_count'] = len(self.patterns['percentage'].findall(content))
        features['currency_count'] = len(self.patterns['currency'].findall(content))
        features['stock_code_count'] = len(self.patterns['stock_code'].findall(content))
        
        # 情感特征
        positive_matches = len(self.patterns['positive_words'].findall(content))
        negative_matches = len(self.patterns['negative_words'].findall(content))
        uncertain_matches = len(self.patterns['uncertain_words'].findall(content))
        
        features['positive_word_count'] = positive_matches
        features['negative_word_count'] = negative_matches
        features['uncertain_word_count'] = uncertain_matches
        
        total_sentiment_words = positive_matches + negative_matches + uncertain_matches
        if total_sentiment_words > 0:
            features['positive_ratio'] = positive_matches / total_sentiment_words
            features['negative_ratio'] = negative_matches / total_sentiment_words
            features['uncertain_ratio'] = uncertain_matches / total_sentiment_words
            features['sentiment_polarity'] = (positive_matches - negative_matches) / total_sentiment_words
        else:
            features['positive_ratio'] = 0
            features['negative_ratio'] = 0
            features['uncertain_ratio'] = 0
            features['sentiment_polarity'] = 0
        
        # 可读性特征
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            # 简化的可读性指数
            features['readability_score'] = 1.0 / (1.0 + features['avg_sentence_length'] / 20.0)
        else:
            features['readability_score'] = 0
        
        # 关键词特征
        if self.config['text_features']['extract_keyword_features']:
            keywords = self.text_processor.extract_keywords(content, topk=10)
            features['top_keyword_weight'] = keywords[0][1] if keywords else 0
            features['keyword_concentration'] = sum(weight for _, weight in keywords[:3]) if keywords else 0
        
        return features
    
    def _extract_numerical_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取数值特征"""
        features = {}
        
        # 预测数值特征
        numerical_fields = ['target_price', 'current_price', 'growth_rate', 'pe_ratio', 'pb_ratio', 'roe', 'market_cap']
        
        for field in numerical_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    features[field] = value
                    
                    # 数值变换特征
                    if value > 0:
                        features[f'{field}_log'] = math.log(value + 1)
                        features[f'{field}_sqrt'] = math.sqrt(value)
                    
                    # 标准化特征（简化版本，实际需要训练集统计信息）
                    if field == 'pe_ratio' and 0 < value < 200:
                        features[f'{field}_normalized'] = value / 50.0  # 假设合理PE范围
                    elif field == 'growth_rate':
                        features[f'{field}_normalized'] = (value + 50) / 100.0  # 标准化到0-1
                    
                except (ValueError, TypeError):
                    features[field] = 0
            else:
                features[field] = 0
        
        # 比率特征
        if 'target_price' in features and 'current_price' in features and features['current_price'] > 0:
            features['price_upside_ratio'] = (features['target_price'] - features['current_price']) / features['current_price']
        
        if 'pe_ratio' in features and 'growth_rate' in features and features['growth_rate'] > 0:
            features['peg_ratio'] = features['pe_ratio'] / features['growth_rate']
        
        # 从文本中提取的数值特征
        content = data.get('content', '')
        if content:
            # 统计数字出现频率
            numbers = re.findall(r'\d+(?:\.\d+)?', content)
            if numbers:
                number_values = [float(n) for n in numbers if float(n) < 1e10]  # 过滤超大数字
                if number_values:
                    features['text_number_count'] = len(number_values)
                    features['text_number_mean'] = np.mean(number_values)
                    features['text_number_std'] = np.std(number_values)
                    features['text_number_max'] = np.max(number_values)
                    features['text_number_min'] = np.min(number_values)
        
        return features
    
    def _extract_temporal_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取时间特征"""
        features = {}
        
        # 解析发布时间
        publish_date = None
        if 'publish_date' in data:
            if isinstance(data['publish_date'], datetime):
                publish_date = data['publish_date']
            elif isinstance(data['publish_date'], str):
                try:
                    publish_date = datetime.fromisoformat(data['publish_date'].replace('Z', '+00:00'))
                except:
                    try:
                        publish_date = datetime.strptime(data['publish_date'], '%Y-%m-%d')
                    except:
                        pass
        
        if publish_date:
            # 基础时间特征
            features['publish_year'] = publish_date.year
            features['publish_month'] = publish_date.month
            features['publish_day'] = publish_date.day
            features['publish_weekday'] = publish_date.weekday()  # 0=Monday, 6=Sunday
            features['publish_hour'] = publish_date.hour
            
            # 市场时间特征
            features['is_trading_day'] = 1 if publish_date.weekday() < 5 else 0  # 简化版本
            features['is_market_hours'] = 1 if 9 <= publish_date.hour <= 15 else 0  # 简化的交易时间
            features['is_after_market'] = 1 if publish_date.hour > 15 else 0
            
            # 周期性特征
            features['month_sin'] = math.sin(2 * math.pi * publish_date.month / 12)
            features['month_cos'] = math.cos(2 * math.pi * publish_date.month / 12)
            features['day_sin'] = math.sin(2 * math.pi * publish_date.day / 31)
            features['day_cos'] = math.cos(2 * math.pi * publish_date.day / 31)
            features['hour_sin'] = math.sin(2 * math.pi * publish_date.hour / 24)
            features['hour_cos'] = math.cos(2 * math.pi * publish_date.hour / 24)
            
            # 相对时间特征
            now = datetime.now()
            time_diff = (now - publish_date).total_seconds()
            features['days_since_publish'] = time_diff / (24 * 3600)
            features['is_recent'] = 1 if time_diff < 7 * 24 * 3600 else 0  # 7天内
            features['is_stale'] = 1 if time_diff > 30 * 24 * 3600 else 0  # 30天外
            
            # 季度特征
            quarter = (publish_date.month - 1) // 3 + 1
            features['quarter'] = quarter
            features['is_quarter_end'] = 1 if publish_date.month in [3, 6, 9, 12] else 0
        
        return features
    
    def _extract_metadata_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取元数据特征"""
        features = {}
        
        # 作者特征
        author = data.get('author', '')
        if author:
            features['author_name_length'] = len(author)
            features['has_author'] = 1
            
            # 作者历史特征（简化版本）
            author_history = self.feature_cache['author_history'][author]
            features['author_report_count'] = len(author_history)
            
            # 多作者特征
            if isinstance(data.get('analysts'), list):
                features['analyst_count'] = len(data['analysts'])
                features['is_multi_analyst'] = 1 if len(data['analysts']) > 1 else 0
            else:
                features['analyst_count'] = 1 if author else 0
                features['is_multi_analyst'] = 0
        else:
            features['has_author'] = 0
            features['author_name_length'] = 0
            features['author_report_count'] = 0
            features['analyst_count'] = 0
            features['is_multi_analyst'] = 0
        
        # 股票特征
        stocks = data.get('stocks', [])
        if isinstance(stocks, list):
            features['stock_count'] = len(stocks)
            features['is_multi_stock'] = 1 if len(stocks) > 1 else 0
            
            # 交易所分布
            exchanges = [stock.split('.')[1] if '.' in stock else 'unknown' for stock in stocks]
            exchange_counter = Counter(exchanges)
            features['has_sz_stock'] = 1 if 'SZ' in exchange_counter else 0
            features['has_sh_stock'] = 1 if 'SH' in exchange_counter else 0
            features['has_bj_stock'] = 1 if 'BJ' in exchange_counter else 0
        else:
            features['stock_count'] = 0
            features['is_multi_stock'] = 0
            features['has_sz_stock'] = 0
            features['has_sh_stock'] = 0
            features['has_bj_stock'] = 0
        
        # 机构特征
        institution = data.get('institution', '')
        if institution:
            features['has_institution'] = 1
            features['institution_name_length'] = len(institution)
        else:
            features['has_institution'] = 0
            features['institution_name_length'] = 0
        
        # 评级特征
        rating = data.get('rating', '')
        if rating:
            features['has_rating'] = 1
            rating_mapping = {'买入': 1, '增持': 0.75, '中性': 0.5, '减持': 0.25, '卖出': 0}
            features['rating_score'] = rating_mapping.get(rating, 0.5)
        else:
            features['has_rating'] = 0
            features['rating_score'] = 0.5
        
        # 报告类型特征
        report_type = data.get('report_type', '')
        type_mapping = {'公司研究': 1, '行业研究': 2, '策略研究': 3, '债券研究': 4}
        features['report_type_code'] = type_mapping.get(report_type, 0)
        
        return features
    
    def _extract_semantic_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取语义特征"""
        features = {}
        content = data.get('content', '')
        
        if not content:
            return features
        
        # 主题相关特征（简化版本）
        # 在实际应用中可以使用LDA或其他主题模型
        
        # 关键词主题分布
        keywords = self.text_processor.extract_keywords(content, topk=20)
        keyword_weights = [weight for _, weight in keywords]
        
        if keyword_weights:
            features['keyword_weight_mean'] = np.mean(keyword_weights)
            features['keyword_weight_std'] = np.std(keyword_weights)
            features['keyword_weight_max'] = np.max(keyword_weights)
            features['top_3_keyword_weight'] = sum(keyword_weights[:3])
        
        # 实体特征（基于规则的简化版本）
        companies = len(self.patterns['company_name'].findall(content))
        dates = len(self.patterns['date'].findall(content))
        quarters = len(self.patterns['quarter'].findall(content))
        
        features['company_mention_count'] = companies
        features['date_mention_count'] = dates
        features['quarter_mention_count'] = quarters
        features['temporal_entity_ratio'] = (dates + quarters) / max(len(content.split()), 1)
        
        # 话题一致性（通过关键词重复度衡量）
        word_counts = Counter(self.text_processor.segment_text(content))
        top_words = word_counts.most_common(10)
        if top_words:
            total_words = sum(word_counts.values())
            features['topic_concentration'] = sum(count for _, count in top_words) / total_words
            features['most_frequent_word_ratio'] = top_words[0][1] / total_words
        else:
            features['topic_concentration'] = 0
            features['most_frequent_word_ratio'] = 0
        
        return features
    
    def _extract_behavioral_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取行为特征"""
        features = {}
        
        # 发布模式特征
        author = data.get('author', '')
        publish_date = data.get('publish_date')
        
        if author and publish_date:
            # 更新作者历史
            if isinstance(publish_date, str):
                try:
                    publish_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                except:
                    pass
            
            if isinstance(publish_date, datetime):
                self.feature_cache['author_history'][author].append({
                    'date': publish_date,
                    'data': data
                })
                
                # 计算发布频率特征
                author_history = self.feature_cache['author_history'][author]
                recent_reports = [r for r in author_history 
                                if (publish_date - r['date']).days <= self.config['behavioral_features']['history_window_days']]
                
                features['recent_report_count'] = len(recent_reports)
                
                if len(recent_reports) > 1:
                    # 发布间隔特征
                    intervals = []
                    sorted_reports = sorted(recent_reports, key=lambda x: x['date'])
                    for i in range(1, len(sorted_reports)):
                        interval = (sorted_reports[i]['date'] - sorted_reports[i-1]['date']).days
                        intervals.append(interval)
                    
                    if intervals:
                        features['avg_publish_interval'] = np.mean(intervals)
                        features['publish_interval_std'] = np.std(intervals)
                        features['min_publish_interval'] = np.min(intervals)
                        features['is_frequent_publisher'] = 1 if np.mean(intervals) < 7 else 0
                
                # 内容变化特征
                if len(recent_reports) > 1:
                    current_content = data.get('content', '')
                    previous_content = recent_reports[-2]['data'].get('content', '')
                    
                    if current_content and previous_content:
                        # 计算内容相似度（简化版本）
                        current_words = set(self.text_processor.segment_text(current_content))
                        previous_words = set(self.text_processor.segment_text(previous_content))
                        
                        if current_words or previous_words:
                            jaccard_similarity = len(current_words & previous_words) / len(current_words | previous_words)
                            features['content_similarity_to_previous'] = jaccard_similarity
                            features['is_repetitive_content'] = 1 if jaccard_similarity > 0.8 else 0
        
        return features
    
    def _engineer_features(self, features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """特征工程 - 创建组合特征和派生特征"""
        engineered = {}
        
        # 获取各类特征
        text_features = features.get('text_features', {})
        numerical_features = features.get('numerical_features', {})
        temporal_features = features.get('temporal_features', {})
        metadata_features = features.get('metadata_features', {})
        
        # 交互特征
        if 'char_count' in text_features and 'stock_count' in metadata_features:
            engineered['content_per_stock'] = text_features['char_count'] / max(metadata_features['stock_count'], 1)
        
        if 'positive_ratio' in text_features and 'rating_score' in metadata_features:
            engineered['sentiment_rating_alignment'] = abs(text_features['positive_ratio'] - metadata_features['rating_score'])
        
        if 'target_price' in numerical_features and 'current_price' in numerical_features:
            if numerical_features['current_price'] > 0:
                engineered['price_change_magnitude'] = abs(numerical_features['target_price'] - numerical_features['current_price'])
                engineered['relative_price_change'] = engineered['price_change_magnitude'] / numerical_features['current_price']
        
        # 比率特征
        if 'word_count' in text_features and 'sentence_count' in text_features:
            if text_features['sentence_count'] > 0:
                engineered['words_per_sentence'] = text_features['word_count'] / text_features['sentence_count']
        
        if 'positive_word_count' in text_features and 'negative_word_count' in text_features:
            total_sentiment = text_features['positive_word_count'] + text_features['negative_word_count']
            if total_sentiment > 0:
                engineered['sentiment_intensity'] = total_sentiment / text_features.get('word_count', 1)
        
        # 复杂度综合特征
        complexity_factors = []
        if 'vocabulary_diversity' in text_features:
            complexity_factors.append(text_features['vocabulary_diversity'])
        if 'avg_sentence_length' in text_features:
            complexity_factors.append(min(text_features['avg_sentence_length'] / 50, 1))  # 标准化
        if 'readability_score' in text_features:
            complexity_factors.append(1 - text_features['readability_score'])  # 反向，复杂度高
        
        if complexity_factors:
            engineered['text_complexity_score'] = np.mean(complexity_factors)
        
        # 时效性综合特征
        timeliness_factors = []
        if 'is_recent' in temporal_features:
            timeliness_factors.append(temporal_features['is_recent'])
        if 'is_trading_day' in temporal_features:
            timeliness_factors.append(temporal_features['is_trading_day'])
        if 'is_market_hours' in temporal_features:
            timeliness_factors.append(temporal_features['is_market_hours'])
        
        if timeliness_factors:
            engineered['timeliness_score'] = np.mean(timeliness_factors)
        
        return engineered
    
    def batch_extract_features(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量提取特征
        
        Args:
            data_list: 数据列表
            
        Returns:
            List[Dict[str, Any]]: 特征列表
        """
        results = []
        
        for i, data in enumerate(data_list):
            try:
                features = self.extract_features(data)
                features['index'] = i
                results.append(features)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"已提取 {i + 1}/{len(data_list)} 份数据的特征")
                    
            except Exception as e:
                logger.error(f"批量特征提取第 {i} 份数据失败: {e}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'feature_metadata': {
                        'extraction_time': 0,
                        'feature_count': 0,
                        'error': str(e)
                    }
                })
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        # 这个方法返回所有可能的特征名称
        # 在实际应用中，可以通过分析已提取的特征来动态生成
        
        text_features = [
            'char_count', 'word_count', 'sentence_count', 'paragraph_count',
            'avg_sentence_length', 'avg_words_per_sentence', 'vocabulary_diversity',
            'unique_word_count', 'number_count', 'percentage_count', 'currency_count',
            'stock_code_count', 'positive_word_count', 'negative_word_count',
            'uncertain_word_count', 'positive_ratio', 'negative_ratio',
            'uncertain_ratio', 'sentiment_polarity', 'readability_score'
        ]
        
        numerical_features = [
            'target_price', 'current_price', 'growth_rate', 'pe_ratio', 'pb_ratio',
            'roe', 'market_cap', 'price_upside_ratio', 'peg_ratio'
        ]
        
        temporal_features = [
            'publish_year', 'publish_month', 'publish_day', 'publish_weekday',
            'publish_hour', 'is_trading_day', 'is_market_hours', 'is_after_market',
            'days_since_publish', 'is_recent', 'is_stale', 'quarter', 'is_quarter_end'
        ]
        
        metadata_features = [
            'author_name_length', 'has_author', 'author_report_count',
            'analyst_count', 'is_multi_analyst', 'stock_count', 'is_multi_stock',
            'has_sz_stock', 'has_sh_stock', 'has_bj_stock', 'has_institution',
            'institution_name_length', 'has_rating', 'rating_score', 'report_type_code'
        ]
        
        return text_features + numerical_features + temporal_features + metadata_features
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        获取特征提取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total = self.extraction_stats['total_extracted']
        
        return {
            'total_extracted': total,
            'average_feature_count': self.extraction_stats['feature_dimensions'],
            'average_processing_time': self.extraction_stats['processing_time'] / max(total, 1),
            'total_processing_time': self.extraction_stats['processing_time'],
            'cache_size': {
                'author_history': len(self.feature_cache['author_history']),
                'stock_history': len(self.feature_cache['stock_history'])
            },
            'config': self.config
        }


# 全局特征提取器实例
_global_feature_extractor = None


def get_feature_extractor() -> FeatureExtractor:
    """
    获取全局特征提取器实例
    
    Returns:
        FeatureExtractor: 提取器实例
    """
    global _global_feature_extractor
    
    if _global_feature_extractor is None:
        _global_feature_extractor = FeatureExtractor()
    
    return _global_feature_extractor


if __name__ == "__main__":
    # 使用示例
    extractor = FeatureExtractor()
    
    # 测试数据
    test_data = {
        'title': '平安银行(000001.SZ)投资价值分析',
        'content': '平安银行作为国内领先的股份制银行，在零售业务方面表现突出。预计2024年营收增长15.5%，净利润增长12.3%。目标价15.50元，当前价12.30元，建议买入。' * 10,
        'author': '张三',
        'analysts': ['张三', '李四'],
        'stocks': ['000001.SZ'],
        'institution': '某证券研究所',
        'rating': '买入',
        'target_price': 15.50,
        'current_price': 12.30,
        'growth_rate': 15.5,
        'pe_ratio': 8.5,
        'publish_date': '2024-01-15T10:30:00',
        'report_type': '公司研究'
    }
    
    features = extractor.extract_features(test_data)
    
    print("特征提取结果:")
    print(f"总特征数: {features['feature_metadata']['feature_count']}")
    print(f"提取时间: {features['feature_metadata']['extraction_time']:.3f}秒")
    
    for category, feature_dict in features.items():
        if isinstance(feature_dict, dict) and category != 'feature_metadata':
            print(f"\n{category} ({len(feature_dict)}个特征):")
            for name, value in list(feature_dict.items())[:5]:  # 只显示前5个
                print(f"  {name}: {value}")
            if len(feature_dict) > 5:
                print(f"  ... 还有{len(feature_dict)-5}个特征")
    
    # 获取统计信息
    stats = extractor.get_extraction_statistics()
    print(f"\n特征提取统计: {stats}") 
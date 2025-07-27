"""
语义异常检测器
使用Qwen模型进行深度语义分析，检测逻辑矛盾、历史观点偏离、信息源异常等
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
import logging
import re
import difflib
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class SemanticAnomalyDetector:
    """
    语义异常检测器
    
    使用Qwen模型进行深度语义分析，检测：
    1. 逻辑矛盾 - 文本内部逻辑不一致
    2. 历史观点偏离 - 与历史观点的无合理解释转变
    3. 信息源异常 - 不可验证或异常详细的数据来源
    4. 语义一致性 - 表达方式与内容的一致性
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 配置参数
        text_processor: 文本处理器
        qwen_model: Qwen模型实例
        historical_views: 历史观点记录
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化语义异常检测器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        self.file_manager = get_file_manager()
        
        # Qwen模型实例（延迟加载）
        self.qwen_model = None
        self.model_available = False
        
        # 历史观点记录
        self.historical_views = {
            'stock_views': defaultdict(list),    # 股票观点历史
            'industry_views': defaultdict(list), # 行业观点历史
            'market_views': [],                  # 整体市场观点
            'author_views': defaultdict(list),   # 作者观点历史
            'view_count': 0,
            'last_update': None
        }
        
        # 异常模式库
        self.anomaly_patterns = {
            'contradiction_keywords': [
                '但是', '然而', '不过', '相反', '相对地', '另一方面',
                '与此相反', '矛盾的是', '截然不同', '完全相反'
            ],
            'uncertainty_keywords': [
                '可能', '或许', '大概', '估计', '预计', '推测',
                '据说', '传言', '听说', '不确定', '尚不清楚'
            ],
            'source_patterns': [
                r'据.*消息',
                r'某.*人士',
                r'内部.*透露',
                r'匿名.*表示',
                r'不愿.*透露.*姓名',
                r'知情.*人士',
                r'可靠.*消息.*源'
            ]
        }
        
        self.is_trained = False
        
        # 加载历史数据
        self._load_historical_data()
        
        logger.info("语义异常检测器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('semantic_anomaly', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'logical_contradiction': {
                'contradiction_threshold': 0.7,    # 矛盾检测阈值
                'semantic_similarity_threshold': 0.3,  # 语义相似度阈值
                'max_text_segments': 20           # 最大文本分段数
            },
            'historical_deviation': {
                'view_change_threshold': 0.6,     # 观点变化阈值
                'similarity_threshold': 0.4,      # 相似度阈值
                'time_decay_days': 180,           # 时间衰减天数
                'min_historical_views': 5         # 最少历史观点数
            },
            'information_source': {
                'source_reliability_threshold': 0.3,  # 信息源可靠性阈值
                'detail_level_threshold': 0.8,        # 详细程度阈值
                'verification_score_threshold': 0.5   # 验证分数阈值
            },
            'semantic_consistency': {
                'consistency_threshold': 0.7,     # 一致性阈值
                'style_deviation_threshold': 0.5, # 风格偏离阈值
                'emotion_consistency_threshold': 0.6  # 情感一致性阈值
            },
            'model_fallback': {
                'enable_simple_detection': True,  # 启用简单检测作为备选
                'max_model_retry': 3,             # 模型重试次数
                'timeout_seconds': 30             # 超时时间
            },
            'anomaly_score_weights': {
                'logical_contradiction': 0.3,
                'historical_deviation': 0.3,
                'information_source': 0.2,
                'semantic_consistency': 0.2
            }
        }
    
    def _load_historical_data(self):
        """加载历史观点数据"""
        try:
            data_path = Path("data/models/semantic_history.pkl")
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.historical_views.update(data.get('views', {}))
                
                if self.historical_views['view_count'] >= 10:
                    self.is_trained = True
                    logger.info(f"已加载 {self.historical_views['view_count']} 份历史观点数据")
                
        except Exception as e:
            logger.warning(f"历史数据加载失败: {e}")
    
    def _save_historical_data(self):
        """保存历史观点数据"""
        try:
            data_path = Path("data/models/semantic_history.pkl")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'views': self.historical_views
            }
            
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug("历史观点数据已保存")
            
        except Exception as e:
            logger.error(f"历史数据保存失败: {e}")
    
    def _get_qwen_model(self):
        """获取Qwen模型实例"""
        if self.qwen_model is None:
            try:
                from ..models.qwen_wrapper import get_qwen_model
                self.qwen_model = get_qwen_model()
                self.model_available = True
                logger.info("Qwen模型加载成功")
            except Exception as e:
                logger.warning(f"Qwen模型加载失败，将使用简单检测方法: {e}")
                self.model_available = False
        
        return self.qwen_model if self.model_available else None
    
    def add_historical_view(self, view_data: Dict[str, Any]):
        """
        添加历史观点记录
        
        Args:
            view_data: 观点数据，包含股票、观点、时间等信息
        """
        try:
            stocks = view_data.get('stocks', [])
            industries = view_data.get('industries', [])
            author = view_data.get('author', 'unknown')
            content = view_data.get('content', '')
            date = view_data.get('date', datetime.now())
            sentiment = view_data.get('sentiment', 'neutral')
            
            # 提取观点摘要
            view_summary = self._extract_view_summary(content)
            
            view_record = {
                'content': content,
                'summary': view_summary,
                'sentiment': sentiment,
                'date': date,
                'author': author,
                'confidence': view_data.get('confidence', 0.5)
            }
            
            # 按股票记录观点
            for stock in stocks:
                self.historical_views['stock_views'][stock].append(view_record.copy())
            
            # 按行业记录观点
            for industry in industries:
                self.historical_views['industry_views'][industry].append(view_record.copy())
            
            # 按作者记录观点
            self.historical_views['author_views'][author].append(view_record.copy())
            
            # 整体市场观点
            if not stocks and not industries:  # 整体市场观点
                self.historical_views['market_views'].append(view_record)
            
            self.historical_views['view_count'] += 1
            self.historical_views['last_update'] = datetime.now()
            
            # 限制历史记录数量
            for view_list in [self.historical_views['stock_views'], 
                            self.historical_views['industry_views'], 
                            self.historical_views['author_views']]:
                for key in view_list:
                    if len(view_list[key]) > 50:
                        view_list[key] = view_list[key][-50:]
            
            if len(self.historical_views['market_views']) > 100:
                self.historical_views['market_views'] = self.historical_views['market_views'][-100:]
            
            # 检查是否可以开始训练
            if self.historical_views['view_count'] >= 10:
                self.is_trained = True
            
            # 保存数据
            self._save_historical_data()
            
            logger.debug(f"添加历史观点: {author} - {len(stocks)}只股票")
            
        except Exception as e:
            logger.error(f"添加历史观点失败: {e}")
            raise
    
    def _extract_view_summary(self, content: str) -> str:
        """提取观点摘要"""
        # 使用关键词提取作为简单摘要
        keywords = self.text_processor.extract_keywords(content, topk=5)
        summary_words = [kw[0] for kw in keywords]
        
        # 提取重要句子
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # 选择包含关键词最多的句子作为摘要
        if sentences and summary_words:
            sentence_scores = []
            for sentence in sentences:
                score = sum(1 for word in summary_words if word in sentence)
                sentence_scores.append((sentence, score))
            
            best_sentence = max(sentence_scores, key=lambda x: x[1])[0]
            return best_sentence[:100] + ('...' if len(best_sentence) > 100 else '')
        
        return content[:100] + ('...' if len(content) > 100 else '')
    
    def detect_semantic_anomalies(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测语义异常
        
        Args:
            report_data: 报告数据，包含内容、股票、作者等信息
            
        Returns:
            Dict[str, Any]: 语义异常检测结果
        """
        try:
            results = {
                'logical_contradiction': self.detect_logical_contradiction(report_data.get('content', '')),
                'historical_deviation': self.detect_historical_deviation(report_data),
                'information_source_anomaly': self.detect_information_source_anomaly(report_data.get('content', '')),
                'semantic_consistency': self.detect_semantic_consistency(report_data.get('content', '')),
                'overall_score': 0.0,
                'anomaly_level': 'NORMAL',
                'details': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 计算综合异常分数
            weights = self.config['anomaly_score_weights']
            anomaly_scores = [
                results['logical_contradiction']['score'] * weights['logical_contradiction'],
                results['historical_deviation']['score'] * weights['historical_deviation'],
                results['information_source_anomaly']['score'] * weights['information_source'],
                results['semantic_consistency']['score'] * weights['semantic_consistency']
            ]
            
            results['overall_score'] = sum(anomaly_scores)
            
            # 确定异常等级
            results['anomaly_level'] = self._determine_anomaly_level(results['overall_score'])
            
            # 详细信息
            results['details'] = {
                'view_count': self.historical_views['view_count'],
                'model_available': self.model_available,
                'is_trained': self.is_trained,
                'analysis_methods': self._get_analysis_methods_used()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"语义异常检测失败: {e}")
            raise
    
    def detect_logical_contradiction(self, text: str) -> Dict[str, Any]:
        """检测逻辑矛盾"""
        if not text:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无文本内容"
            }
        
        # 尝试使用Qwen模型进行深度分析
        qwen_result = self._qwen_detect_contradiction(text)
        if qwen_result:
            return qwen_result
        
        # 备选：简单规则检测
        return self._simple_detect_contradiction(text)
    
    def _qwen_detect_contradiction(self, text: str) -> Optional[Dict[str, Any]]:
        """使用Qwen模型检测逻辑矛盾"""
        model = self._get_qwen_model()
        if not model:
            return None
        
        try:
            prompt = f"""
请分析以下研报文本中是否存在逻辑矛盾。分析以下几个方面：
1. 前后观点是否一致
2. 结论与论证是否匹配
3. 数据与分析是否矛盾
4. 建议与风险评估是否一致

文本内容：
{text[:1500]}

请返回分析结果，格式为：
矛盾程度: [无/轻微/中等/严重]
矛盾类型: [观点矛盾/数据矛盾/逻辑矛盾/其他]
具体问题: [具体描述]
"""
            
            response = model.generate_text(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True
            )
            
            return self._parse_qwen_contradiction_result(response, text)
            
        except Exception as e:
            logger.warning(f"Qwen矛盾检测失败: {e}")
            return None
    
    def _parse_qwen_contradiction_result(self, response: str, text: str) -> Dict[str, Any]:
        """解析Qwen矛盾检测结果"""
        response = response.lower()
        
        # 提取矛盾程度
        if '严重' in response:
            contradiction_level = 0.9
        elif '中等' in response:
            contradiction_level = 0.6
        elif '轻微' in response:
            contradiction_level = 0.3
        else:
            contradiction_level = 0.0
        
        # 提取矛盾类型
        contradiction_types = []
        if '观点矛盾' in response:
            contradiction_types.append('观点矛盾')
        if '数据矛盾' in response:
            contradiction_types.append('数据矛盾')
        if '逻辑矛盾' in response:
            contradiction_types.append('逻辑矛盾')
        
        is_anomaly = contradiction_level > self.config['logical_contradiction']['contradiction_threshold']
        
        return {
            'is_anomaly': is_anomaly,
            'score': contradiction_level,
            'contradiction_types': contradiction_types,
            'model_response': response[:200],
            'description': f"逻辑矛盾程度{contradiction_level:.2f}，类型：{','.join(contradiction_types) if contradiction_types else '无'}"
        }
    
    def _simple_detect_contradiction(self, text: str) -> Dict[str, Any]:
        """简单规则检测逻辑矛盾"""
        # 分割文本为句子
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        contradiction_score = 0.0
        contradiction_patterns = []
        
        # 检查矛盾关键词
        contradiction_keywords = self.anomaly_patterns['contradiction_keywords']
        for sentence in sentences:
            for keyword in contradiction_keywords:
                if keyword in sentence:
                    contradiction_score += 0.1
                    contradiction_patterns.append(f"发现矛盾词：{keyword}")
        
        # 检查情感极性转换
        positive_words = ['利好', '上涨', '增长', '优秀', '强劲']
        negative_words = ['利空', '下跌', '下降', '风险', '压力']
        
        has_positive = any(word in text for word in positive_words)
        has_negative = any(word in text for word in negative_words)
        
        if has_positive and has_negative:
            contradiction_score += 0.2
            contradiction_patterns.append("同时存在正面和负面观点")
        
        # 检查数值矛盾
        numbers = re.findall(r'\d+\.?\d*%?', text)
        if len(numbers) >= 4:
            # 简单检查是否有明显不合理的数值组合
            contradiction_score += 0.1
            contradiction_patterns.append("存在多个数值，可能存在矛盾")
        
        # 标准化分数
        contradiction_score = min(contradiction_score, 1.0)
        is_anomaly = contradiction_score > self.config['logical_contradiction']['contradiction_threshold']
        
        return {
            'is_anomaly': is_anomaly,
            'score': contradiction_score,
            'contradiction_patterns': contradiction_patterns,
            'method': 'simple_rule_based',
            'description': f"简单规则检测矛盾分数{contradiction_score:.2f}"
        }
    
    def detect_historical_deviation(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测历史观点偏离"""
        if not self.is_trained:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "历史数据不足"
            }
        
        stocks = report_data.get('stocks', [])
        author = report_data.get('author', 'unknown')
        current_content = report_data.get('content', '')
        current_sentiment = report_data.get('sentiment', 'neutral')
        
        if not current_content:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无内容信息"
            }
        
        deviation_scores = []
        deviations = []
        
        # 检查股票观点偏离
        for stock in stocks:
            if stock in self.historical_views['stock_views']:
                deviation = self._calculate_view_deviation(
                    current_content, 
                    current_sentiment,
                    self.historical_views['stock_views'][stock],
                    f"股票{stock}"
                )
                if deviation['score'] > 0:
                    deviation_scores.append(deviation['score'])
                    deviations.append(deviation)
        
        # 检查作者观点偏离
        if author in self.historical_views['author_views']:
            deviation = self._calculate_view_deviation(
                current_content,
                current_sentiment, 
                self.historical_views['author_views'][author],
                f"作者{author}"
            )
            if deviation['score'] > 0:
                deviation_scores.append(deviation['score'])
                deviations.append(deviation)
        
        # 综合偏离分数
        overall_score = max(deviation_scores) if deviation_scores else 0.0
        is_anomaly = overall_score > self.config['historical_deviation']['view_change_threshold']
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'deviations_count': len(deviations),
            'deviations': deviations[:3],  # 只显示前3个
            'description': f"发现{len(deviations)}个观点偏离，最高分数{overall_score:.2f}"
        }
    
    def _calculate_view_deviation(self, current_content: str, current_sentiment: str,
                                historical_views: List[Dict], context: str) -> Dict[str, Any]:
        """计算观点偏离度"""
        if not historical_views:
            return {'score': 0.0}
        
        # 获取最近的历史观点
        time_threshold = datetime.now() - timedelta(days=self.config['historical_deviation']['time_decay_days'])
        recent_views = [v for v in historical_views if v['date'] >= time_threshold]
        
        if not recent_views:
            return {'score': 0.0}
        
        # 计算内容相似度
        content_similarities = []
        sentiment_deviations = []
        
        current_summary = self._extract_view_summary(current_content)
        
        for view in recent_views[-5:]:  # 最近5个观点
            # 文本相似度
            similarity = self._calculate_text_similarity(current_summary, view['summary'])
            content_similarities.append(similarity)
            
            # 情感偏离
            sentiment_deviation = self._calculate_sentiment_deviation(current_sentiment, view['sentiment'])
            sentiment_deviations.append(sentiment_deviation)
        
        # 计算偏离分数
        avg_similarity = np.mean(content_similarities)
        avg_sentiment_deviation = np.mean(sentiment_deviations)
        
        # 如果相似度很低且情感有显著变化，则认为有偏离
        similarity_threshold = self.config['historical_deviation']['similarity_threshold']
        
        if avg_similarity < similarity_threshold and avg_sentiment_deviation > 0.5:
            deviation_score = (1 - avg_similarity) * (avg_sentiment_deviation + 0.5)
        else:
            deviation_score = 0.0
        
        return {
            'score': min(deviation_score, 1.0),
            'context': context,
            'similarity': avg_similarity,
            'sentiment_deviation': avg_sentiment_deviation,
            'historical_views_count': len(recent_views)
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 使用Qwen模型计算嵌入相似度
            model = self._get_qwen_model()
            if model:
                embeddings = model.get_text_embeddings([text1, text2])
                if len(embeddings) == 2:
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    return float(similarity)
        except Exception as e:
            logger.debug(f"嵌入相似度计算失败: {e}")
        
        # 备选：简单字符相似度
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_sentiment_deviation(self, sentiment1: str, sentiment2: str) -> float:
        """计算情感偏离度"""
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        s1 = sentiment_mapping.get(sentiment1, 0)
        s2 = sentiment_mapping.get(sentiment2, 0)
        
        return abs(s1 - s2) / 2.0  # 归一化到0-1
    
    def detect_information_source_anomaly(self, text: str) -> Dict[str, Any]:
        """检测信息源异常"""
        if not text:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无文本内容"
            }
        
        # 尝试使用Qwen模型分析
        qwen_result = self._qwen_detect_source_anomaly(text)
        if qwen_result:
            return qwen_result
        
        # 备选：简单规则检测
        return self._simple_detect_source_anomaly(text)
    
    def _qwen_detect_source_anomaly(self, text: str) -> Optional[Dict[str, Any]]:
        """使用Qwen模型检测信息源异常"""
        model = self._get_qwen_model()
        if not model:
            return None
        
        try:
            prompt = f"""
请分析以下研报文本中的信息来源是否可靠和合理：
1. 是否引用了具体可验证的数据源
2. 是否存在模糊不清的信息来源
3. 是否有过于详细但无法验证的信息
4. 信息的权威性和可信度如何

文本内容：
{text[:1000]}

请评估信息源的可靠性等级：[高/中/低/异常]
并指出具体问题。
"""
            
            response = model.generate_text(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
            
            return self._parse_qwen_source_result(response, text)
            
        except Exception as e:
            logger.warning(f"Qwen信息源检测失败: {e}")
            return None
    
    def _parse_qwen_source_result(self, response: str, text: str) -> Dict[str, Any]:
        """解析Qwen信息源检测结果"""
        response = response.lower()
        
        # 提取可靠性等级
        if '异常' in response:
            reliability_score = 0.9
        elif '低' in response:
            reliability_score = 0.6
        elif '中' in response:
            reliability_score = 0.3
        else:
            reliability_score = 0.1
        
        is_anomaly = reliability_score > self.config['information_source']['source_reliability_threshold']
        
        return {
            'is_anomaly': is_anomaly,
            'score': reliability_score,
            'model_response': response[:150],
            'description': f"信息源异常分数{reliability_score:.2f}"
        }
    
    def _simple_detect_source_anomaly(self, text: str) -> Dict[str, Any]:
        """简单规则检测信息源异常"""
        anomaly_score = 0.0
        anomaly_patterns = []
        
        # 检查可疑信息源模式
        source_patterns = self.anomaly_patterns['source_patterns']
        for pattern in source_patterns:
            matches = re.findall(pattern, text)
            if matches:
                anomaly_score += 0.2 * len(matches)
                anomaly_patterns.append(f"可疑信息源: {matches[0]}")
        
        # 检查不确定性词汇
        uncertainty_keywords = self.anomaly_patterns['uncertainty_keywords']
        uncertainty_count = sum(1 for word in uncertainty_keywords if word in text)
        if uncertainty_count > 5:
            anomaly_score += 0.3
            anomaly_patterns.append(f"过多不确定性词汇: {uncertainty_count}个")
        
        # 检查过于详细的数据
        specific_numbers = re.findall(r'\d+\.\d{3,}', text)  # 3位以上小数
        if len(specific_numbers) > 3:
            anomaly_score += 0.2
            anomaly_patterns.append("存在过于精确的数据")
        
        # 标准化分数
        anomaly_score = min(anomaly_score, 1.0)
        threshold = self.config['information_source']['source_reliability_threshold']
        is_anomaly = anomaly_score > threshold
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'anomaly_patterns': anomaly_patterns,
            'method': 'simple_rule_based',
            'description': f"简单规则检测信息源异常分数{anomaly_score:.2f}"
        }
    
    def detect_semantic_consistency(self, text: str) -> Dict[str, Any]:
        """检测语义一致性"""
        if not text:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无文本内容"
            }
        
        # 尝试使用Qwen模型分析
        qwen_result = self._qwen_detect_consistency(text)
        if qwen_result:
            return qwen_result
        
        # 备选：简单规则检测
        return self._simple_detect_consistency(text)
    
    def _qwen_detect_consistency(self, text: str) -> Optional[Dict[str, Any]]:
        """使用Qwen模型检测语义一致性"""
        model = self._get_qwen_model()
        if not model:
            return None
        
        try:
            prompt = f"""
请分析以下研报文本的语义一致性：
1. 表达风格是否一致
2. 情感倾向是否前后一致
3. 论证逻辑是否连贯
4. 用词选择是否适当

文本内容：
{text[:1200]}

请评估一致性程度：[很好/良好/一般/较差/很差]
并指出不一致的地方。
"""
            
            response = model.generate_text(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
            
            return self._parse_qwen_consistency_result(response, text)
            
        except Exception as e:
            logger.warning(f"Qwen一致性检测失败: {e}")
            return None
    
    def _parse_qwen_consistency_result(self, response: str, text: str) -> Dict[str, Any]:
        """解析Qwen一致性检测结果"""
        response = response.lower()
        
        # 提取一致性程度
        if '很差' in response:
            consistency_score = 0.9
        elif '较差' in response:
            consistency_score = 0.7
        elif '一般' in response:
            consistency_score = 0.5
        elif '良好' in response:
            consistency_score = 0.2
        else:
            consistency_score = 0.1
        
        is_anomaly = consistency_score > self.config['semantic_consistency']['consistency_threshold']
        
        return {
            'is_anomaly': is_anomaly,
            'score': consistency_score,
            'model_response': response[:150],
            'description': f"语义一致性异常分数{consistency_score:.2f}"
        }
    
    def _simple_detect_consistency(self, text: str) -> Dict[str, Any]:
        """简单规则检测语义一致性"""
        # 分析文本段落
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "文本段落不足"
            }
        
        inconsistency_score = 0.0
        inconsistency_patterns = []
        
        # 检查情感一致性
        positive_words = ['利好', '上涨', '增长', '优秀', '强劲']
        negative_words = ['利空', '下跌', '下降', '风险', '压力']
        
        paragraph_sentiments = []
        for para in paragraphs:
            pos_count = sum(1 for word in positive_words if word in para)
            neg_count = sum(1 for word in negative_words if word in para)
            
            if pos_count > neg_count:
                paragraph_sentiments.append('positive')
            elif neg_count > pos_count:
                paragraph_sentiments.append('negative')
            else:
                paragraph_sentiments.append('neutral')
        
        # 计算情感变化
        sentiment_changes = 0
        for i in range(1, len(paragraph_sentiments)):
            if paragraph_sentiments[i] != paragraph_sentiments[i-1]:
                sentiment_changes += 1
        
        if sentiment_changes > len(paragraphs) * 0.6:  # 超过60%段落情感发生变化
            inconsistency_score += 0.4
            inconsistency_patterns.append("情感倾向变化频繁")
        
        # 检查用词风格
        formal_words = ['认为', '预计', '分析', '研究', '数据']
        casual_words = ['觉得', '可能', '估计', '大概', '感觉']
        
        formal_count = sum(1 for word in formal_words if word in text)
        casual_count = sum(1 for word in casual_words if word in text)
        
        if formal_count > 0 and casual_count > 0:
            style_inconsistency = min(casual_count / formal_count, 1.0)
            inconsistency_score += style_inconsistency * 0.3
            inconsistency_patterns.append("正式与随意用词混杂")
        
        # 标准化分数
        inconsistency_score = min(inconsistency_score, 1.0)
        threshold = self.config['semantic_consistency']['consistency_threshold']
        is_anomaly = inconsistency_score > threshold
        
        return {
            'is_anomaly': is_anomaly,
            'score': inconsistency_score,
            'inconsistency_patterns': inconsistency_patterns,
            'method': 'simple_rule_based',
            'description': f"简单规则检测一致性异常分数{inconsistency_score:.2f}"
        }
    
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
    
    def _get_analysis_methods_used(self) -> List[str]:
        """获取使用的分析方法"""
        methods = []
        
        if self.model_available:
            methods.append("Qwen深度语义分析")
        
        if self.config['model_fallback']['enable_simple_detection']:
            methods.append("规则基础检测")
        
        if self.is_trained:
            methods.append("历史观点对比")
        
        return methods
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """
        获取语义分析摘要
        
        Returns:
            Dict[str, Any]: 语义分析统计摘要
        """
        return {
            'view_count': self.historical_views['view_count'],
            'tracked_stocks': len(self.historical_views['stock_views']),
            'tracked_industries': len(self.historical_views['industry_views']),
            'tracked_authors': len(self.historical_views['author_views']),
            'model_available': self.model_available,
            'is_trained': self.is_trained,
            'analysis_methods': self._get_analysis_methods_used(),
            'last_update': self.historical_views['last_update'].isoformat() if self.historical_views['last_update'] else None
        }


# 全局检测器实例
_global_semantic_detector = None


def get_semantic_detector() -> SemanticAnomalyDetector:
    """
    获取全局语义异常检测器实例
    
    Returns:
        SemanticAnomalyDetector: 检测器实例
    """
    global _global_semantic_detector
    
    if _global_semantic_detector is None:
        _global_semantic_detector = SemanticAnomalyDetector()
    
    return _global_semantic_detector


if __name__ == "__main__":
    # 使用示例
    detector = SemanticAnomalyDetector()
    
    # 添加历史观点
    sample_views = [
        {
            'stocks': ['000001'],
            'content': '公司基本面良好，业绩稳定增长，建议买入。',
            'sentiment': 'positive',
            'author': '分析师A',
            'date': datetime.now() - timedelta(days=30)
        },
        {
            'stocks': ['000001'],
            'content': '行业前景看好，公司竞争优势明显，维持买入评级。',
            'sentiment': 'positive',
            'author': '分析师A',
            'date': datetime.now() - timedelta(days=15)
        }
    ]
    
    for view in sample_views:
        detector.add_historical_view(view)
    
    # 检测语义异常
    test_report = {
        'content': '公司基本面良好，但是业绩大幅下滑，建议卖出。据某内部人士透露，公司存在重大隐患。',
        'stocks': ['000001'],
        'author': '分析师A',
        'sentiment': 'negative'
    }
    
    result = detector.detect_semantic_anomalies(test_report)
    
    print("语义异常检测结果:")
    print(f"整体异常分数: {result['overall_score']:.3f}")
    print(f"异常等级: {result['anomaly_level']}")
    print(f"逻辑矛盾: {result['logical_contradiction']['is_anomaly']}")
    print(f"历史偏离: {result['historical_deviation']['is_anomaly']}")
    print(f"信息源异常: {result['information_source_anomaly']['is_anomaly']}")
    print(f"语义一致性: {result['semantic_consistency']['is_anomaly']}") 
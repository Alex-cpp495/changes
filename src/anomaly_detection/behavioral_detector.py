"""
行为异常检测器
检测研报发布行为的异常模式，包括时机异常、频率异常、关注转移等
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import logging
import re

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class BehavioralAnomalyDetector:
    """
    行为异常检测器
    
    检测研报发布行为的异常模式：
    1. 时机异常 - 重大事件前后的异常发布时机
    2. 频率异常 - 发布频率的突然变化
    3. 关注转移异常 - 突然关注新股票或行业
    4. 写作习惯变化 - 文本风格和结构的变化
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 配置参数
        text_processor: 文本处理器
        historical_behavior: 历史行为数据
        market_events: 市场事件记录
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化行为异常检测器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        self.file_manager = get_file_manager()
        
        # 历史行为数据
        self.historical_behavior = {
            'publication_times': [],  # 发布时间列表
            'publication_frequency': defaultdict(int),  # 按期间统计的发布频率
            'stock_focus': defaultdict(int),  # 关注的股票频次
            'industry_focus': defaultdict(int),  # 关注的行业频次
            'writing_patterns': {
                'avg_sentence_length': [],
                'avg_paragraph_count': [],
                'keyword_usage': defaultdict(list)
            },
            'report_count': 0,
            'last_update': None
        }
        
        # 市场事件记录
        self.market_events = []
        
        # 统计参数
        self.is_trained = False
        self.frequency_baseline = {}
        self.focus_baseline = {}
        self.writing_baseline = {}
        
        # 加载历史数据
        self._load_historical_data()
        
        logger.info("行为异常检测器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('behavioral_anomaly', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'timing_anomaly': {
                'event_window_hours': 24,  # 重大事件前后时间窗口
                'min_gap_hours': 2,        # 最小发布间隔
                'suspicious_timing_threshold': 0.1  # 可疑时机阈值
            },
            'frequency_anomaly': {
                'baseline_period_days': 30,  # 基线计算周期
                'frequency_change_threshold': 2.0,  # 频率变化阈值（倍数）
                'min_reports_for_baseline': 10  # 基线计算最少报告数
            },
            'focus_shift_anomaly': {
                'new_stock_threshold': 0.3,    # 新关注股票占比阈值
                'focus_change_threshold': 0.5, # 关注转移阈值
                'tracking_period_days': 90     # 关注跟踪周期
            },
            'writing_habit_anomaly': {
                'sentence_length_threshold': 2.0,    # 句长变化Z-score阈值
                'paragraph_count_threshold': 2.0,    # 段落数变化阈值
                'keyword_usage_threshold': 0.3       # 关键词使用变化阈值
            },
            'min_historical_reports': 20,  # 最少历史报告数
            'anomaly_score_weights': {
                'timing': 0.25,
                'frequency': 0.25,
                'focus_shift': 0.25,
                'writing_habit': 0.25
            }
        }
    
    def _load_historical_data(self):
        """加载历史行为数据"""
        try:
            data_path = Path("data/models/behavioral_history.pkl")
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.historical_behavior.update(data.get('behavior', {}))
                self.market_events = data.get('events', [])
                
                # 重建基线模型
                if self.historical_behavior['report_count'] >= self.config['min_historical_reports']:
                    self._build_baseline_models()
                    logger.info(f"已加载 {self.historical_behavior['report_count']} 份历史行为数据")
                
        except Exception as e:
            logger.warning(f"历史数据加载失败: {e}")
    
    def _save_historical_data(self):
        """保存历史行为数据"""
        try:
            data_path = Path("data/models/behavioral_history.pkl")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'behavior': self.historical_behavior,
                'events': self.market_events
            }
            
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug("历史行为数据已保存")
            
        except Exception as e:
            logger.error(f"历史数据保存失败: {e}")
    
    def add_market_event(self, event_time: datetime, event_type: str, 
                        description: str, affected_stocks: List[str] = None):
        """
        添加市场事件记录
        
        Args:
            event_time: 事件时间
            event_type: 事件类型（earnings, news, announcement等）
            description: 事件描述
            affected_stocks: 受影响的股票列表
        """
        event = {
            'time': event_time,
            'type': event_type,
            'description': description,
            'affected_stocks': affected_stocks or [],
            'id': len(self.market_events)
        }
        
        self.market_events.append(event)
        self._save_historical_data()
        
        logger.info(f"添加市场事件: {event_type} - {description}")
    
    def update_historical_behavior(self, reports_data: List[Dict[str, Any]]):
        """
        更新历史行为数据
        
        Args:
            reports_data: 报告数据列表，每个包含时间、股票、内容等信息
        """
        try:
            for report in reports_data:
                # 提取基本信息
                pub_time = report.get('publication_time')
                if isinstance(pub_time, str):
                    pub_time = datetime.fromisoformat(pub_time)
                elif not isinstance(pub_time, datetime):
                    continue
                
                stocks = report.get('stocks', [])
                industries = report.get('industries', [])
                content = report.get('content', '')
                
                # 更新发布时间
                self.historical_behavior['publication_times'].append(pub_time)
                
                # 更新发布频率（按天统计）
                date_key = pub_time.date().isoformat()
                self.historical_behavior['publication_frequency'][date_key] += 1
                
                # 更新股票关注
                for stock in stocks:
                    self.historical_behavior['stock_focus'][stock] += 1
                
                # 更新行业关注
                for industry in industries:
                    self.historical_behavior['industry_focus'][industry] += 1
                
                # 更新写作模式
                if content:
                    self._update_writing_patterns(content)
                
                self.historical_behavior['report_count'] += 1
            
            # 更新时间戳
            self.historical_behavior['last_update'] = datetime.now()
            
            # 重建基线模型
            if self.historical_behavior['report_count'] >= self.config['min_historical_reports']:
                self._build_baseline_models()
            
            # 保存数据
            self._save_historical_data()
            
            logger.info(f"已更新 {len(reports_data)} 份报告的行为数据")
            
        except Exception as e:
            logger.error(f"历史行为数据更新失败: {e}")
            raise
    
    def _update_writing_patterns(self, content: str):
        """更新写作模式统计"""
        try:
            # 句子长度统计
            sentences = re.split(r'[。！？]', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                avg_sentence_length = np.mean([len(s) for s in sentences])
                self.historical_behavior['writing_patterns']['avg_sentence_length'].append(avg_sentence_length)
            
            # 段落统计
            paragraphs = content.split('\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            paragraph_count = len(paragraphs)
            self.historical_behavior['writing_patterns']['avg_paragraph_count'].append(paragraph_count)
            
            # 关键词使用统计
            keywords = self.text_processor.extract_keywords(content, topk=10)
            for keyword, score in keywords:
                self.historical_behavior['writing_patterns']['keyword_usage'][keyword].append(score)
                
        except Exception as e:
            logger.warning(f"写作模式更新失败: {e}")
    
    def _build_baseline_models(self):
        """构建基线模型"""
        try:
            # 频率基线
            frequencies = list(self.historical_behavior['publication_frequency'].values())
            if frequencies:
                self.frequency_baseline = {
                    'mean': np.mean(frequencies),
                    'std': np.std(frequencies),
                    'percentiles': np.percentile(frequencies, [25, 50, 75, 90, 95])
                }
            
            # 关注基线
            stock_counts = list(self.historical_behavior['stock_focus'].values())
            if stock_counts:
                self.focus_baseline = {
                    'stock_diversity': len(self.historical_behavior['stock_focus']),
                    'avg_focus_count': np.mean(stock_counts),
                    'top_stocks': sorted(self.historical_behavior['stock_focus'].items(), 
                                       key=lambda x: x[1], reverse=True)[:20]
                }
            
            # 写作基线
            patterns = self.historical_behavior['writing_patterns']
            if patterns['avg_sentence_length']:
                self.writing_baseline = {
                    'sentence_length': {
                        'mean': np.mean(patterns['avg_sentence_length']),
                        'std': np.std(patterns['avg_sentence_length'])
                    },
                    'paragraph_count': {
                        'mean': np.mean(patterns['avg_paragraph_count']),
                        'std': np.std(patterns['avg_paragraph_count'])
                    }
                }
            
            self.is_trained = True
            logger.info("行为基线模型构建完成")
            
        except Exception as e:
            logger.error(f"基线模型构建失败: {e}")
            raise
    
    def detect_behavioral_anomalies(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测行为异常
        
        Args:
            report_data: 报告数据，包含时间、股票、内容等信息
            
        Returns:
            Dict[str, Any]: 行为异常检测结果
            
        Raises:
            RuntimeError: 检测器未训练
        """
        if not self.is_trained:
            if self.historical_behavior['report_count'] < self.config['min_historical_reports']:
                raise RuntimeError(f"历史数据不足，需要至少 {self.config['min_historical_reports']} 份报告")
            else:
                self._build_baseline_models()
        
        try:
            results = {
                'timing_anomaly': self._detect_timing_anomaly(report_data),
                'frequency_anomaly': self._detect_frequency_anomaly(report_data),
                'focus_shift_anomaly': self._detect_focus_shift_anomaly(report_data),
                'writing_habit_anomaly': self._detect_writing_habit_anomaly(report_data),
                'overall_score': 0.0,
                'anomaly_level': 'NORMAL',
                'details': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 计算综合异常分数
            weights = self.config['anomaly_score_weights']
            anomaly_scores = [
                results['timing_anomaly']['score'] * weights['timing'],
                results['frequency_anomaly']['score'] * weights['frequency'],
                results['focus_shift_anomaly']['score'] * weights['focus_shift'],
                results['writing_habit_anomaly']['score'] * weights['writing_habit']
            ]
            
            results['overall_score'] = sum(anomaly_scores)
            
            # 确定异常等级
            results['anomaly_level'] = self._determine_anomaly_level(results['overall_score'])
            
            # 详细信息
            results['details'] = {
                'report_count': self.historical_behavior['report_count'],
                'last_update': self.historical_behavior['last_update'].isoformat() if self.historical_behavior['last_update'] else None,
                'market_events_count': len(self.market_events),
                'baseline_period': self.config['frequency_anomaly']['baseline_period_days']
            }
            
            return results
            
        except Exception as e:
            logger.error(f"行为异常检测失败: {e}")
            raise
    
    def _detect_timing_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测时机异常"""
        pub_time = report_data.get('publication_time')
        if isinstance(pub_time, str):
            pub_time = datetime.fromisoformat(pub_time)
        
        if not isinstance(pub_time, datetime):
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无效的发布时间"
            }
        
        window_hours = self.config['timing_anomaly']['event_window_hours']
        suspicious_events = []
        
        # 检查是否在重大事件附近发布
        for event in self.market_events:
            event_time = event['time']
            time_diff = abs((pub_time - event_time).total_seconds() / 3600)
            
            if time_diff <= window_hours:
                # 检查是否涉及相关股票
                report_stocks = set(report_data.get('stocks', []))
                event_stocks = set(event.get('affected_stocks', []))
                
                if report_stocks.intersection(event_stocks):
                    suspicious_events.append({
                        'event': event,
                        'time_diff_hours': time_diff,
                        'overlap_stocks': list(report_stocks.intersection(event_stocks))
                    })
        
        # 计算异常分数
        if suspicious_events:
            # 基于时间接近程度和股票重叠度计算分数
            max_score = 0.0
            for susp_event in suspicious_events:
                time_factor = 1.0 - (susp_event['time_diff_hours'] / window_hours)
                overlap_factor = len(susp_event['overlap_stocks']) / max(len(report_data.get('stocks', [])), 1)
                event_score = (time_factor + overlap_factor) / 2
                max_score = max(max_score, event_score)
            
            is_anomaly = max_score > self.config['timing_anomaly']['suspicious_timing_threshold']
        else:
            max_score = 0.0
            is_anomaly = False
        
        return {
            'is_anomaly': is_anomaly,
            'score': max_score,
            'suspicious_events_count': len(suspicious_events),
            'suspicious_events': suspicious_events[:3],  # 只显示前3个
            'description': f"发现{len(suspicious_events)}个时机可疑事件，最高分数{max_score:.2f}"
        }
    
    def _detect_frequency_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测频率异常"""
        pub_time = report_data.get('publication_time')
        if isinstance(pub_time, str):
            pub_time = datetime.fromisoformat(pub_time)
        
        if not isinstance(pub_time, datetime):
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无效的发布时间"
            }
        
        baseline_days = self.config['frequency_anomaly']['baseline_period_days']
        threshold = self.config['frequency_anomaly']['frequency_change_threshold']
        
        # 计算最近频率
        recent_start = pub_time - timedelta(days=baseline_days)
        recent_count = sum(1 for t in self.historical_behavior['publication_times'] 
                          if recent_start <= t <= pub_time)
        recent_frequency = recent_count / baseline_days
        
        # 与基线比较
        baseline_freq = self.frequency_baseline.get('mean', 0)
        
        if baseline_freq > 0:
            frequency_ratio = recent_frequency / baseline_freq
            is_anomaly = frequency_ratio > threshold or frequency_ratio < (1.0 / threshold)
            
            # 异常分数
            if frequency_ratio > threshold:
                anomaly_score = min((frequency_ratio - threshold) / threshold, 1.0)
            elif frequency_ratio < (1.0 / threshold):
                anomaly_score = min((1.0 / threshold - frequency_ratio) / (1.0 / threshold), 1.0)
            else:
                anomaly_score = 0.0
        else:
            frequency_ratio = 1.0
            is_anomaly = False
            anomaly_score = 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'recent_frequency': recent_frequency,
            'baseline_frequency': baseline_freq,
            'frequency_ratio': frequency_ratio,
            'description': f"最近频率{recent_frequency:.2f}/天，基线{baseline_freq:.2f}/天，比率{frequency_ratio:.2f}"
        }
    
    def _detect_focus_shift_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测关注转移异常"""
        current_stocks = set(report_data.get('stocks', []))
        
        if not current_stocks:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无股票信息"
            }
        
        # 历史关注的股票
        historical_stocks = set(self.historical_behavior['stock_focus'].keys())
        
        # 计算新股票占比
        new_stocks = current_stocks - historical_stocks
        new_stock_ratio = len(new_stocks) / len(current_stocks)
        
        threshold = self.config['focus_shift_anomaly']['new_stock_threshold']
        is_anomaly = new_stock_ratio > threshold
        
        # 异常分数
        anomaly_score = min(new_stock_ratio / threshold, 1.0) if threshold > 0 else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'new_stocks_count': len(new_stocks),
            'total_stocks_count': len(current_stocks),
            'new_stock_ratio': new_stock_ratio,
            'new_stocks': list(new_stocks),
            'description': f"新关注股票占比{new_stock_ratio:.2%}，发现{len(new_stocks)}只新股票"
        }
    
    def _detect_writing_habit_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测写作习惯异常"""
        content = report_data.get('content', '')
        
        if not content:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无内容信息"
            }
        
        # 计算当前写作特征
        current_features = self._extract_writing_features(content)
        
        # 与基线比较
        anomaly_scores = []
        details = {}
        
        # 句子长度异常
        if 'sentence_length' in self.writing_baseline:
            baseline = self.writing_baseline['sentence_length']
            current_length = current_features['avg_sentence_length']
            
            if baseline['std'] > 0:
                z_score = abs(current_length - baseline['mean']) / baseline['std']
                threshold = self.config['writing_habit_anomaly']['sentence_length_threshold']
                
                if z_score > threshold:
                    sentence_score = min(z_score / threshold - 1.0, 1.0)
                    anomaly_scores.append(sentence_score)
                    details['sentence_length_anomaly'] = True
                else:
                    details['sentence_length_anomaly'] = False
                
                details['sentence_length_zscore'] = z_score
        
        # 段落数异常
        if 'paragraph_count' in self.writing_baseline:
            baseline = self.writing_baseline['paragraph_count']
            current_count = current_features['paragraph_count']
            
            if baseline['std'] > 0:
                z_score = abs(current_count - baseline['mean']) / baseline['std']
                threshold = self.config['writing_habit_anomaly']['paragraph_count_threshold']
                
                if z_score > threshold:
                    paragraph_score = min(z_score / threshold - 1.0, 1.0)
                    anomaly_scores.append(paragraph_score)
                    details['paragraph_count_anomaly'] = True
                else:
                    details['paragraph_count_anomaly'] = False
                
                details['paragraph_count_zscore'] = z_score
        
        # 综合异常分数
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = overall_score > 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'current_features': current_features,
            'details': details,
            'description': f"写作习惯异常分数{overall_score:.2f}"
        }
    
    def _extract_writing_features(self, content: str) -> Dict[str, Any]:
        """提取写作特征"""
        # 句子分析
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_sentence_length = np.mean([len(s) for s in sentences]) if sentences else 0
        
        # 段落分析
        paragraphs = content.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': paragraph_count,
            'total_length': len(content),
            'sentence_count': len(sentences)
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
    
    def get_behavior_summary(self) -> Dict[str, Any]:
        """
        获取行为摘要
        
        Returns:
            Dict[str, Any]: 行为统计摘要
        """
        return {
            'report_count': self.historical_behavior['report_count'],
            'tracking_period_days': (
                (datetime.now() - min(self.historical_behavior['publication_times'])).days
                if self.historical_behavior['publication_times'] else 0
            ),
            'market_events_count': len(self.market_events),
            'unique_stocks_tracked': len(self.historical_behavior['stock_focus']),
            'unique_industries_tracked': len(self.historical_behavior['industry_focus']),
            'frequency_baseline': self.frequency_baseline,
            'focus_baseline': self.focus_baseline,
            'writing_baseline': self.writing_baseline,
            'is_trained': self.is_trained,
            'last_update': self.historical_behavior['last_update'].isoformat() if self.historical_behavior['last_update'] else None
        }


# 全局检测器实例
_global_behavioral_detector = None


def get_behavioral_detector() -> BehavioralAnomalyDetector:
    """
    获取全局行为异常检测器实例
    
    Returns:
        BehavioralAnomalyDetector: 检测器实例
    """
    global _global_behavioral_detector
    
    if _global_behavioral_detector is None:
        _global_behavioral_detector = BehavioralAnomalyDetector()
    
    return _global_behavioral_detector


if __name__ == "__main__":
    # 使用示例
    detector = BehavioralAnomalyDetector()
    
    # 添加市场事件
    detector.add_market_event(
        event_time=datetime.now() - timedelta(hours=2),
        event_type="earnings",
        description="A股公司发布财报",
        affected_stocks=["000001", "000002"]
    )
    
    # 模拟历史数据
    sample_reports = []
    base_time = datetime.now() - timedelta(days=60)
    
    for i in range(50):
        report = {
            'publication_time': base_time + timedelta(days=i, hours=np.random.randint(9, 17)),
            'stocks': [f"00000{np.random.randint(1, 6)}"],
            'industries': [f"行业{np.random.randint(1, 4)}"],
            'content': f"这是第{i+1}份研报的内容。公司业绩表现良好，预计未来增长稳定。建议投资者关注相关风险。"
        }
        sample_reports.append(report)
    
    # 更新历史数据
    detector.update_historical_behavior(sample_reports)
    
    # 检测异常行为
    test_report = {
        'publication_time': datetime.now(),
        'stocks': ["000001", "999999"],  # 包含新股票
        'industries': ["新兴行业"],
        'content': "这是一个异常简短的报告。暴涨！"
    }
    
    result = detector.detect_behavioral_anomalies(test_report)
    
    print("行为异常检测结果:")
    print(f"整体异常分数: {result['overall_score']:.3f}")
    print(f"异常等级: {result['anomaly_level']}")
    print(f"时机异常: {result['timing_anomaly']['is_anomaly']}")
    print(f"频率异常: {result['frequency_anomaly']['is_anomaly']}")
    print(f"关注转移异常: {result['focus_shift_anomaly']['is_anomaly']}")
    print(f"写作习惯异常: {result['writing_habit_anomaly']['is_anomaly']}") 
"""
集成异常检测器
综合统计、行为、市场、语义四个检测器的结果，提供最终的异常评估
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging
from pathlib import Path

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

from .statistical_detector import get_statistical_detector
from .behavioral_detector import get_behavioral_detector
from .market_detector import get_market_detector
from .semantic_detector import get_semantic_detector

logger = get_logger(__name__)


class EnsembleAnomalyDetector:
    """
    集成异常检测器
    
    综合四个异常检测器的结果：
    1. 统计异常检测器 - 文本统计特征异常
    2. 行为异常检测器 - 发布行为模式异常
    3. 市场异常检测器 - 市场表现相关异常
    4. 语义异常检测器 - 深度语义分析异常
    
    通过加权融合、投票机制、阈值调节等方法提供最终异常评估
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 配置参数
        detectors: 各个检测器实例
        detection_history: 检测历史记录
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化集成异常检测器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.file_manager = get_file_manager()
        
        # 初始化各个检测器
        self.detectors = {
            'statistical': get_statistical_detector(),
            'behavioral': get_behavioral_detector(),
            'market': get_market_detector(),
            'semantic': get_semantic_detector()
        }
        
        # 检测历史记录
        self.detection_history = {
            'reports': [],                    # 检测报告历史
            'anomaly_trends': defaultdict(list),  # 异常趋势统计
            'detector_performance': defaultdict(dict),  # 检测器性能
            'ensemble_metrics': {
                'total_detections': 0,
                'anomaly_count': 0,
                'false_positive_rate': 0.0,
                'detection_accuracy': 0.0
            },
            'last_update': None
        }
        
        # 动态权重调整
        self.dynamic_weights = {
            'statistical': 1.0,
            'behavioral': 1.0,
            'market': 1.0,
            'semantic': 1.0
        }
        
        self.is_trained = False
        
        # 加载历史数据
        self._load_detection_history()
        
        logger.info("集成异常检测器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('ensemble_detector', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'fusion_strategy': {
                'method': 'weighted_average',  # weighted_average, majority_vote, dynamic_threshold
                'base_weights': {
                    'statistical': 0.25,
                    'behavioral': 0.25,
                    'market': 0.25,
                    'semantic': 0.25
                },
                'enable_dynamic_weighting': True,
                'confidence_threshold': 0.6,
                'voting_threshold': 0.5  # 投票法阈值
            },
            'anomaly_levels': {
                'CRITICAL': {'threshold': 0.8, 'confidence': 0.9},
                'HIGH': {'threshold': 0.6, 'confidence': 0.8},
                'MEDIUM': {'threshold': 0.4, 'confidence': 0.7},
                'LOW': {'threshold': 0.2, 'confidence': 0.6},
                'NORMAL': {'threshold': 0.0, 'confidence': 0.5}
            },
            'performance_tracking': {
                'enable_feedback_learning': True,
                'performance_window_days': 30,
                'min_samples_for_weight_update': 20,
                'weight_update_rate': 0.1
            },
            'alert_rules': {
                'immediate_alert_threshold': 0.8,
                'consecutive_anomaly_threshold': 3,
                'detector_consensus_threshold': 3,  # 至少3个检测器同意
                'enable_cross_validation': True
            },
            'output_formatting': {
                'include_individual_scores': True,
                'include_confidence_intervals': True,
                'include_explanation': True,
                'max_details_length': 500
            }
        }
    
    def _load_detection_history(self):
        """加载检测历史记录"""
        try:
            history_path = Path("data/results/detection_history.json")
            if history_path.exists():
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.detection_history.update(data)
                
                # 检查是否有足够的历史数据进行训练
                if self.detection_history['ensemble_metrics']['total_detections'] >= 50:
                    self.is_trained = True
                    logger.info(f"已加载 {self.detection_history['ensemble_metrics']['total_detections']} 份检测历史")
                
        except Exception as e:
            logger.warning(f"检测历史加载失败: {e}")
    
    def _save_detection_history(self):
        """保存检测历史记录"""
        try:
            history_path = Path("data/results/detection_history.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换datetime对象为ISO格式字符串
            serializable_history = self._make_serializable(self.detection_history)
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
            logger.debug("检测历史已保存")
            
        except Exception as e:
            logger.error(f"检测历史保存失败: {e}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.number):
            return float(obj)
        else:
            return obj
    
    def detect_anomalies(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行综合异常检测
        
        Args:
            report_data: 报告数据，包含文本内容、股票信息、作者等
            
        Returns:
            Dict[str, Any]: 综合异常检测结果
        """
        try:
            start_time = datetime.now()
            
            # 1. 执行各个检测器
            logger.debug("开始执行各个异常检测器")
            individual_results = self._run_individual_detectors(report_data)
            
            # 2. 融合检测结果
            logger.debug("开始融合检测结果")
            ensemble_result = self._fuse_detection_results(individual_results, report_data)
            
            # 3. 计算置信度和解释性
            logger.debug("计算置信度和生成解释")
            ensemble_result = self._enhance_result_with_confidence(ensemble_result, individual_results)
            
            # 4. 应用警报规则
            ensemble_result = self._apply_alert_rules(ensemble_result, individual_results)
            
            # 5. 格式化输出
            ensemble_result = self._format_output(ensemble_result, individual_results, report_data)
            
            # 6. 记录检测历史
            detection_time = (datetime.now() - start_time).total_seconds()
            self._record_detection_result(ensemble_result, individual_results, report_data, detection_time)
            
            logger.info(f"集成异常检测完成，总体异常分数: {ensemble_result['overall_anomaly_score']:.3f}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"集成异常检测失败: {e}")
            raise
    
    def _run_individual_detectors(self, report_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """运行各个检测器"""
        results = {}
        
        # 统计异常检测
        try:
            results['statistical'] = self.detectors['statistical'].detect_statistical_anomalies(
                report_data.get('content', '')
            )
        except Exception as e:
            logger.warning(f"统计检测器执行失败: {e}")
            results['statistical'] = self._create_error_result('statistical', str(e))
        
        # 行为异常检测
        try:
            results['behavioral'] = self.detectors['behavioral'].detect_behavioral_anomalies(report_data)
        except Exception as e:
            logger.warning(f"行为检测器执行失败: {e}")
            results['behavioral'] = self._create_error_result('behavioral', str(e))
        
        # 市场异常检测
        try:
            results['market'] = self.detectors['market'].detect_market_anomalies(report_data)
        except Exception as e:
            logger.warning(f"市场检测器执行失败: {e}")
            results['market'] = self._create_error_result('market', str(e))
        
        # 语义异常检测
        try:
            results['semantic'] = self.detectors['semantic'].detect_semantic_anomalies(report_data)
        except Exception as e:
            logger.warning(f"语义检测器执行失败: {e}")
            results['semantic'] = self._create_error_result('semantic', str(e))
        
        return results
    
    def _create_error_result(self, detector_type: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'overall_score': 0.0,
            'anomaly_level': 'NORMAL',
            'error': error_msg,
            'detector_available': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fuse_detection_results(self, individual_results: Dict[str, Dict[str, Any]], 
                              report_data: Dict[str, Any]) -> Dict[str, Any]:
        """融合检测结果"""
        fusion_method = self.config['fusion_strategy']['method']
        
        if fusion_method == 'weighted_average':
            return self._weighted_average_fusion(individual_results)
        elif fusion_method == 'majority_vote':
            return self._majority_vote_fusion(individual_results)
        elif fusion_method == 'dynamic_threshold':
            return self._dynamic_threshold_fusion(individual_results, report_data)
        else:
            # 默认使用加权平均
            return self._weighted_average_fusion(individual_results)
    
    def _weighted_average_fusion(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """加权平均融合"""
        # 获取权重
        if self.config['fusion_strategy']['enable_dynamic_weighting'] and self.is_trained:
            weights = self.dynamic_weights
        else:
            weights = self.config['fusion_strategy']['base_weights']
        
        # 计算加权平均分数
        total_score = 0.0
        total_weight = 0.0
        active_detectors = []
        
        for detector_name, result in individual_results.items():
            if result.get('detector_available', True):  # 检测器可用
                score = result.get('overall_score', 0.0)
                weight = weights.get(detector_name, 0.25)
                
                total_score += score * weight
                total_weight += weight
                active_detectors.append(detector_name)
        
        # 计算最终分数
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        # 确定异常等级
        anomaly_level = self._determine_anomaly_level(final_score)
        
        return {
            'overall_anomaly_score': final_score,
            'anomaly_level': anomaly_level,
            'fusion_method': 'weighted_average',
            'active_detectors': active_detectors,
            'weights_used': {k: weights[k] for k in active_detectors},
            'individual_scores': {k: individual_results[k].get('overall_score', 0.0) for k in active_detectors}
        }
    
    def _majority_vote_fusion(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """多数投票融合"""
        voting_threshold = self.config['fusion_strategy']['voting_threshold']
        
        # 收集投票
        votes = {}
        anomaly_votes = 0
        total_votes = 0
        
        for detector_name, result in individual_results.items():
            if result.get('detector_available', True):
                score = result.get('overall_score', 0.0)
                is_anomaly = score > voting_threshold
                
                votes[detector_name] = {
                    'score': score,
                    'is_anomaly': is_anomaly
                }
                
                if is_anomaly:
                    anomaly_votes += 1
                total_votes += 1
        
        # 计算最终结果
        if total_votes > 0:
            anomaly_ratio = anomaly_votes / total_votes
            final_score = anomaly_ratio
            is_anomaly = anomaly_votes > total_votes / 2  # 超过半数认为异常
        else:
            final_score = 0.0
            is_anomaly = False
        
        anomaly_level = self._determine_anomaly_level(final_score)
        
        return {
            'overall_anomaly_score': final_score,
            'anomaly_level': anomaly_level,
            'fusion_method': 'majority_vote',
            'anomaly_votes': anomaly_votes,
            'total_votes': total_votes,
            'is_anomaly': is_anomaly,
            'votes': votes
        }
    
    def _dynamic_threshold_fusion(self, individual_results: Dict[str, Dict[str, Any]], 
                                 report_data: Dict[str, Any]) -> Dict[str, Any]:
        """动态阈值融合"""
        # 基于历史数据和当前上下文动态调整阈值
        base_result = self._weighted_average_fusion(individual_results)
        
        # 调整因子
        adjustment_factors = []
        
        # 1. 基于检测器一致性的调整
        scores = [result.get('overall_score', 0.0) for result in individual_results.values() 
                 if result.get('detector_available', True)]
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency_factor = 1.0 - min(score_std, 0.5)  # 一致性越高，置信度越高
            adjustment_factors.append(('consistency', consistency_factor))
        
        # 2. 基于数据质量的调整
        data_quality = self._assess_data_quality(report_data)
        adjustment_factors.append(('data_quality', data_quality))
        
        # 3. 基于历史准确性的调整
        if self.is_trained:
            historical_accuracy = self.detection_history['ensemble_metrics']['detection_accuracy']
            adjustment_factors.append(('historical_accuracy', historical_accuracy))
        
        # 计算调整后的分数
        final_score = base_result['overall_anomaly_score']
        adjustments = {}
        
        for factor_name, factor_value in adjustment_factors:
            adjustment = (factor_value - 0.5) * 0.1  # 最大调整±0.1
            final_score += adjustment
            adjustments[factor_name] = adjustment
        
        final_score = max(0.0, min(1.0, final_score))  # 限制在[0,1]范围
        
        anomaly_level = self._determine_anomaly_level(final_score)
        
        result = base_result.copy()
        result.update({
            'overall_anomaly_score': final_score,
            'anomaly_level': anomaly_level,
            'fusion_method': 'dynamic_threshold',
            'adjustments': adjustments,
            'adjustment_factors': adjustment_factors
        })
        
        return result
    
    def _assess_data_quality(self, report_data: Dict[str, Any]) -> float:
        """评估数据质量"""
        quality_score = 1.0
        
        # 检查必要字段
        required_fields = ['content', 'stocks', 'author']
        missing_fields = [field for field in required_fields if not report_data.get(field)]
        if missing_fields:
            quality_score -= 0.2 * len(missing_fields)
        
        # 检查内容长度
        content = report_data.get('content', '')
        if len(content) < 100:
            quality_score -= 0.3
        elif len(content) < 500:
            quality_score -= 0.1
        
        # 检查股票信息
        stocks = report_data.get('stocks', [])
        if not stocks:
            quality_score -= 0.2
        
        return max(0.0, min(1.0, quality_score))
    
    def _enhance_result_with_confidence(self, ensemble_result: Dict[str, Any], 
                                      individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """增强结果的置信度和解释性"""
        # 计算置信度
        confidence = self._calculate_confidence(ensemble_result, individual_results)
        
        # 生成解释
        explanation = self._generate_explanation(ensemble_result, individual_results)
        
        # 识别关键异常类型
        key_anomaly_types = self._identify_key_anomaly_types(individual_results)
        
        ensemble_result.update({
            'confidence': confidence,
            'explanation': explanation,
            'key_anomaly_types': key_anomaly_types,
            'risk_assessment': self._assess_risk_level(ensemble_result, confidence)
        })
        
        return ensemble_result
    
    def _calculate_confidence(self, ensemble_result: Dict[str, Any], 
                            individual_results: Dict[str, Dict[str, Any]]) -> float:
        """计算检测置信度"""
        scores = []
        for result in individual_results.values():
            if result.get('detector_available', True):
                scores.append(result.get('overall_score', 0.0))
        
        if not scores:
            return 0.0
        
        # 基于分数一致性计算置信度
        mean_score = np.mean(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        
        # 一致性越高，置信度越高
        consistency_confidence = 1.0 - min(score_std, 0.5) * 2
        
        # 基于异常分数强度
        intensity_confidence = min(mean_score * 2, 1.0) if mean_score > 0.5 else max(0.5, mean_score * 2)
        
        # 综合置信度
        confidence = (consistency_confidence + intensity_confidence) / 2
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_explanation(self, ensemble_result: Dict[str, Any], 
                            individual_results: Dict[str, Dict[str, Any]]) -> str:
        """生成检测结果解释"""
        explanations = []
        
        score = ensemble_result['overall_anomaly_score']
        level = ensemble_result['anomaly_level']
        
        # 总体评估
        if score >= 0.8:
            explanations.append(f"检测到严重异常（分数: {score:.2f}），建议立即关注")
        elif score >= 0.6:
            explanations.append(f"检测到高度异常（分数: {score:.2f}），需要重点关注")
        elif score >= 0.4:
            explanations.append(f"检测到中等异常（分数: {score:.2f}），建议进一步分析")
        elif score >= 0.2:
            explanations.append(f"检测到轻微异常（分数: {score:.2f}），可能需要关注")
        else:
            explanations.append(f"未检测到明显异常（分数: {score:.2f}）")
        
        # 各检测器贡献
        anomaly_detectors = []
        for detector_name, result in individual_results.items():
            if result.get('detector_available', True):
                detector_score = result.get('overall_score', 0.0)
                if detector_score > 0.4:
                    detector_level = result.get('anomaly_level', 'UNKNOWN')
                    anomaly_detectors.append(f"{detector_name}检测器({detector_level}, {detector_score:.2f})")
        
        if anomaly_detectors:
            explanations.append(f"异常主要来源: {', '.join(anomaly_detectors)}")
        
        return '; '.join(explanations)
    
    def _identify_key_anomaly_types(self, individual_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """识别关键异常类型"""
        anomaly_types = []
        
        for detector_name, result in individual_results.items():
            if result.get('detector_available', True) and result.get('overall_score', 0.0) > 0.4:
                # 根据检测器类型添加异常类型
                if detector_name == 'statistical':
                    if result.get('length_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('文本长度异常')
                    if result.get('vocabulary_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('词汇使用异常')
                    if result.get('sentiment_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('情感强度异常')
                
                elif detector_name == 'behavioral':
                    if result.get('timing_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('发布时机异常')
                    if result.get('frequency_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('发布频率异常')
                    if result.get('focus_shift_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('关注焦点异常')
                
                elif detector_name == 'market':
                    if result.get('return_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('收益率异常')
                    if result.get('volume_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('成交量异常')
                    if result.get('prediction_accuracy_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('预测准确性异常')
                
                elif detector_name == 'semantic':
                    if result.get('logical_contradiction', {}).get('is_anomaly', False):
                        anomaly_types.append('逻辑矛盾')
                    if result.get('historical_deviation', {}).get('is_anomaly', False):
                        anomaly_types.append('观点偏离')
                    if result.get('information_source_anomaly', {}).get('is_anomaly', False):
                        anomaly_types.append('信息源异常')
        
        return list(set(anomaly_types))  # 去重
    
    def _assess_risk_level(self, ensemble_result: Dict[str, Any], confidence: float) -> str:
        """评估风险等级"""
        score = ensemble_result['overall_anomaly_score']
        
        if score >= 0.8 and confidence >= 0.8:
            return 'CRITICAL'
        elif score >= 0.6 and confidence >= 0.7:
            return 'HIGH'
        elif score >= 0.4 and confidence >= 0.6:
            return 'MEDIUM'
        elif score >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _apply_alert_rules(self, ensemble_result: Dict[str, Any], 
                          individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """应用警报规则"""
        alert_config = self.config['alert_rules']
        alerts = []
        
        # 立即警报阈值
        if ensemble_result['overall_anomaly_score'] >= alert_config['immediate_alert_threshold']:
            alerts.append({
                'type': 'IMMEDIATE',
                'message': '检测到严重异常，需要立即处理',
                'priority': 'HIGH'
            })
        
        # 检测器共识
        consensus_count = sum(1 for result in individual_results.values() 
                            if result.get('detector_available', True) and 
                               result.get('overall_score', 0.0) > 0.5)
        
        if consensus_count >= alert_config['detector_consensus_threshold']:
            alerts.append({
                'type': 'CONSENSUS',
                'message': f'{consensus_count}个检测器同时发现异常',
                'priority': 'MEDIUM'
            })
        
        # 交叉验证
        if alert_config['enable_cross_validation']:
            cross_validation_result = self._cross_validate_results(individual_results)
            if cross_validation_result['is_reliable']:
                alerts.append({
                    'type': 'VALIDATED',
                    'message': '异常检测结果已通过交叉验证',
                    'priority': 'INFO'
                })
        
        ensemble_result['alerts'] = alerts
        return ensemble_result
    
    def _cross_validate_results(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """交叉验证检测结果"""
        # 简单的交叉验证：检查检测器之间的相关性
        scores = []
        for result in individual_results.values():
            if result.get('detector_available', True):
                scores.append(result.get('overall_score', 0.0))
        
        if len(scores) < 2:
            return {'is_reliable': False, 'reason': '检测器数量不足'}
        
        # 计算分数相关性
        correlation = np.corrcoef(scores) if len(scores) > 1 else np.array([[1.0]])
        avg_correlation = np.mean(correlation[np.triu_indices_from(correlation, k=1)])
        
        is_reliable = avg_correlation > 0.3  # 相关性阈值
        
        return {
            'is_reliable': is_reliable,
            'correlation': float(avg_correlation),
            'reason': '检测器结果相关性' + ('充足' if is_reliable else '不足')
        }
    
    def _format_output(self, ensemble_result: Dict[str, Any], 
                      individual_results: Dict[str, Dict[str, Any]], 
                      report_data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化输出结果"""
        output_config = self.config['output_formatting']
        
        # 基本结果
        formatted_result = {
            'detection_id': f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'overall_anomaly_score': ensemble_result['overall_anomaly_score'],
            'anomaly_level': ensemble_result['anomaly_level'],
            'confidence': ensemble_result.get('confidence', 0.0),
            'risk_assessment': ensemble_result.get('risk_assessment', 'UNKNOWN'),
            'fusion_method': ensemble_result.get('fusion_method', 'weighted_average'),
            'explanation': ensemble_result.get('explanation', '')[:output_config['max_details_length']],
            'key_anomaly_types': ensemble_result.get('key_anomaly_types', []),
            'alerts': ensemble_result.get('alerts', [])
        }
        
        # 包含个体分数
        if output_config['include_individual_scores']:
            formatted_result['individual_detector_results'] = {}
            for detector_name, result in individual_results.items():
                if result.get('detector_available', True):
                    formatted_result['individual_detector_results'][detector_name] = {
                        'score': result.get('overall_score', 0.0),
                        'anomaly_level': result.get('anomaly_level', 'NORMAL'),
                        'available': True
                    }
                else:
                    formatted_result['individual_detector_results'][detector_name] = {
                        'available': False,
                        'error': result.get('error', 'Unknown error')
                    }
        
        # 包含置信区间
        if output_config['include_confidence_intervals']:
            formatted_result['confidence_interval'] = self._calculate_confidence_interval(
                ensemble_result, individual_results
            )
        
        # 包含报告基本信息
        formatted_result['report_info'] = {
            'stocks': report_data.get('stocks', []),
            'author': report_data.get('author', 'Unknown'),
            'content_length': len(report_data.get('content', '')),
            'analysis_date': report_data.get('analysis_date', datetime.now()).isoformat() 
                           if isinstance(report_data.get('analysis_date'), datetime) 
                           else str(report_data.get('analysis_date', datetime.now()))
        }
        
        return formatted_result
    
    def _calculate_confidence_interval(self, ensemble_result: Dict[str, Any], 
                                     individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算置信区间"""
        scores = [result.get('overall_score', 0.0) for result in individual_results.values() 
                 if result.get('detector_available', True)]
        
        if len(scores) < 2:
            return {'lower': ensemble_result['overall_anomaly_score'], 
                   'upper': ensemble_result['overall_anomaly_score']}
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 95% 置信区间
        margin_of_error = 1.96 * std_score / np.sqrt(len(scores))
        
        return {
            'lower': max(0.0, mean_score - margin_of_error),
            'upper': min(1.0, mean_score + margin_of_error)
        }
    
    def _determine_anomaly_level(self, score: float) -> str:
        """确定异常等级"""
        levels = self.config['anomaly_levels']
        
        for level, config in levels.items():
            if score >= config['threshold']:
                return level
        
        return 'NORMAL'
    
    def _record_detection_result(self, ensemble_result: Dict[str, Any], 
                               individual_results: Dict[str, Dict[str, Any]], 
                               report_data: Dict[str, Any], 
                               detection_time: float):
        """记录检测结果"""
        try:
            # 添加到历史记录
            detection_record = {
                'detection_id': ensemble_result['detection_id'],
                'timestamp': ensemble_result['timestamp'],
                'overall_score': ensemble_result['overall_anomaly_score'],
                'anomaly_level': ensemble_result['anomaly_level'],
                'confidence': ensemble_result.get('confidence', 0.0),
                'individual_scores': {k: v.get('overall_score', 0.0) for k, v in individual_results.items()},
                'detection_time_seconds': detection_time,
                'report_info': ensemble_result['report_info']
            }
            
            self.detection_history['reports'].append(detection_record)
            
            # 更新统计信息
            self.detection_history['ensemble_metrics']['total_detections'] += 1
            if ensemble_result['overall_anomaly_score'] > 0.5:
                self.detection_history['ensemble_metrics']['anomaly_count'] += 1
            
            # 限制历史记录数量
            if len(self.detection_history['reports']) > 1000:
                self.detection_history['reports'] = self.detection_history['reports'][-1000:]
            
            self.detection_history['last_update'] = datetime.now()
            
            # 更新动态权重
            if self.config['fusion_strategy']['enable_dynamic_weighting']:
                self._update_dynamic_weights(individual_results, ensemble_result)
            
            # 保存历史记录
            self._save_detection_history()
            
        except Exception as e:
            logger.error(f"记录检测结果失败: {e}")
    
    def _update_dynamic_weights(self, individual_results: Dict[str, Dict[str, Any]], 
                              ensemble_result: Dict[str, Any]):
        """更新动态权重"""
        if not self.is_trained or len(self.detection_history['reports']) < 20:
            return
        
        update_rate = self.config['performance_tracking']['weight_update_rate']
        
        # 基于检测器性能更新权重
        for detector_name in self.dynamic_weights:
            if detector_name in individual_results:
                detector_result = individual_results[detector_name]
                if detector_result.get('detector_available', True):
                    # 简单的性能评估：与集成结果的一致性
                    detector_score = detector_result.get('overall_score', 0.0)
                    ensemble_score = ensemble_result['overall_anomaly_score']
                    
                    # 计算一致性
                    consistency = 1.0 - abs(detector_score - ensemble_score)
                    
                    # 更新权重
                    current_weight = self.dynamic_weights[detector_name]
                    new_weight = current_weight + update_rate * (consistency - 0.5)
                    self.dynamic_weights[detector_name] = max(0.1, min(1.5, new_weight))
        
        # 归一化权重
        total_weight = sum(self.dynamic_weights.values())
        if total_weight > 0:
            for detector_name in self.dynamic_weights:
                self.dynamic_weights[detector_name] /= total_weight
    
    def add_feedback(self, detection_id: str, is_true_positive: bool, 
                    feedback_details: Optional[str] = None):
        """
        添加人工反馈，用于改进检测性能
        
        Args:
            detection_id: 检测ID
            is_true_positive: 是否为真正例
            feedback_details: 反馈详情
        """
        try:
            # 查找对应的检测记录
            detection_record = None
            for record in self.detection_history['reports']:
                if record['detection_id'] == detection_id:
                    detection_record = record
                    break
            
            if not detection_record:
                logger.warning(f"未找到检测记录: {detection_id}")
                return
            
            # 添加反馈信息
            feedback = {
                'timestamp': datetime.now().isoformat(),
                'is_true_positive': is_true_positive,
                'details': feedback_details
            }
            
            if 'feedback' not in detection_record:
                detection_record['feedback'] = []
            detection_record['feedback'].append(feedback)
            
            # 更新性能指标
            self._update_performance_metrics()
            
            # 保存更新
            self._save_detection_history()
            
            logger.info(f"已添加检测反馈: {detection_id}")
            
        except Exception as e:
            logger.error(f"添加反馈失败: {e}")
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        feedback_records = [record for record in self.detection_history['reports'] 
                           if 'feedback' in record and record['feedback']]
        
        if not feedback_records:
            return
        
        # 计算准确率
        true_positives = sum(1 for record in feedback_records 
                           if any(fb['is_true_positive'] for fb in record['feedback']))
        
        total_feedback = len(feedback_records)
        accuracy = true_positives / total_feedback if total_feedback > 0 else 0.0
        
        # 计算假正例率
        false_positives = total_feedback - true_positives
        total_positive_predictions = sum(1 for record in self.detection_history['reports'] 
                                       if record['overall_score'] > 0.5)
        
        false_positive_rate = false_positives / max(total_positive_predictions, 1)
        
        # 更新指标
        self.detection_history['ensemble_metrics'].update({
            'detection_accuracy': accuracy,
            'false_positive_rate': false_positive_rate,
            'feedback_samples': total_feedback
        })
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        # 检查各检测器状态
        detector_status = {}
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'is_trained'):
                    is_trained = detector.is_trained
                else:
                    is_trained = True  # 假设可用
                
                detector_status[name] = {
                    'available': True,
                    'is_trained': is_trained
                }
            except Exception as e:
                detector_status[name] = {
                    'available': False,
                    'error': str(e)
                }
        
        return {
            'ensemble_detector': {
                'is_trained': self.is_trained,
                'total_detections': self.detection_history['ensemble_metrics']['total_detections'],
                'detection_accuracy': self.detection_history['ensemble_metrics']['detection_accuracy'],
                'false_positive_rate': self.detection_history['ensemble_metrics']['false_positive_rate']
            },
            'individual_detectors': detector_status,
            'dynamic_weights': self.dynamic_weights,
            'fusion_method': self.config['fusion_strategy']['method'],
            'last_update': self.detection_history['last_update']
        }


# 全局检测器实例
_global_ensemble_detector = None


def get_ensemble_detector() -> EnsembleAnomalyDetector:
    """
    获取全局集成异常检测器实例
    
    Returns:
        EnsembleAnomalyDetector: 检测器实例
    """
    global _global_ensemble_detector
    
    if _global_ensemble_detector is None:
        _global_ensemble_detector = EnsembleAnomalyDetector()
    
    return _global_ensemble_detector


if __name__ == "__main__":
    # 使用示例
    detector = EnsembleAnomalyDetector()
    
    # 模拟检测数据
    test_report = {
        'content': '公司基本面良好，但是业绩大幅下滑，建议卖出。据某内部人士透露，公司存在重大隐患。',
        'stocks': ['000001', '000002'],
        'author': '分析师A',
        'analysis_date': datetime.now(),
        'sentiment': 'negative'
    }
    
    # 执行集成检测
    result = detector.detect_anomalies(test_report)
    
    print("集成异常检测结果:")
    print(f"检测ID: {result['detection_id']}")
    print(f"整体异常分数: {result['overall_anomaly_score']:.3f}")
    print(f"异常等级: {result['anomaly_level']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"风险评估: {result['risk_assessment']}")
    print(f"关键异常类型: {result['key_anomaly_types']}")
    print(f"解释: {result['explanation']}")
    
    if result.get('alerts'):
        print("警报:")
        for alert in result['alerts']:
            print(f"  - {alert['type']}: {alert['message']} (优先级: {alert['priority']})")
    
    # 获取系统状态
    status = detector.get_system_status()
    print(f"\n系统状态:")
    print(f"总检测次数: {status['ensemble_detector']['total_detections']}")
    print(f"检测准确率: {status['ensemble_detector']['detection_accuracy']:.3f}")
    print(f"各检测器权重: {status['dynamic_weights']}") 
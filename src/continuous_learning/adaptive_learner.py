"""
自适应学习器
基于用户反馈和性能监控数据，自动调整模型参数和检测阈值，实现持续学习
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import joblib
import yaml
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .feedback_collector import get_feedback_collector, FeedbackType
from .model_monitor import get_model_monitor
from ..utils.logger import get_logger
from ..utils.config_loader import load_config
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class AdaptationStrategy(Enum):
    """自适应策略"""
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"    # 阈值调整
    PARAMETER_TUNING = "parameter_tuning"           # 参数调优
    FEATURE_WEIGHTING = "feature_weighting"         # 特征权重调整
    MODEL_SELECTION = "model_selection"             # 模型选择
    ENSEMBLE_WEIGHTING = "ensemble_weighting"       # 集成权重调整


class LearningTrigger(Enum):
    """学习触发条件"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEEDBACK_ACCUMULATION = "feedback_accumulation"
    SCHEDULED_UPDATE = "scheduled_update"
    MANUAL_TRIGGER = "manual_trigger"
    DATA_DRIFT_DETECTED = "data_drift_detected"


@dataclass
class AdaptationResult:
    """自适应结果"""
    strategy: AdaptationStrategy
    trigger: LearningTrigger
    timestamp: datetime
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement: float
    confidence: float
    applied: bool = False
    rollback_info: Optional[Dict[str, Any]] = None


@dataclass
class LearningObjective:
    """学习目标"""
    metric_name: str
    target_value: float
    tolerance: float
    weight: float = 1.0
    maximize: bool = True


class AdaptiveLearner:
    """
    自适应学习器
    
    功能：
    1. 阈值自适应 - 基于反馈自动调整检测阈值
    2. 参数优化 - 根据性能数据调优模型参数
    3. 特征权重调整 - 动态调整特征重要性
    4. 集成权重优化 - 优化集成模型权重
    5. 性能监控集成 - 与监控系统联动
    6. 自动回滚 - 性能下降时自动回滚
    7. 学习策略选择 - 智能选择最佳学习策略
    8. 数据漂移适应 - 检测和适应数据分布变化
    
    Args:
        config: 学习器配置
        
    Attributes:
        learning_history: 学习历史记录
        current_parameters: 当前参数
        objectives: 学习目标
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自适应学习器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 获取其他组件
        self.feedback_collector = get_feedback_collector()
        self.model_monitor = get_model_monitor()
        
        # 数据存储
        self.data_dir = Path("data/adaptive_learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 学习历史
        self.learning_history: List[AdaptationResult] = []
        self.parameter_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # 当前参数和配置
        self.current_parameters = self._load_current_parameters()
        self.learning_objectives = self._load_learning_objectives()
        
        # 学习状态
        self.learning_active = False
        self.learning_thread: Optional[threading.Thread] = None
        self.learning_lock = threading.RLock()
        
        # 性能基线
        self.performance_baseline = self._load_performance_baseline()
        
        # 学习策略配置
        self.adaptation_strategies = {
            AdaptationStrategy.THRESHOLD_ADJUSTMENT: self._adapt_thresholds,
            AdaptationStrategy.PARAMETER_TUNING: self._tune_parameters,
            AdaptationStrategy.FEATURE_WEIGHTING: self._adjust_feature_weights,
            AdaptationStrategy.ENSEMBLE_WEIGHTING: self._optimize_ensemble_weights
        }
        
        logger.info("自适应学习器初始化完成")
    
    def _load_current_parameters(self) -> Dict[str, Any]:
        """加载当前参数"""
        try:
            param_file = self.data_dir / "current_parameters.json"
            if param_file.exists():
                with open(param_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 默认参数
            return {
                'detection_thresholds': {
                    'statistical_anomaly': 0.7,
                    'behavioral_anomaly': 0.6,
                    'market_anomaly': 0.8,
                    'semantic_anomaly': 0.65,
                    'overall_anomaly': 0.7
                },
                'ensemble_weights': {
                    'statistical_detector': 0.25,
                    'behavioral_detector': 0.25,
                    'market_detector': 0.25,
                    'semantic_detector': 0.25
                },
                'feature_weights': {
                    'text_features': 1.0,
                    'market_features': 1.0,
                    'behavioral_features': 1.0,
                    'temporal_features': 1.0
                },
                'model_parameters': {
                    'confidence_threshold': 0.5,
                    'attention_temperature': 1.0,
                    'max_sequence_length': 512
                }
            }
            
        except Exception as e:
            logger.error(f"加载当前参数失败: {e}")
            return {}
    
    def _save_current_parameters(self):
        """保存当前参数"""
        try:
            param_file = self.data_dir / "current_parameters.json"
            with open(param_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_parameters, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存当前参数失败: {e}")
    
    def _load_learning_objectives(self) -> List[LearningObjective]:
        """加载学习目标"""
        try:
            objectives_file = self.data_dir / "learning_objectives.yaml"
            if objectives_file.exists():
                with open(objectives_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    return [LearningObjective(**obj) for obj in data.get('objectives', [])]
            
            # 默认学习目标
            return [
                LearningObjective(
                    metric_name="accuracy",
                    target_value=0.9,
                    tolerance=0.05,
                    weight=1.0,
                    maximize=True
                ),
                LearningObjective(
                    metric_name="f1_score",
                    target_value=0.85,
                    tolerance=0.05,
                    weight=0.8,
                    maximize=True
                ),
                LearningObjective(
                    metric_name="false_positive_rate",
                    target_value=0.1,
                    tolerance=0.02,
                    weight=0.6,
                    maximize=False
                )
            ]
            
        except Exception as e:
            logger.error(f"加载学习目标失败: {e}")
            return []
    
    def _load_performance_baseline(self) -> Dict[str, float]:
        """加载性能基线"""
        try:
            baseline_file = self.data_dir / "performance_baseline.json"
            if baseline_file.exists():
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 默认基线
            return {
                'accuracy': 0.8,
                'f1_score': 0.75,
                'precision': 0.8,
                'recall': 0.7,
                'false_positive_rate': 0.15
            }
            
        except Exception as e:
            logger.error(f"加载性能基线失败: {e}")
            return {}
    
    def start_adaptive_learning(self, check_interval: float = 3600.0):
        """
        开始自适应学习
        
        Args:
            check_interval: 检查间隔（秒）
        """
        if self.learning_active:
            logger.warning("自适应学习已经在运行中")
            return
        
        self.learning_active = True
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            args=(check_interval,),
            daemon=True
        )
        self.learning_thread.start()
        
        logger.info(f"开始自适应学习，检查间隔: {check_interval}秒")
    
    def stop_adaptive_learning(self):
        """停止自适应学习"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10.0)
        
        logger.info("自适应学习已停止")
    
    def _learning_loop(self, check_interval: float):
        """学习主循环"""
        while self.learning_active:
            try:
                # 检查学习触发条件
                triggers = self._check_learning_triggers()
                
                if triggers:
                    logger.info(f"检测到学习触发条件: {[t.value for t in triggers]}")
                    
                    # 执行自适应学习
                    for trigger in triggers:
                        self._execute_adaptive_learning(trigger)
                
                # 等待下次检查
                import time
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"自适应学习循环异常: {e}")
                import time
                time.sleep(60)  # 出错时等待1分钟再重试
    
    def _check_learning_triggers(self) -> List[LearningTrigger]:
        """检查学习触发条件"""
        triggers = []
        
        try:
            # 1. 检查性能退化
            if self._check_performance_degradation():
                triggers.append(LearningTrigger.PERFORMANCE_DEGRADATION)
            
            # 2. 检查反馈积累
            if self._check_feedback_accumulation():
                triggers.append(LearningTrigger.FEEDBACK_ACCUMULATION)
            
            # 3. 检查定期更新
            if self._check_scheduled_update():
                triggers.append(LearningTrigger.SCHEDULED_UPDATE)
            
            # 4. 检查数据漂移
            if self._check_data_drift():
                triggers.append(LearningTrigger.DATA_DRIFT_DETECTED)
            
        except Exception as e:
            logger.error(f"检查学习触发条件失败: {e}")
        
        return triggers
    
    def _check_performance_degradation(self) -> bool:
        """检查性能退化"""
        try:
            # 获取最近的性能指标
            current_metrics = self.model_monitor.get_current_metrics()
            
            if not current_metrics.get('metrics'):
                return False
            
            # 检查关键指标是否低于基线
            degradation_count = 0
            
            for metric_name, baseline_value in self.performance_baseline.items():
                current_value = current_metrics['metrics'].get(metric_name, {}).get('value')
                
                if current_value is not None:
                    # 检查是否有显著退化（超过10%）
                    if current_value < baseline_value * 0.9:
                        degradation_count += 1
            
            # 如果多个指标同时退化，触发学习
            return degradation_count >= 2
            
        except Exception as e:
            logger.error(f"检查性能退化失败: {e}")
            return False
    
    def _check_feedback_accumulation(self) -> bool:
        """检查反馈积累"""
        try:
            # 获取最近24小时的反馈
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            recent_feedback = self.feedback_collector.get_feedback_by_date_range(
                start_time, end_time
            )
            
            # 如果累积了足够的反馈（如20条），触发学习
            if len(recent_feedback) >= 20:
                # 检查负面反馈比例
                negative_feedback = sum(1 for f in recent_feedback if not f.is_correct)
                negative_ratio = negative_feedback / len(recent_feedback)
                
                # 如果负面反馈超过30%，触发学习
                return negative_ratio > 0.3
            
            return False
            
        except Exception as e:
            logger.error(f"检查反馈积累失败: {e}")
            return False
    
    def _check_scheduled_update(self) -> bool:
        """检查定期更新"""
        try:
            # 检查是否到了定期更新时间（如每周一次）
            if not self.learning_history:
                return True  # 如果从未学习过，立即触发
            
            last_learning = self.learning_history[-1].timestamp
            days_since_last = (datetime.now() - last_learning).days
            
            # 每7天触发一次定期学习
            return days_since_last >= 7
            
        except Exception as e:
            logger.error(f"检查定期更新失败: {e}")
            return False
    
    def _check_data_drift(self) -> bool:
        """检查数据漂移"""
        try:
            # 简化的数据漂移检测
            # 实际实现可能需要更复杂的统计方法
            
            # 获取最近的预测置信度分布
            current_metrics = self.model_monitor.get_current_metrics()
            confidence_trend = self.model_monitor.get_performance_trends('confidence', hours=24)
            
            if confidence_trend.get('data'):
                confidence_values = [d['value'] for d in confidence_trend['data']]
                
                if len(confidence_values) >= 20:
                    # 检查置信度分布是否发生显著变化
                    recent_mean = np.mean(confidence_values[-10:])
                    earlier_mean = np.mean(confidence_values[:10])
                    
                    # 如果置信度平均值下降超过20%，可能存在数据漂移
                    return recent_mean < earlier_mean * 0.8
            
            return False
            
        except Exception as e:
            logger.error(f"检查数据漂移失败: {e}")
            return False
    
    def _execute_adaptive_learning(self, trigger: LearningTrigger):
        """执行自适应学习"""
        try:
            with self.learning_lock:
                logger.info(f"执行自适应学习，触发条件: {trigger.value}")
                
                # 选择适应策略
                strategies = self._select_adaptation_strategies(trigger)
                
                best_result = None
                best_improvement = -float('inf')
                
                # 尝试每个策略
                for strategy in strategies:
                    try:
                        result = self._try_adaptation_strategy(strategy, trigger)
                        
                        if result and result.improvement > best_improvement:
                            best_result = result
                            best_improvement = result.improvement
                            
                    except Exception as e:
                        logger.error(f"策略 {strategy.value} 执行失败: {e}")
                
                # 应用最佳结果
                if best_result and best_improvement > 0:
                    self._apply_adaptation_result(best_result)
                    logger.info(f"应用自适应结果: 改进 {best_improvement:.4f}")
                else:
                    logger.info("没有找到有效的改进策略")
                    
        except Exception as e:
            logger.error(f"执行自适应学习失败: {e}")
    
    def _select_adaptation_strategies(self, trigger: LearningTrigger) -> List[AdaptationStrategy]:
        """选择适应策略"""
        strategy_map = {
            LearningTrigger.PERFORMANCE_DEGRADATION: [
                AdaptationStrategy.THRESHOLD_ADJUSTMENT,
                AdaptationStrategy.ENSEMBLE_WEIGHTING
            ],
            LearningTrigger.FEEDBACK_ACCUMULATION: [
                AdaptationStrategy.THRESHOLD_ADJUSTMENT,
                AdaptationStrategy.FEATURE_WEIGHTING
            ],
            LearningTrigger.SCHEDULED_UPDATE: [
                AdaptationStrategy.PARAMETER_TUNING,
                AdaptationStrategy.ENSEMBLE_WEIGHTING
            ],
            LearningTrigger.DATA_DRIFT_DETECTED: [
                AdaptationStrategy.FEATURE_WEIGHTING,
                AdaptationStrategy.PARAMETER_TUNING
            ]
        }
        
        return strategy_map.get(trigger, [AdaptationStrategy.THRESHOLD_ADJUSTMENT])
    
    def _try_adaptation_strategy(self, strategy: AdaptationStrategy, 
                                trigger: LearningTrigger) -> Optional[AdaptationResult]:
        """尝试适应策略"""
        try:
            # 获取当前性能
            performance_before = self._evaluate_current_performance()
            
            # 执行策略
            adaptation_func = self.adaptation_strategies.get(strategy)
            if not adaptation_func:
                logger.warning(f"未找到策略实现: {strategy.value}")
                return None
            
            old_parameters = self.current_parameters.copy()
            new_parameters = adaptation_func(trigger)
            
            if not new_parameters:
                return None
            
            # 临时应用新参数评估性能
            temp_parameters = self.current_parameters.copy()
            self.current_parameters = new_parameters
            
            try:
                performance_after = self._evaluate_current_performance()
                improvement = self._calculate_improvement(performance_before, performance_after)
                
                # 创建结果对象
                result = AdaptationResult(
                    strategy=strategy,
                    trigger=trigger,
                    timestamp=datetime.now(),
                    old_parameters=old_parameters,
                    new_parameters=new_parameters,
                    performance_before=performance_before,
                    performance_after=performance_after,
                    improvement=improvement,
                    confidence=self._calculate_confidence(improvement, performance_before, performance_after)
                )
                
                return result
                
            finally:
                # 恢复原参数
                self.current_parameters = temp_parameters
                
        except Exception as e:
            logger.error(f"尝试适应策略失败: {e}")
            return None
    
    def _adapt_thresholds(self, trigger: LearningTrigger) -> Optional[Dict[str, Any]]:
        """阈值调整策略"""
        try:
            new_params = self.current_parameters.copy()
            
            # 获取最近的反馈数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            recent_feedback = self.feedback_collector.get_feedback_by_date_range(start_time, end_time)
            
            if not recent_feedback:
                return None
            
            # 分析反馈模式
            false_positives = [f for f in recent_feedback if f.feedback_type == FeedbackType.FALSE_POSITIVE]
            missed_detections = [f for f in recent_feedback if f.feedback_type == FeedbackType.MISSING_DETECTION]
            
            # 调整阈值
            thresholds = new_params['detection_thresholds']
            
            # 如果误报太多，提高阈值
            if len(false_positives) > len(recent_feedback) * 0.3:
                for key in thresholds:
                    thresholds[key] = min(0.9, thresholds[key] + 0.05)
            
            # 如果漏检太多，降低阈值
            elif len(missed_detections) > len(recent_feedback) * 0.2:
                for key in thresholds:
                    thresholds[key] = max(0.1, thresholds[key] - 0.05)
            
            new_params['detection_thresholds'] = thresholds
            return new_params
            
        except Exception as e:
            logger.error(f"阈值调整失败: {e}")
            return None
    
    def _tune_parameters(self, trigger: LearningTrigger) -> Optional[Dict[str, Any]]:
        """参数调优策略"""
        try:
            new_params = self.current_parameters.copy()
            
            # 简化的参数调优
            model_params = new_params['model_parameters']
            
            # 基于历史性能调整参数
            if trigger == LearningTrigger.PERFORMANCE_DEGRADATION:
                # 性能退化时，尝试提高模型敏感性
                model_params['confidence_threshold'] = max(0.1, model_params['confidence_threshold'] - 0.1)
                model_params['attention_temperature'] = min(2.0, model_params['attention_temperature'] + 0.2)
            
            elif trigger == LearningTrigger.DATA_DRIFT_DETECTED:
                # 数据漂移时，调整模型适应性
                model_params['attention_temperature'] = min(2.0, model_params['attention_temperature'] + 0.1)
            
            new_params['model_parameters'] = model_params
            return new_params
            
        except Exception as e:
            logger.error(f"参数调优失败: {e}")
            return None
    
    def _adjust_feature_weights(self, trigger: LearningTrigger) -> Optional[Dict[str, Any]]:
        """特征权重调整策略"""
        try:
            new_params = self.current_parameters.copy()
            
            # 获取反馈中的特征重要性信息
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            recent_feedback = self.feedback_collector.get_feedback_by_date_range(start_time, end_time)
            
            feature_weights = new_params['feature_weights']
            
            # 基于反馈调整特征权重
            feedback_with_features = [f for f in recent_feedback if f.feature_feedback]
            
            if feedback_with_features:
                # 分析哪些特征在错误预测中权重过高
                for feedback in feedback_with_features:
                    if not feedback.is_correct and feedback.feature_feedback:
                        for feature_type, importance in feedback.feature_feedback.items():
                            if feature_type in feature_weights and importance > 0.8:
                                # 降低导致错误的高权重特征
                                feature_weights[feature_type] *= 0.9
            
            # 归一化权重
            total_weight = sum(feature_weights.values())
            if total_weight > 0:
                for key in feature_weights:
                    feature_weights[key] /= total_weight
                    feature_weights[key] *= 4.0  # 保持总权重为4
            
            new_params['feature_weights'] = feature_weights
            return new_params
            
        except Exception as e:
            logger.error(f"特征权重调整失败: {e}")
            return None
    
    def _optimize_ensemble_weights(self, trigger: LearningTrigger) -> Optional[Dict[str, Any]]:
        """集成权重优化策略"""
        try:
            new_params = self.current_parameters.copy()
            
            # 获取各检测器的最近性能数据
            # 这里需要实际的检测器性能数据，暂时使用模拟逻辑
            
            ensemble_weights = new_params['ensemble_weights']
            
            # 简化的权重调整逻辑
            # 基于历史性能调整各检测器权重
            
            if trigger == LearningTrigger.PERFORMANCE_DEGRADATION:
                # 性能退化时，增加语义检测器权重
                ensemble_weights['semantic_detector'] = min(0.4, ensemble_weights['semantic_detector'] + 0.05)
                
                # 重新归一化
                total = sum(ensemble_weights.values())
                for key in ensemble_weights:
                    ensemble_weights[key] /= total
            
            new_params['ensemble_weights'] = ensemble_weights
            return new_params
            
        except Exception as e:
            logger.error(f"集成权重优化失败: {e}")
            return None
    
    def _evaluate_current_performance(self) -> Dict[str, float]:
        """评估当前性能"""
        try:
            # 获取最近的性能指标
            current_metrics = self.model_monitor.get_current_metrics()
            
            performance = {}
            
            # 提取关键性能指标
            if current_metrics.get('metrics'):
                for metric_name in ['accuracy', 'f1_score', 'precision', 'recall']:
                    metric_data = current_metrics['metrics'].get(metric_name)
                    if metric_data:
                        performance[metric_name] = metric_data.get('value', 0.0)
            
            # 如果没有足够的指标，使用默认值
            if not performance:
                performance = {
                    'accuracy': 0.8,
                    'f1_score': 0.75,
                    'precision': 0.8,
                    'recall': 0.7
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"评估当前性能失败: {e}")
            return {}
    
    def _calculate_improvement(self, before: Dict[str, float], 
                             after: Dict[str, float]) -> float:
        """计算改进程度"""
        try:
            if not before or not after:
                return 0.0
            
            total_improvement = 0.0
            count = 0
            
            for objective in self.learning_objectives:
                metric_name = objective.metric_name
                
                if metric_name in before and metric_name in after:
                    before_value = before[metric_name]
                    after_value = after[metric_name]
                    
                    if objective.maximize:
                        improvement = (after_value - before_value) / max(before_value, 0.001)
                    else:
                        improvement = (before_value - after_value) / max(before_value, 0.001)
                    
                    weighted_improvement = improvement * objective.weight
                    total_improvement += weighted_improvement
                    count += 1
            
            return total_improvement / max(count, 1)
            
        except Exception as e:
            logger.error(f"计算改进程度失败: {e}")
            return 0.0
    
    def _calculate_confidence(self, improvement: float, 
                            before: Dict[str, float], 
                            after: Dict[str, float]) -> float:
        """计算置信度"""
        try:
            # 基于改进幅度和指标稳定性计算置信度
            base_confidence = min(1.0, max(0.0, improvement + 0.5))
            
            # 如果所有指标都改进，增加置信度
            all_improved = True
            for objective in self.learning_objectives:
                metric_name = objective.metric_name
                if metric_name in before and metric_name in after:
                    if objective.maximize:
                        if after[metric_name] <= before[metric_name]:
                            all_improved = False
                            break
                    else:
                        if after[metric_name] >= before[metric_name]:
                            all_improved = False
                            break
            
            if all_improved:
                base_confidence *= 1.2
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def _apply_adaptation_result(self, result: AdaptationResult):
        """应用自适应结果"""
        try:
            if result.confidence < 0.6:
                logger.warning(f"置信度过低，不应用结果: {result.confidence:.3f}")
                return
            
            # 保存回滚信息
            result.rollback_info = {
                'previous_parameters': self.current_parameters.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 应用新参数
            self.current_parameters = result.new_parameters.copy()
            result.applied = True
            
            # 保存参数
            self._save_current_parameters()
            
            # 记录学习历史
            self.learning_history.append(result)
            
            # 更新参数历史
            for param_name, param_value in result.new_parameters.items():
                self.parameter_history[param_name].append((result.timestamp, param_value))
            
            # 保存学习历史
            self._save_learning_history()
            
            logger.info(f"自适应结果已应用: {result.strategy.value}")
            
        except Exception as e:
            logger.error(f"应用自适应结果失败: {e}")
    
    def _save_learning_history(self):
        """保存学习历史"""
        try:
            history_file = self.data_dir / "learning_history.json"
            
            # 转换为可序列化格式
            serializable_history = []
            for result in self.learning_history:
                result_dict = asdict(result)
                result_dict['timestamp'] = result.timestamp.isoformat()
                result_dict['strategy'] = result.strategy.value
                result_dict['trigger'] = result.trigger.value
                serializable_history.append(result_dict)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存学习历史失败: {e}")
    
    def rollback_last_adaptation(self) -> bool:
        """回滚最后一次自适应"""
        try:
            if not self.learning_history:
                logger.warning("没有可回滚的自适应历史")
                return False
            
            last_result = self.learning_history[-1]
            
            if not last_result.applied or not last_result.rollback_info:
                logger.warning("最后一次自适应无法回滚")
                return False
            
            # 恢复参数
            self.current_parameters = last_result.rollback_info['previous_parameters']
            
            # 标记为已回滚
            last_result.applied = False
            
            # 保存状态
            self._save_current_parameters()
            self._save_learning_history()
            
            logger.info("成功回滚最后一次自适应")
            return True
            
        except Exception as e:
            logger.error(f"回滚自适应失败: {e}")
            return False
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        try:
            recent_adaptations = [
                r for r in self.learning_history
                if (datetime.now() - r.timestamp).days <= 30
            ]
            
            return {
                'learning_active': self.learning_active,
                'total_adaptations': len(self.learning_history),
                'recent_adaptations': len(recent_adaptations),
                'current_parameters': self.current_parameters,
                'performance_baseline': self.performance_baseline,
                'learning_objectives': [asdict(obj) for obj in self.learning_objectives],
                'last_adaptation': self.learning_history[-1].timestamp.isoformat() if self.learning_history else None,
                'average_improvement': np.mean([r.improvement for r in recent_adaptations]) if recent_adaptations else 0.0
            }
            
        except Exception as e:
            logger.error(f"获取学习摘要失败: {e}")
            return {'error': str(e)}


# 全局自适应学习器实例
_global_adaptive_learner = None


def get_adaptive_learner() -> AdaptiveLearner:
    """
    获取全局自适应学习器实例
    
    Returns:
        AdaptiveLearner: 学习器实例
    """
    global _global_adaptive_learner
    
    if _global_adaptive_learner is None:
        _global_adaptive_learner = AdaptiveLearner()
    
    return _global_adaptive_learner


if __name__ == "__main__":
    # 使用示例
    print("自适应学习器测试:")
    
    # 创建学习器
    learner = AdaptiveLearner()
    
    # 开始自适应学习
    learner.start_adaptive_learning(check_interval=60.0)
    
    # 模拟触发学习
    result = learner._try_adaptation_strategy(
        AdaptationStrategy.THRESHOLD_ADJUSTMENT,
        LearningTrigger.FEEDBACK_ACCUMULATION
    )
    
    if result:
        print(f"自适应结果: 改进 {result.improvement:.4f}, 置信度 {result.confidence:.3f}")
    
    # 获取摘要
    summary = learner.get_learning_summary()
    print(f"学习摘要: {summary}")
    
    # 停止学习
    learner.stop_adaptive_learning() 
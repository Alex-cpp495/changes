"""
特征重要性分析器
分析影响异常检测结果的关键特征，提供可解释的特征贡献度分析
"""

import json
import math
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from collections import defaultdict

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)

# 尝试导入科学计算库
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy不可用，将使用Python内置实现")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("绘图库不可用，将跳过可视化功能")


@dataclass
class FeatureImportance:
    """特征重要性数据结构"""
    feature_name: str
    importance_score: float
    importance_type: str  # "positive", "negative", "neutral"
    confidence: float = 1.0
    explanation: Optional[str] = None
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None


@dataclass
class ImportanceAnalysisResult:
    """重要性分析结果"""
    feature_importances: List[FeatureImportance]
    global_explanation: str
    analysis_method: str
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: str


class BaseImportanceAnalyzer(ABC):
    """特征重要性分析器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze_importance(self, features: Dict[str, Any], 
                         prediction: Any, 
                         model: Any) -> List[FeatureImportance]:
        """分析特征重要性"""
        pass


class ShapleyValueAnalyzer(BaseImportanceAnalyzer):
    """Shapley值分析器（简化实现）"""
    
    def __init__(self, num_samples: int = 100):
        super().__init__("shapley_value")
        self.num_samples = num_samples
    
    def analyze_importance(self, features: Dict[str, Any], 
                         prediction: Any, 
                         model: Any) -> List[FeatureImportance]:
        """计算Shapley值"""
        try:
            feature_names = list(features.keys())
            importances = []
            
            # 获取基准预测（所有特征的平均值）
            baseline_features = self._get_baseline_features(features)
            baseline_prediction = self._get_model_prediction(model, baseline_features)
            
            for feature_name in feature_names:
                shapley_value = self._compute_shapley_value(
                    feature_name, features, model, baseline_prediction
                )
                
                importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=abs(shapley_value),
                    importance_type="positive" if shapley_value > 0 else "negative",
                    confidence=0.8,  # 简化的置信度
                    explanation=f"Shapley值: {shapley_value:.4f}",
                    raw_value=shapley_value
                )
                
                importances.append(importance)
            
            return importances
            
        except Exception as e:
            logger.error(f"Shapley值分析失败: {e}")
            return []
    
    def _get_baseline_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """获取基准特征值（简化为使用中位数/平均值）"""
        baseline = {}
        
        for name, value in features.items():
            if isinstance(value, (int, float)):
                baseline[name] = value * 0.5  # 简化的基准值
            elif isinstance(value, str):
                baseline[name] = ""  # 空字符串作为基准
            elif isinstance(value, list):
                baseline[name] = []  # 空列表作为基准
            else:
                baseline[name] = None
        
        return baseline
    
    def _compute_shapley_value(self, target_feature: str, 
                             features: Dict[str, Any], 
                             model: Any,
                             baseline_prediction: Any) -> float:
        """计算单个特征的Shapley值"""
        try:
            feature_names = list(features.keys())
            target_idx = feature_names.index(target_feature)
            
            shapley_value = 0.0
            
            # 蒙特卡洛采样估计Shapley值
            for _ in range(self.num_samples):
                # 随机选择一个子集（不包含目标特征）
                other_features = [f for f in feature_names if f != target_feature]
                random.shuffle(other_features)
                
                # 随机选择子集大小
                subset_size = random.randint(0, len(other_features))
                subset = other_features[:subset_size]
                
                # 计算边际贡献
                marginal_contribution = self._compute_marginal_contribution(
                    target_feature, subset, features, model, baseline_prediction
                )
                
                shapley_value += marginal_contribution
            
            return shapley_value / self.num_samples
            
        except Exception as e:
            logger.error(f"Shapley值计算失败: {e}")
            return 0.0
    
    def _compute_marginal_contribution(self, target_feature: str,
                                     subset: List[str],
                                     features: Dict[str, Any],
                                     model: Any,
                                     baseline_prediction: Any) -> float:
        """计算边际贡献"""
        try:
            baseline_features = self._get_baseline_features(features)
            
            # 不包含目标特征的预测
            subset_features = baseline_features.copy()
            for feature in subset:
                subset_features[feature] = features[feature]
            
            pred_without_target = self._get_model_prediction(model, subset_features)
            
            # 包含目标特征的预测
            subset_with_target = subset_features.copy()
            subset_with_target[target_feature] = features[target_feature]
            
            pred_with_target = self._get_model_prediction(model, subset_with_target)
            
            # 计算边际贡献
            marginal_contribution = self._prediction_difference(pred_with_target, pred_without_target)
            
            return marginal_contribution
            
        except Exception as e:
            logger.error(f"边际贡献计算失败: {e}")
            return 0.0
    
    def _get_model_prediction(self, model: Any, features: Dict[str, Any]) -> Any:
        """获取模型预测（需要根据具体模型实现）"""
        try:
            if hasattr(model, 'predict'):
                # sklearn风格的模型
                return model.predict([list(features.values())])[0]
            elif hasattr(model, 'detect_anomalies'):
                # 异常检测模型
                return model.detect_anomalies(features)
            else:
                # 简化的预测
                return sum(v if isinstance(v, (int, float)) else 0 for v in features.values())
                
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return 0.0
    
    def _prediction_difference(self, pred1: Any, pred2: Any) -> float:
        """计算预测差异"""
        try:
            if isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
                return float(pred1 - pred2)
            elif isinstance(pred1, dict) and isinstance(pred2, dict):
                # 异常检测结果是字典的情况
                score1 = pred1.get('overall_anomaly_score', 0.0)
                score2 = pred2.get('overall_anomaly_score', 0.0)
                return float(score1 - score2)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"预测差异计算失败: {e}")
            return 0.0


class PermutationImportanceAnalyzer(BaseImportanceAnalyzer):
    """排列重要性分析器"""
    
    def __init__(self, num_permutations: int = 10):
        super().__init__("permutation_importance")
        self.num_permutations = num_permutations
    
    def analyze_importance(self, features: Dict[str, Any], 
                         prediction: Any, 
                         model: Any) -> List[FeatureImportance]:
        """计算排列重要性"""
        try:
            # 获取基准预测
            baseline_prediction = self._get_model_prediction(model, features)
            baseline_score = self._extract_score(baseline_prediction)
            
            importances = []
            
            for feature_name in features.keys():
                importance_scores = []
                
                # 多次排列该特征
                for _ in range(self.num_permutations):
                    permuted_features = features.copy()
                    permuted_features[feature_name] = self._permute_feature(features[feature_name])
                    
                    permuted_prediction = self._get_model_prediction(model, permuted_features)
                    permuted_score = self._extract_score(permuted_prediction)
                    
                    # 计算性能下降
                    importance = baseline_score - permuted_score
                    importance_scores.append(importance)
                
                # 计算平均重要性
                avg_importance = sum(importance_scores) / len(importance_scores)
                std_importance = self._calculate_std(importance_scores)
                
                importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=abs(avg_importance),
                    importance_type="positive" if avg_importance > 0 else "negative",
                    confidence=max(0.1, 1.0 - std_importance / (abs(avg_importance) + 1e-6)),
                    explanation=f"排列重要性: {avg_importance:.4f} ± {std_importance:.4f}",
                    raw_value=avg_importance
                )
                
                importances.append(importance)
            
            return importances
            
        except Exception as e:
            logger.error(f"排列重要性分析失败: {e}")
            return []
    
    def _permute_feature(self, feature_value: Any) -> Any:
        """排列特征值"""
        if isinstance(feature_value, (int, float)):
            # 数值特征：添加随机噪声
            noise_scale = abs(feature_value) * 0.1 + 0.01
            return feature_value + random.gauss(0, noise_scale)
        elif isinstance(feature_value, str):
            # 字符串特征：随机打乱字符或使用随机字符串
            if len(feature_value) > 1:
                chars = list(feature_value)
                random.shuffle(chars)
                return ''.join(chars)
            else:
                return '随机'
        elif isinstance(feature_value, list):
            # 列表特征：随机打乱
            if feature_value:
                shuffled = feature_value.copy()
                random.shuffle(shuffled)
                return shuffled
            else:
                return []
        else:
            return feature_value
    
    def _extract_score(self, prediction: Any) -> float:
        """从预测结果中提取分数"""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            return prediction.get('overall_anomaly_score', 0.0)
        else:
            return 0.0
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _get_model_prediction(self, model: Any, features: Dict[str, Any]) -> Any:
        """获取模型预测"""
        try:
            if hasattr(model, 'detect_anomalies'):
                return model.detect_anomalies(features)
            elif hasattr(model, 'predict'):
                return model.predict([list(features.values())])[0]
            else:
                return sum(v if isinstance(v, (int, float)) else 0 for v in features.values())
                
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return 0.0


class GradientBasedAnalyzer(BaseImportanceAnalyzer):
    """基于梯度的重要性分析器"""
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__("gradient_based")
        self.epsilon = epsilon
    
    def analyze_importance(self, features: Dict[str, Any], 
                         prediction: Any, 
                         model: Any) -> List[FeatureImportance]:
        """计算基于梯度的重要性"""
        try:
            baseline_prediction = self._get_model_prediction(model, features)
            baseline_score = self._extract_score(baseline_prediction)
            
            importances = []
            
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    # 数值特征：计算数值梯度
                    gradient = self._compute_numerical_gradient(
                        feature_name, feature_value, features, model, baseline_score
                    )
                    
                    importance = FeatureImportance(
                        feature_name=feature_name,
                        importance_score=abs(gradient),
                        importance_type="positive" if gradient > 0 else "negative",
                        confidence=0.7,
                        explanation=f"梯度: {gradient:.6f}",
                        raw_value=gradient
                    )
                    
                    importances.append(importance)
                else:
                    # 非数值特征：使用简化方法
                    importance = self._compute_categorical_importance(
                        feature_name, feature_value, features, model, baseline_score
                    )
                    importances.append(importance)
            
            return importances
            
        except Exception as e:
            logger.error(f"基于梯度的重要性分析失败: {e}")
            return []
    
    def _compute_numerical_gradient(self, feature_name: str, feature_value: float,
                                  features: Dict[str, Any], model: Any, 
                                  baseline_score: float) -> float:
        """计算数值梯度"""
        try:
            # 前向差分
            perturbed_features = features.copy()
            perturbed_features[feature_name] = feature_value + self.epsilon
            
            perturbed_prediction = self._get_model_prediction(model, perturbed_features)
            perturbed_score = self._extract_score(perturbed_prediction)
            
            gradient = (perturbed_score - baseline_score) / self.epsilon
            return gradient
            
        except Exception as e:
            logger.error(f"数值梯度计算失败: {e}")
            return 0.0
    
    def _compute_categorical_importance(self, feature_name: str, feature_value: Any,
                                      features: Dict[str, Any], model: Any,
                                      baseline_score: float) -> FeatureImportance:
        """计算分类特征重要性"""
        try:
            # 移除该特征
            reduced_features = features.copy()
            if isinstance(feature_value, str):
                reduced_features[feature_name] = ""
            elif isinstance(feature_value, list):
                reduced_features[feature_name] = []
            else:
                reduced_features[feature_name] = None
            
            reduced_prediction = self._get_model_prediction(model, reduced_features)
            reduced_score = self._extract_score(reduced_prediction)
            
            importance_score = abs(baseline_score - reduced_score)
            
            return FeatureImportance(
                feature_name=feature_name,
                importance_score=importance_score,
                importance_type="positive" if baseline_score > reduced_score else "negative",
                confidence=0.6,
                explanation=f"移除特征的影响: {baseline_score - reduced_score:.4f}",
                raw_value=baseline_score - reduced_score
            )
            
        except Exception as e:
            logger.error(f"分类特征重要性计算失败: {e}")
            return FeatureImportance(
                feature_name=feature_name,
                importance_score=0.0,
                importance_type="neutral",
                confidence=0.0,
                explanation="计算失败"
            )
    
    def _get_model_prediction(self, model: Any, features: Dict[str, Any]) -> Any:
        """获取模型预测"""
        try:
            if hasattr(model, 'detect_anomalies'):
                return model.detect_anomalies(features)
            elif hasattr(model, 'predict'):
                return model.predict([list(features.values())])[0]
            else:
                return sum(v if isinstance(v, (int, float)) else 0 for v in features.values())
                
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return 0.0
    
    def _extract_score(self, prediction: Any) -> float:
        """从预测结果中提取分数"""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            return prediction.get('overall_anomaly_score', 0.0)
        else:
            return 0.0


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    
    提供多种特征重要性分析方法：
    1. Shapley值分析 - 基于博弈论的公平分配
    2. 排列重要性 - 基于特征排列的重要性评估
    3. 梯度分析 - 基于梯度的敏感性分析
    4. 统计分析 - 基于统计相关性的分析
    5. 集成分析 - 多种方法的综合评估
    6. 可视化展示 - 重要性结果的直观展示
    
    Args:
        config: 分析配置
        
    Attributes:
        analyzers: 分析器字典
        output_dir: 输出目录
        analysis_history: 分析历史
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化特征重要性分析器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 输出目录
        self.output_dir = Path("data/feature_importance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 注册分析器
        self.analyzers: Dict[str, BaseImportanceAnalyzer] = {}
        self._register_analyzers()
        
        # 分析历史
        self.analysis_history: List[ImportanceAnalysisResult] = []
        
        logger.info("特征重要性分析器初始化完成")
    
    def _register_analyzers(self):
        """注册分析器"""
        self.analyzers['shapley'] = ShapleyValueAnalyzer(
            num_samples=self.config.get('shapley_samples', 50)
        )
        self.analyzers['permutation'] = PermutationImportanceAnalyzer(
            num_permutations=self.config.get('permutation_iterations', 10)
        )
        self.analyzers['gradient'] = GradientBasedAnalyzer(
            epsilon=self.config.get('gradient_epsilon', 1e-6)
        )
    
    def analyze_feature_importance(self, features: Dict[str, Any],
                                 prediction: Any,
                                 model: Any,
                                 methods: Optional[List[str]] = None) -> ImportanceAnalysisResult:
        """
        分析特征重要性
        
        Args:
            features: 特征字典
            prediction: 模型预测结果
            model: 模型实例
            methods: 分析方法列表
            
        Returns:
            ImportanceAnalysisResult: 分析结果
        """
        try:
            if methods is None:
                methods = ['permutation', 'gradient']  # 默认使用较快的方法
            
            all_importances = []
            method_results = {}
            
            # 运行不同的分析方法
            for method in methods:
                if method in self.analyzers:
                    logger.info(f"运行{method}分析...")
                    
                    analyzer = self.analyzers[method]
                    importances = analyzer.analyze_importance(features, prediction, model)
                    
                    method_results[method] = importances
                    all_importances.extend(importances)
            
            # 聚合多种方法的结果
            aggregated_importances = self._aggregate_importances(method_results, features.keys())
            
            # 生成全局解释
            global_explanation = self._generate_global_explanation(aggregated_importances, prediction)
            
            # 计算置信度
            confidence_score = self._calculate_confidence(method_results)
            
            # 创建分析结果
            result = ImportanceAnalysisResult(
                feature_importances=aggregated_importances,
                global_explanation=global_explanation,
                analysis_method='+'.join(methods),
                confidence_score=confidence_score,
                metadata={
                    'feature_count': len(features),
                    'methods_used': methods,
                    'prediction': str(prediction),
                    'analysis_time': datetime.now().isoformat()
                },
                timestamp=datetime.now().isoformat()
            )
            
            # 保存结果
            self._save_analysis_result(result)
            
            # 添加到历史
            self.analysis_history.append(result)
            
            logger.info(f"特征重要性分析完成，共分析{len(aggregated_importances)}个特征")
            
            return result
            
        except Exception as e:
            logger.error(f"特征重要性分析失败: {e}")
            
            # 返回空结果
            return ImportanceAnalysisResult(
                feature_importances=[],
                global_explanation=f"分析失败: {str(e)}",
                analysis_method="error",
                confidence_score=0.0,
                metadata={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    def _aggregate_importances(self, method_results: Dict[str, List[FeatureImportance]],
                             feature_names: List[str]) -> List[FeatureImportance]:
        """聚合多种方法的重要性结果"""
        aggregated = []
        
        for feature_name in feature_names:
            # 收集该特征在各方法中的重要性
            feature_scores = []
            feature_explanations = []
            
            for method, importances in method_results.items():
                for importance in importances:
                    if importance.feature_name == feature_name:
                        feature_scores.append(importance.importance_score)
                        feature_explanations.append(f"{method}: {importance.explanation}")
                        break
            
            if feature_scores:
                # 计算平均重要性
                avg_score = sum(feature_scores) / len(feature_scores)
                
                # 计算置信度（基于方法间的一致性）
                if len(feature_scores) > 1:
                    std_dev = math.sqrt(sum((s - avg_score) ** 2 for s in feature_scores) / (len(feature_scores) - 1))
                    confidence = max(0.1, 1.0 - std_dev / (avg_score + 1e-6))
                else:
                    confidence = 0.7
                
                # 确定重要性类型
                positive_count = sum(1 for s in feature_scores if s > 0)
                importance_type = "positive" if positive_count > len(feature_scores) / 2 else "negative"
                
                aggregated_importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=avg_score,
                    importance_type=importance_type,
                    confidence=confidence,
                    explanation="; ".join(feature_explanations),
                    raw_value=avg_score
                )
                
                aggregated.append(aggregated_importance)
        
        # 按重要性排序
        aggregated.sort(key=lambda x: x.importance_score, reverse=True)
        
        return aggregated
    
    def _generate_global_explanation(self, importances: List[FeatureImportance], 
                                   prediction: Any) -> str:
        """生成全局解释"""
        try:
            if not importances:
                return "未找到显著的特征重要性。"
            
            # 提取预测信息
            if isinstance(prediction, dict):
                anomaly_score = prediction.get('overall_anomaly_score', 0.0)
                anomaly_level = prediction.get('overall_anomaly_level', 'UNKNOWN')
            else:
                anomaly_score = float(prediction) if isinstance(prediction, (int, float)) else 0.0
                anomaly_level = 'HIGH' if anomaly_score > 0.7 else 'MEDIUM' if anomaly_score > 0.3 else 'LOW'
            
            # 找出最重要的特征
            top_features = importances[:3]  # 前3个最重要的特征
            
            explanation_parts = [
                f"模型预测异常等级为 {anomaly_level}（分数: {anomaly_score:.3f}）。"
            ]
            
            if top_features:
                explanation_parts.append("主要影响因素：")
                
                for i, importance in enumerate(top_features, 1):
                    impact_type = "正面影响" if importance.importance_type == "positive" else "负面影响"
                    explanation_parts.append(
                        f"{i}. {importance.feature_name}: {impact_type}，"
                        f"重要性分数 {importance.importance_score:.3f}"
                    )
            
            # 添加置信度信息
            avg_confidence = sum(imp.confidence for imp in importances) / len(importances)
            explanation_parts.append(f"分析置信度: {avg_confidence:.2f}")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"全局解释生成失败: {e}")
            return f"全局解释生成失败: {str(e)}"
    
    def _calculate_confidence(self, method_results: Dict[str, List[FeatureImportance]]) -> float:
        """计算整体置信度"""
        try:
            if not method_results:
                return 0.0
            
            all_confidences = []
            for importances in method_results.values():
                for importance in importances:
                    all_confidences.append(importance.confidence)
            
            if all_confidences:
                return sum(all_confidences) / len(all_confidences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.0
    
    def _save_analysis_result(self, result: ImportanceAnalysisResult):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_importance_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # 转换为可序列化的格式
            result_dict = {
                'feature_importances': [
                    {
                        'feature_name': imp.feature_name,
                        'importance_score': imp.importance_score,
                        'importance_type': imp.importance_type,
                        'confidence': imp.confidence,
                        'explanation': imp.explanation,
                        'raw_value': imp.raw_value
                    }
                    for imp in result.feature_importances
                ],
                'global_explanation': result.global_explanation,
                'analysis_method': result.analysis_method,
                'confidence_score': result.confidence_score,
                'metadata': result.metadata,
                'timestamp': result.timestamp
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def create_importance_visualization(self, result: ImportanceAnalysisResult) -> str:
        """创建重要性可视化图表"""
        try:
            if not PLOTTING_AVAILABLE:
                return self._create_text_visualization(result)
            
            importances = result.feature_importances[:10]  # 显示前10个
            
            if not importances:
                return ""
            
            # 准备数据
            feature_names = [imp.feature_name for imp in importances]
            scores = [imp.importance_score for imp in importances]
            colors = ['red' if imp.importance_type == 'negative' else 'blue' for imp in importances]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(feature_names, scores, color=colors, alpha=0.7)
            
            # 设置标签和标题
            ax.set_xlabel('重要性分数', fontsize=12)
            ax.set_title('特征重要性分析', fontsize=14)
            
            # 添加数值标注
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontsize=10)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='正面影响'),
                Patch(facecolor='red', alpha=0.7, label='负面影响')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_importance_plot_{timestamp}.png"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"重要性可视化图表已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"重要性可视化失败: {e}")
            return ""
    
    def _create_text_visualization(self, result: ImportanceAnalysisResult) -> str:
        """创建文本版本的可视化"""
        try:
            lines = [
                "特征重要性分析结果",
                "=" * 40,
                f"分析方法: {result.analysis_method}",
                f"置信度: {result.confidence_score:.3f}",
                "",
                "特征重要性排序:"
            ]
            
            for i, importance in enumerate(result.feature_importances[:10], 1):
                impact_symbol = "+" if importance.importance_type == "positive" else "-"
                lines.append(
                    f"{i:2d}. {importance.feature_name:<20} "
                    f"{impact_symbol} {importance.importance_score:.4f} "
                    f"({importance.confidence:.2f})"
                )
            
            lines.extend([
                "",
                "全局解释:",
                result.global_explanation
            ])
            
            # 保存文本报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_importance_report_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"文本重要性报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"文本可视化失败: {e}")
            return ""
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            'total_analyses': len(self.analysis_history),
            'available_methods': list(self.analyzers.keys()),
            'output_directory': str(self.output_dir),
            'config': self.config,
            'recent_analyses': [
                {
                    'timestamp': result.timestamp,
                    'method': result.analysis_method,
                    'feature_count': len(result.feature_importances),
                    'confidence': result.confidence_score
                }
                for result in self.analysis_history[-5:]
            ],
            'plotting_available': PLOTTING_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        }


# 工厂函数
def create_feature_importance_analyzer(config: Optional[Dict[str, Any]] = None) -> FeatureImportanceAnalyzer:
    """
    创建特征重要性分析器实例
    
    Args:
        config: 配置字典
        
    Returns:
        FeatureImportanceAnalyzer: 分析器实例
    """
    return FeatureImportanceAnalyzer(config)


if __name__ == "__main__":
    # 使用示例
    print("特征重要性分析器测试:")
    
    # 创建分析器
    analyzer = FeatureImportanceAnalyzer()
    
    # 模拟特征数据
    features = {
        'content_length': 1500,
        'sentiment_score': 0.8,
        'readability_score': 0.6,
        'technical_terms_count': 25,
        'financial_data_mentions': 10,
        'author_credibility': 0.9,
        'publication_timing': 'market_hours'
    }
    
    # 模拟预测结果
    prediction = {
        'overall_anomaly_score': 0.75,
        'overall_anomaly_level': 'HIGH'
    }
    
    # 模拟模型
    class MockModel:
        def detect_anomalies(self, features):
            # 简化的异常检测逻辑
            score = 0.0
            if features.get('content_length', 0) < 500:
                score += 0.3
            if features.get('sentiment_score', 0) > 0.9:
                score += 0.2
            if features.get('technical_terms_count', 0) < 5:
                score += 0.4
            
            return {
                'overall_anomaly_score': min(score, 1.0),
                'overall_anomaly_level': 'HIGH' if score > 0.7 else 'MEDIUM' if score > 0.3 else 'LOW'
            }
    
    model = MockModel()
    
    print(f"分析特征: {list(features.keys())}")
    print(f"预测结果: {prediction}")
    
    # 进行重要性分析
    result = analyzer.analyze_feature_importance(
        features=features,
        prediction=prediction,
        model=model,
        methods=['permutation', 'gradient']
    )
    
    print(f"\n分析完成:")
    print(f"置信度: {result.confidence_score:.3f}")
    print(f"全局解释: {result.global_explanation}")
    
    print(f"\n前5个重要特征:")
    for i, importance in enumerate(result.feature_importances[:5], 1):
        print(f"{i}. {importance.feature_name}: {importance.importance_score:.4f} "
              f"({importance.importance_type})")
    
    # 创建可视化
    vis_file = analyzer.create_importance_visualization(result)
    if vis_file:
        print(f"\n可视化已保存: {vis_file}")
    
    # 获取摘要
    summary = analyzer.get_analysis_summary()
    print(f"\n分析摘要: {summary}") 
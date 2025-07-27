"""
反事实生成器
生成反事实样本和假设场景，帮助解释模型决策和理解决策边界
"""

import json
import math
import random
import copy
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

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


class CounterfactualType(Enum):
    """反事实类型枚举"""
    MINIMAL_CHANGE = "minimal_change"      # 最小变化
    TARGETED_OUTCOME = "targeted_outcome"  # 目标结果
    DIVERSE_SET = "diverse_set"           # 多样化集合
    ROBUST_ANALYSIS = "robust_analysis"    # 鲁棒性分析


@dataclass
class CounterfactualExample:
    """反事实样本数据结构"""
    original_features: Dict[str, Any]
    counterfactual_features: Dict[str, Any]
    original_prediction: Any
    counterfactual_prediction: Any
    
    # 变化信息
    changed_features: List[str]
    feature_changes: Dict[str, Dict[str, Any]]  # {feature: {old_value, new_value, change_type}}
    
    # 质量指标
    distance: float = 0.0
    validity: float = 1.0
    plausibility: float = 1.0
    
    # 解释信息
    explanation: str = ""
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CounterfactualSet:
    """反事实集合"""
    examples: List[CounterfactualExample]
    generation_method: str
    diversity_score: float
    coverage_score: float
    quality_metrics: Dict[str, float]
    global_explanation: str
    timestamp: str


class BaseCounterfactualGenerator(ABC):
    """反事实生成器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_counterfactuals(self, features: Dict[str, Any], 
                               prediction: Any, 
                               model: Any,
                               target_outcome: Optional[Any] = None,
                               num_examples: int = 5) -> List[CounterfactualExample]:
        """生成反事实样本"""
        pass


class MinimalChangeGenerator(BaseCounterfactualGenerator):
    """最小变化反事实生成器"""
    
    def __init__(self, max_changes: int = 3, max_iterations: int = 100):
        super().__init__("minimal_change")
        self.max_changes = max_changes
        self.max_iterations = max_iterations
    
    def generate_counterfactuals(self, features: Dict[str, Any], 
                               prediction: Any, 
                               model: Any,
                               target_outcome: Optional[Any] = None,
                               num_examples: int = 5) -> List[CounterfactualExample]:
        """生成最小变化的反事实样本"""
        try:
            counterfactuals = []
            original_score = self._extract_score(prediction)
            
            # 确定目标
            if target_outcome is None:
                target_score = 1.0 - original_score if original_score > 0.5 else original_score + 0.3
            else:
                target_score = self._extract_score(target_outcome)
            
            for _ in range(num_examples):
                counterfactual = self._generate_single_minimal_change(
                    features, prediction, model, target_score
                )
                if counterfactual:
                    counterfactuals.append(counterfactual)
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"最小变化反事实生成失败: {e}")
            return []
    
    def _generate_single_minimal_change(self, features: Dict[str, Any],
                                      original_prediction: Any,
                                      model: Any,
                                      target_score: float) -> Optional[CounterfactualExample]:
        """生成单个最小变化反事实"""
        try:
            current_features = features.copy()
            changed_features = []
            feature_changes = {}
            
            original_score = self._extract_score(original_prediction)
            best_distance = float('inf')
            best_counterfactual = None
            
            for iteration in range(self.max_iterations):
                if len(changed_features) >= self.max_changes:
                    break
                
                # 选择一个尚未修改的特征
                unchanged_features = [f for f in features.keys() if f not in changed_features]
                if not unchanged_features:
                    break
                
                feature_to_change = random.choice(unchanged_features)
                
                # 生成变化
                new_value = self._generate_feature_change(
                    feature_to_change, current_features[feature_to_change]
                )
                
                # 应用变化
                old_value = current_features[feature_to_change]
                current_features[feature_to_change] = new_value
                
                # 评估新的预测
                new_prediction = self._get_model_prediction(model, current_features)
                new_score = self._extract_score(new_prediction)
                
                # 检查是否达到目标
                if abs(new_score - target_score) < abs(original_score - target_score):
                    changed_features.append(feature_to_change)
                    feature_changes[feature_to_change] = {
                        'old_value': old_value,
                        'new_value': new_value,
                        'change_type': self._get_change_type(old_value, new_value)
                    }
                    
                    # 计算距离
                    distance = self._calculate_distance(features, current_features)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_counterfactual = CounterfactualExample(
                            original_features=features.copy(),
                            counterfactual_features=current_features.copy(),
                            original_prediction=original_prediction,
                            counterfactual_prediction=new_prediction,
                            changed_features=changed_features.copy(),
                            feature_changes=feature_changes.copy(),
                            distance=distance,
                            validity=1.0,
                            plausibility=self._calculate_plausibility(features, current_features),
                            explanation=self._generate_explanation(feature_changes, original_score, new_score),
                            confidence=0.8
                        )
                    
                    # 如果足够接近目标，可以提前停止
                    if abs(new_score - target_score) < 0.1:
                        break
                else:
                    # 撤销变化
                    current_features[feature_to_change] = old_value
            
            return best_counterfactual
            
        except Exception as e:
            logger.error(f"单个最小变化反事实生成失败: {e}")
            return None
    
    def _generate_feature_change(self, feature_name: str, original_value: Any) -> Any:
        """生成特征变化"""
        if isinstance(original_value, (int, float)):
            # 数值特征：小幅随机变化
            change_factor = random.uniform(0.8, 1.2)
            return original_value * change_factor
        elif isinstance(original_value, str):
            # 字符串特征：基于规则的变化
            return self._modify_text_feature(original_value)
        elif isinstance(original_value, list):
            # 列表特征：添加或删除元素
            if original_value and random.random() > 0.5:
                # 删除元素
                modified = original_value.copy()
                modified.pop(random.randint(0, len(modified) - 1))
                return modified
            else:
                # 添加元素（简化）
                modified = original_value.copy()
                modified.append(f"新元素_{random.randint(1, 100)}")
                return modified
        else:
            return original_value
    
    def _modify_text_feature(self, text: str) -> str:
        """修改文本特征"""
        if not text:
            return "修改后的文本"
        
        modifications = [
            lambda t: t + "（修改版）",
            lambda t: t.replace("好", "不错") if "好" in t else t,
            lambda t: t.replace("高", "低") if "高" in t else t,
            lambda t: t[:len(t)//2] if len(t) > 10 else t + "_修改",
        ]
        
        modification = random.choice(modifications)
        return modification(text)
    
    def _calculate_distance(self, original: Dict[str, Any], 
                          counterfactual: Dict[str, Any]) -> float:
        """计算特征空间中的距离"""
        distance = 0.0
        
        for feature in original.keys():
            if feature in counterfactual:
                feature_distance = self._feature_distance(
                    original[feature], counterfactual[feature]
                )
                distance += feature_distance
        
        return distance
    
    def _feature_distance(self, val1: Any, val2: Any) -> float:
        """计算单个特征的距离"""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-6)
        elif isinstance(val1, str) and isinstance(val2, str):
            # 简化的字符串距离
            return 1.0 if val1 != val2 else 0.0
        elif isinstance(val1, list) and isinstance(val2, list):
            # 简化的列表距离
            return abs(len(val1) - len(val2)) / (len(val1) + len(val2) + 1)
        else:
            return 1.0 if val1 != val2 else 0.0
    
    def _calculate_plausibility(self, original: Dict[str, Any], 
                              counterfactual: Dict[str, Any]) -> float:
        """计算合理性分数"""
        # 简化的合理性评估
        changes = 0
        total_features = len(original)
        
        for feature in original.keys():
            if feature in counterfactual and original[feature] != counterfactual[feature]:
                changes += 1
        
        # 变化越少，合理性越高
        plausibility = 1.0 - (changes / total_features)
        return max(0.1, plausibility)
    
    def _get_change_type(self, old_value: Any, new_value: Any) -> str:
        """获取变化类型"""
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if new_value > old_value:
                return "increase"
            elif new_value < old_value:
                return "decrease"
            else:
                return "unchanged"
        else:
            return "modification"
    
    def _generate_explanation(self, feature_changes: Dict[str, Dict[str, Any]],
                            original_score: float, new_score: float) -> str:
        """生成解释"""
        if not feature_changes:
            return "无显著变化"
        
        explanation_parts = [
            f"通过修改 {len(feature_changes)} 个特征，"
            f"异常分数从 {original_score:.3f} 变为 {new_score:.3f}。"
        ]
        
        explanation_parts.append("具体变化：")
        for feature, change_info in feature_changes.items():
            change_type = change_info['change_type']
            explanation_parts.append(
                f"- {feature}: {change_info['old_value']} → {change_info['new_value']} ({change_type})"
            )
        
        return " ".join(explanation_parts)
    
    def _extract_score(self, prediction: Any) -> float:
        """从预测结果中提取分数"""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            return prediction.get('overall_anomaly_score', 0.0)
        else:
            return 0.0
    
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


class DiverseSetGenerator(BaseCounterfactualGenerator):
    """多样化反事实集合生成器"""
    
    def __init__(self, diversity_threshold: float = 0.3):
        super().__init__("diverse_set")
        self.diversity_threshold = diversity_threshold
    
    def generate_counterfactuals(self, features: Dict[str, Any], 
                               prediction: Any, 
                               model: Any,
                               target_outcome: Optional[Any] = None,
                               num_examples: int = 5) -> List[CounterfactualExample]:
        """生成多样化的反事实集合"""
        try:
            counterfactuals = []
            original_score = self._extract_score(prediction)
            
            # 生成多个候选反事实
            candidates = []
            for _ in range(num_examples * 3):  # 生成更多候选
                candidate = self._generate_diverse_candidate(
                    features, prediction, model, original_score
                )
                if candidate:
                    candidates.append(candidate)
            
            # 选择多样化的子集
            if candidates:
                selected = self._select_diverse_subset(candidates, num_examples)
                counterfactuals.extend(selected)
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"多样化反事实生成失败: {e}")
            return []
    
    def _generate_diverse_candidate(self, features: Dict[str, Any],
                                  original_prediction: Any,
                                  model: Any,
                                  original_score: float) -> Optional[CounterfactualExample]:
        """生成多样化候选"""
        try:
            # 随机选择要修改的特征数量
            num_changes = random.randint(1, min(len(features), 5))
            
            # 随机选择要修改的特征
            features_to_change = random.sample(list(features.keys()), num_changes)
            
            # 生成变化
            counterfactual_features = features.copy()
            feature_changes = {}
            
            for feature in features_to_change:
                old_value = counterfactual_features[feature]
                new_value = self._generate_diverse_change(feature, old_value)
                counterfactual_features[feature] = new_value
                
                feature_changes[feature] = {
                    'old_value': old_value,
                    'new_value': new_value,
                    'change_type': self._get_change_type(old_value, new_value)
                }
            
            # 获取新预测
            new_prediction = self._get_model_prediction(model, counterfactual_features)
            new_score = self._extract_score(new_prediction)
            
            # 计算质量指标
            distance = self._calculate_distance(features, counterfactual_features)
            plausibility = self._calculate_plausibility(features, counterfactual_features)
            
            return CounterfactualExample(
                original_features=features.copy(),
                counterfactual_features=counterfactual_features,
                original_prediction=original_prediction,
                counterfactual_prediction=new_prediction,
                changed_features=features_to_change,
                feature_changes=feature_changes,
                distance=distance,
                validity=1.0,
                plausibility=plausibility,
                explanation=self._generate_explanation(feature_changes, original_score, new_score),
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"多样化候选生成失败: {e}")
            return None
    
    def _generate_diverse_change(self, feature_name: str, original_value: Any) -> Any:
        """生成多样化的特征变化"""
        if isinstance(original_value, (int, float)):
            # 数值特征：更大范围的变化
            change_factor = random.uniform(0.3, 2.0)
            return original_value * change_factor
        elif isinstance(original_value, str):
            # 字符串特征：更多样的变化
            modifications = [
                lambda t: t + "_多样化修改",
                lambda t: t[::-1] if len(t) > 2 else t + "_反转",
                lambda t: "完全不同的文本",
                lambda t: t.upper() if t.islower() else t.lower(),
                lambda t: t.replace("的", "之") if "的" in t else t + "额外"
            ]
            modification = random.choice(modifications)
            return modification(original_value)
        elif isinstance(original_value, list):
            # 列表特征：更多样的操作
            if not original_value:
                return [f"新元素_{i}" for i in range(random.randint(1, 3))]
            
            operations = [
                lambda l: [],  # 清空
                lambda l: l + [f"新增_{i}" for i in range(random.randint(1, 3))],  # 添加
                lambda l: l[:len(l)//2] if len(l) > 1 else l,  # 截断
                lambda l: l * 2 if len(l) < 5 else l  # 重复
            ]
            operation = random.choice(operations)
            return operation(original_value.copy())
        else:
            return original_value
    
    def _select_diverse_subset(self, candidates: List[CounterfactualExample], 
                             num_examples: int) -> List[CounterfactualExample]:
        """选择多样化的子集"""
        if len(candidates) <= num_examples:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # 首先选择质量最高的
        remaining.sort(key=lambda x: x.plausibility, reverse=True)
        selected.append(remaining.pop(0))
        
        # 逐步添加最多样化的样本
        while len(selected) < num_examples and remaining:
            best_candidate = None
            best_diversity = -1
            
            for candidate in remaining:
                diversity = self._calculate_diversity_score(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _calculate_diversity_score(self, candidate: CounterfactualExample,
                                 selected: List[CounterfactualExample]) -> float:
        """计算多样性分数"""
        if not selected:
            return 1.0
        
        min_distance = float('inf')
        
        for existing in selected:
            distance = self._calculate_distance(
                candidate.counterfactual_features,
                existing.counterfactual_features
            )
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    # 继承其他方法
    def _extract_score(self, prediction: Any) -> float:
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            return prediction.get('overall_anomaly_score', 0.0)
        else:
            return 0.0
    
    def _get_model_prediction(self, model: Any, features: Dict[str, Any]) -> Any:
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
    
    def _calculate_distance(self, original: Dict[str, Any], 
                          counterfactual: Dict[str, Any]) -> float:
        distance = 0.0
        for feature in original.keys():
            if feature in counterfactual:
                if isinstance(original[feature], (int, float)) and isinstance(counterfactual[feature], (int, float)):
                    distance += abs(original[feature] - counterfactual[feature]) / (abs(original[feature]) + abs(counterfactual[feature]) + 1e-6)
                else:
                    distance += 1.0 if original[feature] != counterfactual[feature] else 0.0
        return distance
    
    def _calculate_plausibility(self, original: Dict[str, Any], 
                              counterfactual: Dict[str, Any]) -> float:
        changes = sum(1 for k in original.keys() 
                     if k in counterfactual and original[k] != counterfactual[k])
        return max(0.1, 1.0 - (changes / len(original)))
    
    def _get_change_type(self, old_value: Any, new_value: Any) -> str:
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if new_value > old_value:
                return "increase"
            elif new_value < old_value:
                return "decrease"
            else:
                return "unchanged"
        else:
            return "modification"
    
    def _generate_explanation(self, feature_changes: Dict[str, Dict[str, Any]],
                            original_score: float, new_score: float) -> str:
        if not feature_changes:
            return "无显著变化"
        
        return f"修改了 {len(feature_changes)} 个特征，异常分数从 {original_score:.3f} 变为 {new_score:.3f}"


class CounterfactualGenerator:
    """
    反事实生成器主类
    
    提供多种反事实生成策略：
    1. 最小变化 - 找到改变预测的最小修改
    2. 目标导向 - 生成达到特定目标的反事实
    3. 多样化集合 - 生成多样化的反事实集合
    4. 鲁棒性分析 - 分析模型的决策边界
    5. 解释生成 - 自动生成反事实解释
    6. 质量评估 - 评估反事实的质量和有效性
    
    Args:
        config: 生成器配置
        
    Attributes:
        generators: 生成器字典
        output_dir: 输出目录
        generation_history: 生成历史
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化反事实生成器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 输出目录
        self.output_dir = Path("data/counterfactuals")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 注册生成器
        self.generators: Dict[str, BaseCounterfactualGenerator] = {}
        self._register_generators()
        
        # 生成历史
        self.generation_history: List[CounterfactualSet] = []
        
        logger.info("反事实生成器初始化完成")
    
    def _register_generators(self):
        """注册生成器"""
        self.generators['minimal_change'] = MinimalChangeGenerator(
            max_changes=self.config.get('max_changes', 3),
            max_iterations=self.config.get('max_iterations', 100)
        )
        self.generators['diverse_set'] = DiverseSetGenerator(
            diversity_threshold=self.config.get('diversity_threshold', 0.3)
        )
    
    def generate_counterfactuals(self, features: Dict[str, Any],
                               prediction: Any,
                               model: Any,
                               method: str = "minimal_change",
                               target_outcome: Optional[Any] = None,
                               num_examples: int = 5) -> CounterfactualSet:
        """
        生成反事实样本集合
        
        Args:
            features: 原始特征
            prediction: 原始预测
            model: 模型实例
            method: 生成方法
            target_outcome: 目标结果
            num_examples: 生成数量
            
        Returns:
            CounterfactualSet: 反事实集合
        """
        try:
            if method not in self.generators:
                raise ValueError(f"未知的生成方法: {method}")
            
            logger.info(f"使用{method}方法生成{num_examples}个反事实样本...")
            
            generator = self.generators[method]
            examples = generator.generate_counterfactuals(
                features, prediction, model, target_outcome, num_examples
            )
            
            # 计算集合质量指标
            quality_metrics = self._calculate_set_quality(examples)
            diversity_score = self._calculate_diversity_score(examples)
            coverage_score = self._calculate_coverage_score(examples, features)
            
            # 生成全局解释
            global_explanation = self._generate_global_explanation(
                examples, features, prediction
            )
            
            # 创建反事实集合
            counterfactual_set = CounterfactualSet(
                examples=examples,
                generation_method=method,
                diversity_score=diversity_score,
                coverage_score=coverage_score,
                quality_metrics=quality_metrics,
                global_explanation=global_explanation,
                timestamp=datetime.now().isoformat()
            )
            
            # 保存结果
            self._save_counterfactual_set(counterfactual_set)
            
            # 添加到历史
            self.generation_history.append(counterfactual_set)
            
            logger.info(f"反事实生成完成，共{len(examples)}个样本")
            
            return counterfactual_set
            
        except Exception as e:
            logger.error(f"反事实生成失败: {e}")
            
            # 返回空集合
            return CounterfactualSet(
                examples=[],
                generation_method=method,
                diversity_score=0.0,
                coverage_score=0.0,
                quality_metrics={'error': 1.0},
                global_explanation=f"生成失败: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _calculate_set_quality(self, examples: List[CounterfactualExample]) -> Dict[str, float]:
        """计算集合质量指标"""
        if not examples:
            return {'average_validity': 0.0, 'average_plausibility': 0.0, 'average_distance': 0.0}
        
        total_validity = sum(ex.validity for ex in examples)
        total_plausibility = sum(ex.plausibility for ex in examples)
        total_distance = sum(ex.distance for ex in examples)
        
        return {
            'average_validity': total_validity / len(examples),
            'average_plausibility': total_plausibility / len(examples),
            'average_distance': total_distance / len(examples),
            'count': len(examples)
        }
    
    def _calculate_diversity_score(self, examples: List[CounterfactualExample]) -> float:
        """计算多样性分数"""
        if len(examples) < 2:
            return 0.0
        
        total_distance = 0.0
        pairs = 0
        
        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                distance = self._calculate_feature_distance(
                    examples[i].counterfactual_features,
                    examples[j].counterfactual_features
                )
                total_distance += distance
                pairs += 1
        
        return total_distance / pairs if pairs > 0 else 0.0
    
    def _calculate_coverage_score(self, examples: List[CounterfactualExample], 
                                original_features: Dict[str, Any]) -> float:
        """计算覆盖度分数"""
        if not examples:
            return 0.0
        
        # 计算被修改的特征覆盖率
        all_features = set(original_features.keys())
        modified_features = set()
        
        for example in examples:
            modified_features.update(example.changed_features)
        
        coverage = len(modified_features) / len(all_features) if all_features else 0.0
        return coverage
    
    def _calculate_feature_distance(self, features1: Dict[str, Any], 
                                  features2: Dict[str, Any]) -> float:
        """计算特征距离"""
        distance = 0.0
        
        for feature in features1.keys():
            if feature in features2:
                if isinstance(features1[feature], (int, float)) and isinstance(features2[feature], (int, float)):
                    distance += abs(features1[feature] - features2[feature]) / (abs(features1[feature]) + abs(features2[feature]) + 1e-6)
                else:
                    distance += 1.0 if features1[feature] != features2[feature] else 0.0
        
        return distance
    
    def _generate_global_explanation(self, examples: List[CounterfactualExample],
                                   original_features: Dict[str, Any],
                                   original_prediction: Any) -> str:
        """生成全局解释"""
        try:
            if not examples:
                return "未能生成有效的反事实样本。"
            
            # 分析最常修改的特征
            feature_change_count = {}
            for example in examples:
                for feature in example.changed_features:
                    feature_change_count[feature] = feature_change_count.get(feature, 0) + 1
            
            # 排序获取最重要的特征
            most_changed = sorted(feature_change_count.items(), key=lambda x: x[1], reverse=True)
            
            # 分析分数变化
            original_score = self._extract_score(original_prediction)
            score_changes = []
            for example in examples:
                new_score = self._extract_score(example.counterfactual_prediction)
                score_changes.append(new_score - original_score)
            
            avg_score_change = sum(score_changes) / len(score_changes)
            
            explanation_parts = [
                f"生成了 {len(examples)} 个反事实样本。",
                f"原始异常分数: {original_score:.3f}，"
                f"平均变化: {avg_score_change:+.3f}。"
            ]
            
            if most_changed:
                explanation_parts.append(
                    f"最常修改的特征: {', '.join(f[0] for f in most_changed[:3])}。"
                )
            
            # 添加质量评估
            avg_plausibility = sum(ex.plausibility for ex in examples) / len(examples)
            explanation_parts.append(f"平均合理性: {avg_plausibility:.2f}")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"全局解释生成失败: {e}")
            return f"全局解释生成失败: {str(e)}"
    
    def _extract_score(self, prediction: Any) -> float:
        """从预测结果中提取分数"""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            return prediction.get('overall_anomaly_score', 0.0)
        else:
            return 0.0
    
    def _save_counterfactual_set(self, counterfactual_set: CounterfactualSet):
        """保存反事实集合"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"counterfactuals_{counterfactual_set.generation_method}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # 转换为可序列化格式
            set_dict = {
                'examples': [
                    {
                        'original_features': ex.original_features,
                        'counterfactual_features': ex.counterfactual_features,
                        'original_prediction': str(ex.original_prediction),
                        'counterfactual_prediction': str(ex.counterfactual_prediction),
                        'changed_features': ex.changed_features,
                        'feature_changes': ex.feature_changes,
                        'distance': ex.distance,
                        'validity': ex.validity,
                        'plausibility': ex.plausibility,
                        'explanation': ex.explanation,
                        'confidence': ex.confidence
                    }
                    for ex in counterfactual_set.examples
                ],
                'generation_method': counterfactual_set.generation_method,
                'diversity_score': counterfactual_set.diversity_score,
                'coverage_score': counterfactual_set.coverage_score,
                'quality_metrics': counterfactual_set.quality_metrics,
                'global_explanation': counterfactual_set.global_explanation,
                'timestamp': counterfactual_set.timestamp
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(set_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"反事实集合已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存反事实集合失败: {e}")
    
    def create_counterfactual_comparison(self, counterfactual_set: CounterfactualSet) -> str:
        """创建反事实对比报告"""
        try:
            lines = [
                "反事实分析报告",
                "=" * 50,
                f"生成方法: {counterfactual_set.generation_method}",
                f"样本数量: {len(counterfactual_set.examples)}",
                f"多样性分数: {counterfactual_set.diversity_score:.3f}",
                f"覆盖度分数: {counterfactual_set.coverage_score:.3f}",
                "",
                "质量指标:",
            ]
            
            for metric, value in counterfactual_set.quality_metrics.items():
                lines.append(f"  {metric}: {value:.3f}")
            
            lines.extend([
                "",
                "全局解释:",
                counterfactual_set.global_explanation,
                "",
                "详细对比:"
            ])
            
            for i, example in enumerate(counterfactual_set.examples, 1):
                lines.extend([
                    f"\n反事实样本 {i}:",
                    f"  变化特征: {', '.join(example.changed_features)}",
                    f"  距离: {example.distance:.3f}",
                    f"  合理性: {example.plausibility:.3f}",
                    f"  解释: {example.explanation}"
                ])
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"counterfactual_report_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"反事实对比报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"反事实对比报告生成失败: {e}")
            return ""
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """获取生成摘要"""
        return {
            'total_generations': len(self.generation_history),
            'available_methods': list(self.generators.keys()),
            'output_directory': str(self.output_dir),
            'config': self.config,
            'recent_generations': [
                {
                    'timestamp': cf_set.timestamp,
                    'method': cf_set.generation_method,
                    'examples_count': len(cf_set.examples),
                    'diversity_score': cf_set.diversity_score,
                    'coverage_score': cf_set.coverage_score
                }
                for cf_set in self.generation_history[-5:]
            ],
            'numpy_available': NUMPY_AVAILABLE
        }


# 工厂函数
def create_counterfactual_generator(config: Optional[Dict[str, Any]] = None) -> CounterfactualGenerator:
    """
    创建反事实生成器实例
    
    Args:
        config: 配置字典
        
    Returns:
        CounterfactualGenerator: 生成器实例
    """
    return CounterfactualGenerator(config)


if __name__ == "__main__":
    # 使用示例
    print("反事实生成器测试:")
    
    # 创建生成器
    generator = CounterfactualGenerator()
    
    # 模拟原始样本
    features = {
        'content_length': 1200,
        'sentiment_score': 0.8,
        'technical_terms': 15,
        'author_reputation': 0.9,
        'market_timing': 'after_hours',
        'financial_data_count': 8
    }
    
    prediction = {
        'overall_anomaly_score': 0.75,
        'overall_anomaly_level': 'HIGH'
    }
    
    # 模拟模型
    class MockModel:
        def detect_anomalies(self, features):
            score = 0.0
            if features.get('content_length', 0) < 500:
                score += 0.3
            if features.get('sentiment_score', 0) > 0.9:
                score += 0.2
            if features.get('technical_terms', 0) < 5:
                score += 0.4
            if features.get('author_reputation', 0) < 0.5:
                score += 0.3
            
            return {
                'overall_anomaly_score': min(score, 1.0),
                'overall_anomaly_level': 'HIGH' if score > 0.7 else 'MEDIUM' if score > 0.3 else 'LOW'
            }
    
    model = MockModel()
    
    print(f"原始特征: {features}")
    print(f"原始预测: {prediction}")
    
    # 生成最小变化反事实
    print("\n生成最小变化反事实:")
    minimal_set = generator.generate_counterfactuals(
        features=features,
        prediction=prediction,
        model=model,
        method="minimal_change",
        num_examples=3
    )
    
    print(f"生成结果: {len(minimal_set.examples)}个样本")
    print(f"多样性分数: {minimal_set.diversity_score:.3f}")
    print(f"全局解释: {minimal_set.global_explanation}")
    
    # 生成多样化反事实
    print("\n生成多样化反事实:")
    diverse_set = generator.generate_counterfactuals(
        features=features,
        prediction=prediction,
        model=model,
        method="diverse_set",
        num_examples=3
    )
    
    print(f"生成结果: {len(diverse_set.examples)}个样本")
    print(f"多样性分数: {diverse_set.diversity_score:.3f}")
    
    # 展示样本详情
    if minimal_set.examples:
        example = minimal_set.examples[0]
        print(f"\n样本详情:")
        print(f"  变化特征: {example.changed_features}")
        print(f"  距离: {example.distance:.3f}")
        print(f"  合理性: {example.plausibility:.3f}")
        print(f"  解释: {example.explanation}")
    
    # 创建对比报告
    report_file = generator.create_counterfactual_comparison(minimal_set)
    if report_file:
        print(f"\n对比报告已保存: {report_file}")
    
    # 获取摘要
    summary = generator.get_generation_summary()
    print(f"\n生成摘要: {summary}") 
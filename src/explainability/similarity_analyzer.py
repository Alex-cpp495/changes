"""
相似性分析器
通过找到相似的正常/异常样本来解释异常检测结果，提供基于相似性的解释
"""

import json
import math
import heapq
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
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


class SimilarityMetric(Enum):
    """相似性度量枚举"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    SEMANTIC = "semantic"
    WEIGHTED = "weighted"


@dataclass
class SimilarSample:
    """相似样本数据结构"""
    sample_id: str
    features: Dict[str, Any]
    prediction: Any
    similarity_score: float
    distance: float
    
    # 解释信息
    matching_features: List[str]
    contrasting_features: List[str]
    explanation: str
    confidence: float = 1.0
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SimilarityAnalysisResult:
    """相似性分析结果"""
    target_sample: Dict[str, Any]
    target_prediction: Any
    
    # 相似样本
    similar_normal_samples: List[SimilarSample]
    similar_anomaly_samples: List[SimilarSample]
    contrasting_samples: List[SimilarSample]
    
    # 分析结果
    explanation: str
    confidence_score: float
    analysis_method: str
    
    # 统计信息
    statistics: Dict[str, Any]
    timestamp: str


class BaseSimilarityCalculator(ABC):
    """相似性计算器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_similarity(self, features1: Dict[str, Any], 
                           features2: Dict[str, Any]) -> float:
        """计算两个样本之间的相似性"""
        pass
    
    @abstractmethod
    def calculate_distance(self, features1: Dict[str, Any], 
                         features2: Dict[str, Any]) -> float:
        """计算两个样本之间的距离"""
        pass


class CosineSimCalculator(BaseSimilarityCalculator):
    """余弦相似性计算器"""
    
    def __init__(self):
        super().__init__("cosine")
    
    def calculate_similarity(self, features1: Dict[str, Any], 
                           features2: Dict[str, Any]) -> float:
        """计算余弦相似性"""
        try:
            # 提取数值特征
            vec1 = self._extract_numeric_vector(features1)
            vec2 = self._extract_numeric_vector(features2)
            
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            # 计算余弦相似性
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception as e:
            logger.error(f"余弦相似性计算失败: {e}")
            return 0.0
    
    def calculate_distance(self, features1: Dict[str, Any], 
                         features2: Dict[str, Any]) -> float:
        """计算余弦距离"""
        similarity = self.calculate_similarity(features1, features2)
        return 1.0 - similarity
    
    def _extract_numeric_vector(self, features: Dict[str, Any]) -> List[float]:
        """提取数值向量"""
        vector = []
        
        for key in sorted(features.keys()):  # 保证顺序一致
            value = features[key]
            
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, str):
                # 字符串长度作为特征
                vector.append(float(len(value)))
            elif isinstance(value, list):
                # 列表长度作为特征
                vector.append(float(len(value)))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(0.0)
        
        return vector


class EuclideanSimCalculator(BaseSimilarityCalculator):
    """欧几里得相似性计算器"""
    
    def __init__(self):
        super().__init__("euclidean")
    
    def calculate_similarity(self, features1: Dict[str, Any], 
                           features2: Dict[str, Any]) -> float:
        """计算基于欧几里得距离的相似性"""
        distance = self.calculate_distance(features1, features2)
        # 转换为相似性（距离越小，相似性越高）
        return 1.0 / (1.0 + distance)
    
    def calculate_distance(self, features1: Dict[str, Any], 
                         features2: Dict[str, Any]) -> float:
        """计算欧几里得距离"""
        try:
            vec1 = self._extract_numeric_vector(features1)
            vec2 = self._extract_numeric_vector(features2)
            
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return float('inf')
            
            # 计算欧几里得距离
            squared_diff = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
            return math.sqrt(squared_diff)
            
        except Exception as e:
            logger.error(f"欧几里得距离计算失败: {e}")
            return float('inf')
    
    def _extract_numeric_vector(self, features: Dict[str, Any]) -> List[float]:
        """提取数值向量"""
        vector = []
        
        for key in sorted(features.keys()):
            value = features[key]
            
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, str):
                vector.append(float(len(value)))
            elif isinstance(value, list):
                vector.append(float(len(value)))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(0.0)
        
        return vector


class JaccardSimCalculator(BaseSimilarityCalculator):
    """Jaccard相似性计算器（适用于集合特征）"""
    
    def __init__(self):
        super().__init__("jaccard")
    
    def calculate_similarity(self, features1: Dict[str, Any], 
                           features2: Dict[str, Any]) -> float:
        """计算Jaccard相似性"""
        try:
            # 提取集合特征
            sets1 = self._extract_set_features(features1)
            sets2 = self._extract_set_features(features2)
            
            if not sets1 or not sets2:
                return 0.0
            
            total_similarity = 0.0
            feature_count = 0
            
            # 计算每个特征的Jaccard相似性
            all_features = set(sets1.keys()) | set(sets2.keys())
            
            for feature in all_features:
                set1 = sets1.get(feature, set())
                set2 = sets2.get(feature, set())
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if union > 0:
                    similarity = intersection / union
                    total_similarity += similarity
                    feature_count += 1
            
            return total_similarity / feature_count if feature_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Jaccard相似性计算失败: {e}")
            return 0.0
    
    def calculate_distance(self, features1: Dict[str, Any], 
                         features2: Dict[str, Any]) -> float:
        """计算Jaccard距离"""
        similarity = self.calculate_similarity(features1, features2)
        return 1.0 - similarity
    
    def _extract_set_features(self, features: Dict[str, Any]) -> Dict[str, set]:
        """提取集合特征"""
        set_features = {}
        
        for key, value in features.items():
            if isinstance(value, list):
                set_features[key] = set(value)
            elif isinstance(value, str):
                # 将字符串转换为字符集合
                set_features[key] = set(value.split())
            elif isinstance(value, set):
                set_features[key] = value
            else:
                # 单值转换为单元素集合
                set_features[key] = {str(value)}
        
        return set_features


class WeightedSimCalculator(BaseSimilarityCalculator):
    """加权相似性计算器"""
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        super().__init__("weighted")
        self.feature_weights = feature_weights or {}
    
    def calculate_similarity(self, features1: Dict[str, Any], 
                           features2: Dict[str, Any]) -> float:
        """计算加权相似性"""
        try:
            total_weighted_similarity = 0.0
            total_weight = 0.0
            
            all_features = set(features1.keys()) | set(features2.keys())
            
            for feature in all_features:
                if feature in features1 and feature in features2:
                    # 计算单个特征的相似性
                    feature_sim = self._calculate_feature_similarity(
                        features1[feature], features2[feature]
                    )
                    
                    # 获取权重
                    weight = self.feature_weights.get(feature, 1.0)
                    
                    total_weighted_similarity += feature_sim * weight
                    total_weight += weight
            
            return total_weighted_similarity / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"加权相似性计算失败: {e}")
            return 0.0
    
    def calculate_distance(self, features1: Dict[str, Any], 
                         features2: Dict[str, Any]) -> float:
        """计算加权距离"""
        similarity = self.calculate_similarity(features1, features2)
        return 1.0 - similarity
    
    def _calculate_feature_similarity(self, value1: Any, value2: Any) -> float:
        """计算单个特征的相似性"""
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # 数值特征：基于归一化差异
            max_val = max(abs(value1), abs(value2))
            if max_val == 0:
                return 1.0
            diff = abs(value1 - value2) / max_val
            return 1.0 - min(diff, 1.0)
        elif isinstance(value1, str) and isinstance(value2, str):
            # 字符串特征：基于编辑距离
            return self._string_similarity(value1, value2)
        elif isinstance(value1, list) and isinstance(value2, list):
            # 列表特征：基于Jaccard相似性
            set1, set2 = set(value1), set(value2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        else:
            # 其他类型：精确匹配
            return 1.0 if value1 == value2 else 0.0
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似性（简化的编辑距离）"""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # 简化的字符串相似性计算
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        # 计算公共字符数
        common_chars = sum(1 for c in str1 if c in str2)
        return common_chars / max_len


class SimilarityAnalyzer:
    """
    相似性分析器
    
    提供基于相似性的可解释功能：
    1. 相似样本搜索 - 找到与目标样本相似的正常/异常样本
    2. 对比分析 - 分析相似样本与目标样本的差异
    3. 解释生成 - 基于相似性生成可理解的解释
    4. 异常原因推断 - 通过对比推断异常的可能原因
    5. 决策边界分析 - 分析模型的决策边界
    6. 样本聚类 - 对样本进行相似性聚类
    
    Args:
        config: 分析器配置
        
    Attributes:
        calculators: 相似性计算器字典
        sample_database: 样本数据库
        analysis_history: 分析历史
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化相似性分析器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 输出目录
        self.output_dir = Path("data/similarity_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 注册相似性计算器
        self.calculators: Dict[str, BaseSimilarityCalculator] = {}
        self._register_calculators()
        
        # 样本数据库
        self.sample_database: List[Dict[str, Any]] = []
        
        # 分析历史
        self.analysis_history: List[SimilarityAnalysisResult] = []
        
        logger.info("相似性分析器初始化完成")
    
    def _register_calculators(self):
        """注册相似性计算器"""
        self.calculators['cosine'] = CosineSimCalculator()
        self.calculators['euclidean'] = EuclideanSimCalculator()
        self.calculators['jaccard'] = JaccardSimCalculator()
        
        # 注册加权计算器（如果提供了权重配置）
        feature_weights = self.config.get('feature_weights')
        if feature_weights:
            self.calculators['weighted'] = WeightedSimCalculator(feature_weights)
    
    def add_samples_to_database(self, samples: List[Dict[str, Any]]):
        """添加样本到数据库"""
        try:
            for sample in samples:
                if 'features' in sample and 'prediction' in sample:
                    # 添加样本ID
                    if 'sample_id' not in sample:
                        sample['sample_id'] = f"sample_{len(self.sample_database)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    self.sample_database.append(sample)
            
            logger.info(f"添加了 {len(samples)} 个样本到数据库，总数: {len(self.sample_database)}")
            
        except Exception as e:
            logger.error(f"添加样本到数据库失败: {e}")
    
    def analyze_similarity(self, target_features: Dict[str, Any],
                         target_prediction: Any,
                         similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
                         num_similar: int = 5,
                         num_contrasting: int = 3) -> SimilarityAnalysisResult:
        """
        分析目标样本的相似性
        
        Args:
            target_features: 目标样本特征
            target_prediction: 目标样本预测
            similarity_metric: 相似性度量
            num_similar: 相似样本数量
            num_contrasting: 对比样本数量
            
        Returns:
            SimilarityAnalysisResult: 分析结果
        """
        try:
            if similarity_metric.value not in self.calculators:
                raise ValueError(f"不支持的相似性度量: {similarity_metric}")
            
            if not self.sample_database:
                raise ValueError("样本数据库为空，请先添加样本")
            
            logger.info(f"开始相似性分析，使用{similarity_metric.value}度量...")
            
            calculator = self.calculators[similarity_metric.value]
            target_is_anomaly = self._is_anomaly(target_prediction)
            
            # 计算与数据库中所有样本的相似性
            similarities = []
            
            for sample in self.sample_database:
                similarity_score = calculator.calculate_similarity(
                    target_features, sample['features']
                )
                distance = calculator.calculate_distance(
                    target_features, sample['features']
                )
                
                similarities.append({
                    'sample': sample,
                    'similarity_score': similarity_score,
                    'distance': distance,
                    'is_anomaly': self._is_anomaly(sample['prediction'])
                })
            
            # 按相似性排序
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # 找到相似的正常样本和异常样本
            similar_normal_samples = []
            similar_anomaly_samples = []
            contrasting_samples = []
            
            for sim in similarities:
                if sim['is_anomaly'] == target_is_anomaly:
                    # 同类样本
                    if sim['is_anomaly'] and len(similar_anomaly_samples) < num_similar:
                        similar_sample = self._create_similar_sample(
                            sim, target_features, target_prediction
                        )
                        similar_anomaly_samples.append(similar_sample)
                    elif not sim['is_anomaly'] and len(similar_normal_samples) < num_similar:
                        similar_sample = self._create_similar_sample(
                            sim, target_features, target_prediction
                        )
                        similar_normal_samples.append(similar_sample)
                else:
                    # 异类样本（用于对比）
                    if len(contrasting_samples) < num_contrasting:
                        contrasting_sample = self._create_similar_sample(
                            sim, target_features, target_prediction
                        )
                        contrasting_samples.append(contrasting_sample)
            
            # 生成解释
            explanation = self._generate_similarity_explanation(
                target_features, target_prediction,
                similar_normal_samples, similar_anomaly_samples, contrasting_samples
            )
            
            # 计算置信度
            confidence_score = self._calculate_confidence(
                similar_normal_samples, similar_anomaly_samples, contrasting_samples
            )
            
            # 生成统计信息
            statistics = self._generate_statistics(
                similarities, target_is_anomaly
            )
            
            # 创建分析结果
            result = SimilarityAnalysisResult(
                target_sample=target_features,
                target_prediction=target_prediction,
                similar_normal_samples=similar_normal_samples,
                similar_anomaly_samples=similar_anomaly_samples,
                contrasting_samples=contrasting_samples,
                explanation=explanation,
                confidence_score=confidence_score,
                analysis_method=similarity_metric.value,
                statistics=statistics,
                timestamp=datetime.now().isoformat()
            )
            
            # 保存结果
            self._save_analysis_result(result)
            
            # 添加到历史
            self.analysis_history.append(result)
            
            logger.info(f"相似性分析完成")
            
            return result
            
        except Exception as e:
            logger.error(f"相似性分析失败: {e}")
            
            # 返回空结果
            return SimilarityAnalysisResult(
                target_sample=target_features,
                target_prediction=target_prediction,
                similar_normal_samples=[],
                similar_anomaly_samples=[],
                contrasting_samples=[],
                explanation=f"分析失败: {str(e)}",
                confidence_score=0.0,
                analysis_method=similarity_metric.value,
                statistics={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    def _is_anomaly(self, prediction: Any) -> bool:
        """判断预测是否为异常"""
        if isinstance(prediction, dict):
            # 检查异常等级
            level = prediction.get('overall_anomaly_level', 'NORMAL')
            if level in ['HIGH', 'CRITICAL']:
                return True
            
            # 检查异常分数
            score = prediction.get('overall_anomaly_score', 0.0)
            return score > 0.5
        elif isinstance(prediction, (int, float)):
            return float(prediction) > 0.5
        else:
            return False
    
    def _create_similar_sample(self, similarity_data: Dict[str, Any],
                             target_features: Dict[str, Any],
                             target_prediction: Any) -> SimilarSample:
        """创建相似样本对象"""
        sample = similarity_data['sample']
        
        # 分析匹配和对比特征
        matching_features, contrasting_features = self._analyze_feature_differences(
            target_features, sample['features']
        )
        
        # 生成解释
        explanation = self._generate_sample_explanation(
            similarity_data, matching_features, contrasting_features
        )
        
        return SimilarSample(
            sample_id=sample.get('sample_id', 'unknown'),
            features=sample['features'],
            prediction=sample['prediction'],
            similarity_score=similarity_data['similarity_score'],
            distance=similarity_data['distance'],
            matching_features=matching_features,
            contrasting_features=contrasting_features,
            explanation=explanation,
            confidence=min(similarity_data['similarity_score'], 1.0),
            metadata=sample.get('metadata', {})
        )
    
    def _analyze_feature_differences(self, features1: Dict[str, Any],
                                   features2: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """分析特征差异"""
        matching_features = []
        contrasting_features = []
        
        all_features = set(features1.keys()) | set(features2.keys())
        
        for feature in all_features:
            if feature in features1 and feature in features2:
                val1, val2 = features1[feature], features2[feature]
                
                # 判断是否匹配
                if self._features_match(val1, val2):
                    matching_features.append(feature)
                else:
                    contrasting_features.append(feature)
            else:
                # 缺失特征
                contrasting_features.append(feature)
        
        return matching_features, contrasting_features
    
    def _features_match(self, val1: Any, val2: Any, threshold: float = 0.1) -> bool:
        """判断两个特征值是否匹配"""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # 数值特征：相对差异小于阈值
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return True
            diff = abs(val1 - val2) / max_val
            return diff < threshold
        elif isinstance(val1, str) and isinstance(val2, str):
            # 字符串特征：精确匹配或长度相近
            if val1 == val2:
                return True
            len_diff = abs(len(val1) - len(val2)) / max(len(val1), len(val2), 1)
            return len_diff < threshold
        else:
            # 其他类型：精确匹配
            return val1 == val2
    
    def _generate_sample_explanation(self, similarity_data: Dict[str, Any],
                                   matching_features: List[str],
                                   contrasting_features: List[str]) -> str:
        """生成样本解释"""
        similarity_score = similarity_data['similarity_score']
        is_anomaly = similarity_data['is_anomaly']
        
        explanation_parts = [
            f"相似度: {similarity_score:.3f}",
            f"类型: {'异常' if is_anomaly else '正常'}"
        ]
        
        if matching_features:
            explanation_parts.append(f"匹配特征: {', '.join(matching_features[:3])}")
        
        if contrasting_features:
            explanation_parts.append(f"差异特征: {', '.join(contrasting_features[:3])}")
        
        return " | ".join(explanation_parts)
    
    def _generate_similarity_explanation(self, target_features: Dict[str, Any],
                                       target_prediction: Any,
                                       similar_normal_samples: List[SimilarSample],
                                       similar_anomaly_samples: List[SimilarSample],
                                       contrasting_samples: List[SimilarSample]) -> str:
        """生成相似性解释"""
        try:
            target_is_anomaly = self._is_anomaly(target_prediction)
            
            explanation_parts = []
            
            if target_is_anomaly:
                explanation_parts.append("目标样本被检测为异常。")
                
                if similar_anomaly_samples:
                    avg_similarity = sum(s.similarity_score for s in similar_anomaly_samples) / len(similar_anomaly_samples)
                    explanation_parts.append(
                        f"找到 {len(similar_anomaly_samples)} 个相似的异常样本"
                        f"（平均相似度: {avg_similarity:.3f}）。"
                    )
                
                if contrasting_samples:
                    explanation_parts.append(
                        f"与 {len(contrasting_samples)} 个正常样本存在显著差异。"
                    )
            else:
                explanation_parts.append("目标样本被检测为正常。")
                
                if similar_normal_samples:
                    avg_similarity = sum(s.similarity_score for s in similar_normal_samples) / len(similar_normal_samples)
                    explanation_parts.append(
                        f"找到 {len(similar_normal_samples)} 个相似的正常样本"
                        f"（平均相似度: {avg_similarity:.3f}）。"
                    )
            
            # 分析主要差异特征
            if contrasting_samples:
                common_contrasting_features = self._find_common_contrasting_features(contrasting_samples)
                if common_contrasting_features:
                    explanation_parts.append(
                        f"主要差异特征: {', '.join(common_contrasting_features[:3])}"
                    )
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"相似性解释生成失败: {e}")
            return f"解释生成失败: {str(e)}"
    
    def _find_common_contrasting_features(self, contrasting_samples: List[SimilarSample]) -> List[str]:
        """找到共同的对比特征"""
        feature_count = defaultdict(int)
        
        for sample in contrasting_samples:
            for feature in sample.contrasting_features:
                feature_count[feature] += 1
        
        # 按出现频率排序
        common_features = sorted(feature_count.items(), key=lambda x: x[1], reverse=True)
        
        return [feature for feature, count in common_features if count >= len(contrasting_samples) // 2]
    
    def _calculate_confidence(self, similar_normal_samples: List[SimilarSample],
                            similar_anomaly_samples: List[SimilarSample],
                            contrasting_samples: List[SimilarSample]) -> float:
        """计算置信度"""
        try:
            # 基于相似样本的数量和质量计算置信度
            same_type_samples = similar_normal_samples + similar_anomaly_samples
            
            if not same_type_samples:
                return 0.1
            
            # 平均相似度
            avg_similarity = sum(s.similarity_score for s in same_type_samples) / len(same_type_samples)
            
            # 样本数量权重
            quantity_weight = min(len(same_type_samples) / 5.0, 1.0)  # 最多5个样本为满分
            
            # 对比样本权重（有对比样本时置信度更高）
            contrast_weight = 1.0 + (0.2 if contrasting_samples else 0.0)
            
            confidence = avg_similarity * quantity_weight * contrast_weight
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _generate_statistics(self, similarities: List[Dict[str, Any]], 
                           target_is_anomaly: bool) -> Dict[str, Any]:
        """生成统计信息"""
        try:
            total_samples = len(similarities)
            normal_samples = sum(1 for s in similarities if not s['is_anomaly'])
            anomaly_samples = total_samples - normal_samples
            
            similarity_scores = [s['similarity_score'] for s in similarities]
            
            return {
                'total_samples': total_samples,
                'normal_samples': normal_samples,
                'anomaly_samples': anomaly_samples,
                'target_is_anomaly': target_is_anomaly,
                'avg_similarity': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0,
                'max_similarity': max(similarity_scores) if similarity_scores else 0.0,
                'min_similarity': min(similarity_scores) if similarity_scores else 0.0,
                'similarity_std': self._calculate_std(similarity_scores)
            }
            
        except Exception as e:
            logger.error(f"统计信息生成失败: {e}")
            return {'error': str(e)}
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _save_analysis_result(self, result: SimilarityAnalysisResult):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"similarity_analysis_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # 转换为可序列化格式
            result_dict = {
                'target_sample': result.target_sample,
                'target_prediction': str(result.target_prediction),
                'similar_normal_samples': [
                    {
                        'sample_id': s.sample_id,
                        'features': s.features,
                        'prediction': str(s.prediction),
                        'similarity_score': s.similarity_score,
                        'distance': s.distance,
                        'matching_features': s.matching_features,
                        'contrasting_features': s.contrasting_features,
                        'explanation': s.explanation,
                        'confidence': s.confidence
                    }
                    for s in result.similar_normal_samples
                ],
                'similar_anomaly_samples': [
                    {
                        'sample_id': s.sample_id,
                        'features': s.features,
                        'prediction': str(s.prediction),
                        'similarity_score': s.similarity_score,
                        'distance': s.distance,
                        'matching_features': s.matching_features,
                        'contrasting_features': s.contrasting_features,
                        'explanation': s.explanation,
                        'confidence': s.confidence
                    }
                    for s in result.similar_anomaly_samples
                ],
                'contrasting_samples': [
                    {
                        'sample_id': s.sample_id,
                        'features': s.features,
                        'prediction': str(s.prediction),
                        'similarity_score': s.similarity_score,
                        'distance': s.distance,
                        'matching_features': s.matching_features,
                        'contrasting_features': s.contrasting_features,
                        'explanation': s.explanation,
                        'confidence': s.confidence
                    }
                    for s in result.contrasting_samples
                ],
                'explanation': result.explanation,
                'confidence_score': result.confidence_score,
                'analysis_method': result.analysis_method,
                'statistics': result.statistics,
                'timestamp': result.timestamp
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"相似性分析结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def create_similarity_report(self, result: SimilarityAnalysisResult) -> str:
        """创建相似性分析报告"""
        try:
            lines = [
                "相似性分析报告",
                "=" * 50,
                f"分析方法: {result.analysis_method}",
                f"置信度: {result.confidence_score:.3f}",
                f"分析时间: {result.timestamp}",
                "",
                "目标样本:",
                f"  预测: {result.target_prediction}",
                f"  特征数量: {len(result.target_sample)}",
                "",
                "分析结果:",
                result.explanation,
                "",
                "统计信息:"
            ]
            
            for key, value in result.statistics.items():
                lines.append(f"  {key}: {value}")
            
            # 相似正常样本
            if result.similar_normal_samples:
                lines.extend([
                    "",
                    f"相似正常样本 ({len(result.similar_normal_samples)}个):"
                ])
                
                for i, sample in enumerate(result.similar_normal_samples, 1):
                    lines.extend([
                        f"  {i}. {sample.sample_id}",
                        f"     相似度: {sample.similarity_score:.3f}",
                        f"     匹配特征: {', '.join(sample.matching_features[:5])}",
                        f"     解释: {sample.explanation}"
                    ])
            
            # 相似异常样本
            if result.similar_anomaly_samples:
                lines.extend([
                    "",
                    f"相似异常样本 ({len(result.similar_anomaly_samples)}个):"
                ])
                
                for i, sample in enumerate(result.similar_anomaly_samples, 1):
                    lines.extend([
                        f"  {i}. {sample.sample_id}",
                        f"     相似度: {sample.similarity_score:.3f}",
                        f"     匹配特征: {', '.join(sample.matching_features[:5])}",
                        f"     解释: {sample.explanation}"
                    ])
            
            # 对比样本
            if result.contrasting_samples:
                lines.extend([
                    "",
                    f"对比样本 ({len(result.contrasting_samples)}个):"
                ])
                
                for i, sample in enumerate(result.contrasting_samples, 1):
                    lines.extend([
                        f"  {i}. {sample.sample_id}",
                        f"     相似度: {sample.similarity_score:.3f}",
                        f"     差异特征: {', '.join(sample.contrasting_features[:5])}",
                        f"     解释: {sample.explanation}"
                    ])
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"similarity_report_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"相似性分析报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"相似性分析报告生成失败: {e}")
            return ""
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            'total_analyses': len(self.analysis_history),
            'database_size': len(self.sample_database),
            'available_metrics': list(self.calculators.keys()),
            'output_directory': str(self.output_dir),
            'config': self.config,
            'recent_analyses': [
                {
                    'timestamp': result.timestamp,
                    'method': result.analysis_method,
                    'confidence': result.confidence_score,
                    'similar_normal_count': len(result.similar_normal_samples),
                    'similar_anomaly_count': len(result.similar_anomaly_samples),
                    'contrasting_count': len(result.contrasting_samples)
                }
                for result in self.analysis_history[-5:]
            ],
            'numpy_available': NUMPY_AVAILABLE
        }


# 工厂函数
def create_similarity_analyzer(config: Optional[Dict[str, Any]] = None) -> SimilarityAnalyzer:
    """
    创建相似性分析器实例
    
    Args:
        config: 配置字典
        
    Returns:
        SimilarityAnalyzer: 分析器实例
    """
    return SimilarityAnalyzer(config)


if __name__ == "__main__":
    # 使用示例
    print("相似性分析器测试:")
    
    # 创建分析器
    analyzer = SimilarityAnalyzer()
    
    # 添加样本到数据库
    sample_database = [
        {
            'sample_id': 'normal_1',
            'features': {
                'content_length': 1200,
                'sentiment_score': 0.7,
                'technical_terms': 15,
                'author_reputation': 0.8
            },
            'prediction': {'overall_anomaly_score': 0.2, 'overall_anomaly_level': 'LOW'}
        },
        {
            'sample_id': 'normal_2',
            'features': {
                'content_length': 1100,
                'sentiment_score': 0.6,
                'technical_terms': 12,
                'author_reputation': 0.9
            },
            'prediction': {'overall_anomaly_score': 0.3, 'overall_anomaly_level': 'LOW'}
        },
        {
            'sample_id': 'anomaly_1',
            'features': {
                'content_length': 300,
                'sentiment_score': 0.95,
                'technical_terms': 2,
                'author_reputation': 0.1
            },
            'prediction': {'overall_anomaly_score': 0.9, 'overall_anomaly_level': 'HIGH'}
        },
        {
            'sample_id': 'anomaly_2',
            'features': {
                'content_length': 250,
                'sentiment_score': 0.98,
                'technical_terms': 1,
                'author_reputation': 0.2
            },
            'prediction': {'overall_anomaly_score': 0.85, 'overall_anomaly_level': 'HIGH'}
        }
    ]
    
    analyzer.add_samples_to_database(sample_database)
    print(f"样本数据库大小: {len(analyzer.sample_database)}")
    
    # 目标样本（异常）
    target_features = {
        'content_length': 280,
        'sentiment_score': 0.92,
        'technical_terms': 3,
        'author_reputation': 0.15
    }
    
    target_prediction = {'overall_anomaly_score': 0.88, 'overall_anomaly_level': 'HIGH'}
    
    print(f"\n目标样本: {target_features}")
    print(f"目标预测: {target_prediction}")
    
    # 进行相似性分析
    result = analyzer.analyze_similarity(
        target_features=target_features,
        target_prediction=target_prediction,
        similarity_metric=SimilarityMetric.COSINE,
        num_similar=3,
        num_contrasting=2
    )
    
    print(f"\n分析结果:")
    print(f"置信度: {result.confidence_score:.3f}")
    print(f"解释: {result.explanation}")
    
    print(f"\n相似异常样本: {len(result.similar_anomaly_samples)}个")
    for sample in result.similar_anomaly_samples:
        print(f"  - {sample.sample_id}: 相似度 {sample.similarity_score:.3f}")
    
    print(f"\n对比正常样本: {len(result.contrasting_samples)}个")
    for sample in result.contrasting_samples:
        print(f"  - {sample.sample_id}: 相似度 {sample.similarity_score:.3f}")
    
    # 创建分析报告
    report_file = analyzer.create_similarity_report(result)
    if report_file:
        print(f"\n分析报告已保存: {report_file}")
    
    # 获取摘要
    summary = analyzer.get_analysis_summary()
    print(f"\n分析摘要: {summary}") 
"""
可解释性系统模块
包含注意力可视化、特征重要性分析、反事实生成、相似性分析等可解释性组件
"""

from .attention_visualizer import (
    AttentionVisualizer, AttentionData, VisualizationConfig,
    create_attention_visualizer
)
from .feature_importance import (
    FeatureImportanceAnalyzer, FeatureImportance, ImportanceAnalysisResult,
    BaseImportanceAnalyzer, ShapleyValueAnalyzer, PermutationImportanceAnalyzer,
    GradientBasedAnalyzer, create_feature_importance_analyzer
)
from .counterfactual_generator import (
    CounterfactualGenerator, CounterfactualExample, CounterfactualSet,
    CounterfactualType, BaseCounterfactualGenerator, MinimalChangeGenerator,
    DiverseSetGenerator, create_counterfactual_generator
)
from .similarity_analyzer import (
    SimilarityAnalyzer, SimilarSample, SimilarityAnalysisResult,
    SimilarityMetric, BaseSimilarityCalculator, CosineSimCalculator,
    EuclideanSimCalculator, JaccardSimCalculator, WeightedSimCalculator,
    create_similarity_analyzer
)

__all__ = [
    # 注意力可视化
    'AttentionVisualizer',
    'AttentionData',
    'VisualizationConfig',
    'create_attention_visualizer',
    
    # 特征重要性分析
    'FeatureImportanceAnalyzer',
    'FeatureImportance',
    'ImportanceAnalysisResult',
    'BaseImportanceAnalyzer',
    'ShapleyValueAnalyzer',
    'PermutationImportanceAnalyzer', 
    'GradientBasedAnalyzer',
    'create_feature_importance_analyzer',
    
    # 反事实生成
    'CounterfactualGenerator',
    'CounterfactualExample',
    'CounterfactualSet',
    'CounterfactualType',
    'BaseCounterfactualGenerator',
    'MinimalChangeGenerator',
    'DiverseSetGenerator',
    'create_counterfactual_generator',
    
    # 相似性分析
    'SimilarityAnalyzer',
    'SimilarSample',
    'SimilarityAnalysisResult',
    'SimilarityMetric',
    'BaseSimilarityCalculator',
    'CosineSimCalculator',
    'EuclideanSimCalculator',
    'JaccardSimCalculator',
    'WeightedSimCalculator',
    'create_similarity_analyzer'
] 
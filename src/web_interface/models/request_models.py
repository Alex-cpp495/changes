"""
Web API请求数据模型
定义所有API端点的请求数据结构和验证规则
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import json


class DetectionMode(str, Enum):
    """检测模式"""
    SINGLE = "single"           # 单个文档检测
    BATCH = "batch"            # 批量检测
    REAL_TIME = "real_time"    # 实时检测


class FeedbackTypeEnum(str, Enum):
    """反馈类型"""
    CORRECT_DETECTION = "correct_detection"
    INCORRECT_DETECTION = "incorrect_detection"
    MISSING_DETECTION = "missing_detection"
    FALSE_POSITIVE = "false_positive"
    SEVERITY_ADJUSTMENT = "severity_adjustment"


class ReportTypeEnum(str, Enum):
    """报告类型"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class DetectionRequest(BaseModel):
    """异常检测请求模型"""
    
    # 基本信息
    report_content: str = Field(
        ..., 
        min_length=10,
        max_length=50000,
        description="研报文本内容"
    )
    
    report_id: Optional[str] = Field(
        None,
        description="报告ID，如果不提供将自动生成"
    )
    
    report_title: Optional[str] = Field(
        None,
        max_length=500,
        description="报告标题"
    )
    
    # 检测配置
    detection_mode: DetectionMode = Field(
        DetectionMode.SINGLE,
        description="检测模式"
    )
    
    enable_statistical: bool = Field(
        True,
        description="是否启用统计异常检测"
    )
    
    enable_behavioral: bool = Field(
        True,
        description="是否启用行为异常检测"
    )
    
    enable_market: bool = Field(
        True,
        description="是否启用市场异常检测"
    )
    
    enable_semantic: bool = Field(
        True,
        description="是否启用语义异常检测"
    )
    
    # 阈值设置
    custom_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="自定义检测阈值"
    )
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="附加元数据"
    )
    
    # 输出选项
    include_explanations: bool = Field(
        True,
        description="是否包含解释信息"
    )
    
    include_raw_scores: bool = Field(
        False,
        description="是否包含原始分数"
    )
    
    @validator('custom_thresholds')
    def validate_thresholds(cls, v):
        if v is not None:
            for key, value in v.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"阈值 {key} 必须是数字")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"阈值 {key} 必须在0-1之间")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "report_content": "某公司Q3财报显示营收增长25%，但现金流为负...",
                "report_title": "某公司三季度财报分析",
                "detection_mode": "single",
                "include_explanations": True,
                "custom_thresholds": {
                    "overall_anomaly": 0.7
                }
            }
        }


class FeedbackRequest(BaseModel):
    """用户反馈请求模型"""
    
    # 基本信息
    report_id: str = Field(
        ...,
        description="报告ID"
    )
    
    original_prediction: Dict[str, Any] = Field(
        ...,
        description="原始预测结果"
    )
    
    # 反馈内容
    feedback_type: FeedbackTypeEnum = Field(
        ...,
        description="反馈类型"
    )
    
    is_correct: bool = Field(
        ...,
        description="检测结果是否正确"
    )
    
    corrected_label: Optional[str] = Field(
        None,
        description="修正后的标签"
    )
    
    confidence_rating: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="置信度评分(1-10)"
    )
    
    explanation: Optional[str] = Field(
        None,
        max_length=2000,
        description="反馈说明"
    )
    
    # 详细反馈
    feature_feedback: Optional[Dict[str, Any]] = Field(
        None,
        description="特征级别的反馈"
    )
    
    severity_feedback: Optional[str] = Field(
        None,
        description="严重程度反馈"
    )
    
    additional_notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="附加备注"
    )
    
    # 用户信息
    user_id: Optional[str] = Field(
        None,
        description="用户ID"
    )
    
    user_expertise: Optional[str] = Field(
        None,
        description="用户专业程度(expert/analyst/user)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "report_id": "report_123",
                "original_prediction": {
                    "overall_anomaly_score": 0.8,
                    "overall_anomaly_level": "HIGH"
                },
                "feedback_type": "correct_detection",
                "is_correct": True,
                "confidence_rating": 8,
                "explanation": "检测结果准确，确实存在异常",
                "user_expertise": "expert"
            }
        }


class ConfigUpdateRequest(BaseModel):
    """配置更新请求模型"""
    
    # 检测阈值
    detection_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="检测阈值配置"
    )
    
    # 集成权重
    ensemble_weights: Optional[Dict[str, float]] = Field(
        None,
        description="集成检测器权重"
    )
    
    # 特征权重
    feature_weights: Optional[Dict[str, float]] = Field(
        None,
        description="特征权重配置"
    )
    
    # 模型参数
    model_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="模型参数配置"
    )
    
    # 监控配置
    monitoring_config: Optional[Dict[str, Any]] = Field(
        None,
        description="监控配置"
    )
    
    # 学习配置
    learning_config: Optional[Dict[str, Any]] = Field(
        None,
        description="自适应学习配置"
    )
    
    @validator('detection_thresholds', 'ensemble_weights', 'feature_weights')
    def validate_weights_and_thresholds(cls, v):
        if v is not None:
            for key, value in v.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"权重/阈值 {key} 必须是数字")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"权重/阈值 {key} 必须在0-1之间")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "detection_thresholds": {
                    "statistical_anomaly": 0.7,
                    "overall_anomaly": 0.65
                },
                "ensemble_weights": {
                    "statistical_detector": 0.3,
                    "semantic_detector": 0.4
                }
            }
        }


class ReportGenerationRequest(BaseModel):
    """报告生成请求模型"""
    
    report_type: ReportTypeEnum = Field(
        ...,
        description="报告类型"
    )
    
    start_date: Optional[datetime] = Field(
        None,
        description="开始日期"
    )
    
    end_date: Optional[datetime] = Field(
        None,
        description="结束日期"
    )
    
    include_charts: bool = Field(
        True,
        description="是否包含图表"
    )
    
    include_recommendations: bool = Field(
        True,
        description="是否包含建议"
    )
    
    format: str = Field(
        "json",
        description="报告格式(json/pdf/html)"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="过滤条件"
    )
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v <= values['start_date']:
                raise ValueError("结束日期必须晚于开始日期")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "report_type": "weekly",
                "include_charts": True,
                "include_recommendations": True,
                "format": "json"
            }
        }


class BatchProcessRequest(BaseModel):
    """批量处理请求模型"""
    
    # 批量数据
    reports: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="报告列表"
    )
    
    # 处理选项
    parallel_processing: bool = Field(
        True,
        description="是否并行处理"
    )
    
    max_workers: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="最大工作线程数"
    )
    
    # 输出选项
    output_format: str = Field(
        "json",
        description="输出格式"
    )
    
    include_individual_results: bool = Field(
        True,
        description="是否包含单个结果"
    )
    
    include_summary: bool = Field(
        True,
        description="是否包含汇总信息"
    )
    
    # 错误处理
    continue_on_error: bool = Field(
        True,
        description="出错时是否继续处理"
    )
    
    @validator('reports')
    def validate_reports(cls, v):
        for i, report in enumerate(v):
            if 'content' not in report:
                raise ValueError(f"报告 {i} 缺少 content 字段")
            if not isinstance(report['content'], str):
                raise ValueError(f"报告 {i} 的 content 必须是字符串")
            if len(report['content']) < 10:
                raise ValueError(f"报告 {i} 的内容过短")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "reports": [
                    {
                        "content": "报告内容1...",
                        "title": "报告1",
                        "id": "report_001"
                    },
                    {
                        "content": "报告内容2...",
                        "title": "报告2", 
                        "id": "report_002"
                    }
                ],
                "parallel_processing": True,
                "max_workers": 4,
                "include_summary": True
            }
        }


class HealthCheckRequest(BaseModel):
    """健康检查请求模型"""
    
    include_detailed_status: bool = Field(
        False,
        description="是否包含详细状态信息"
    )
    
    check_components: Optional[List[str]] = Field(
        None,
        description="要检查的组件列表"
    )
    
    timeout_seconds: Optional[int] = Field(
        30,
        ge=1,
        le=300,
        description="超时时间(秒)"
    )


class SystemControlRequest(BaseModel):
    """系统控制请求模型"""
    
    action: str = Field(
        ...,
        description="操作类型(start/stop/restart/reload)"
    )
    
    component: Optional[str] = Field(
        None,
        description="组件名称"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="操作参数"
    )
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = ['start', 'stop', 'restart', 'reload', 'reset']
        if v not in valid_actions:
            raise ValueError(f"无效操作: {v}，有效操作: {valid_actions}")
        return v 
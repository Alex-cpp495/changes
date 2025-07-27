"""
Web界面基础数据模型
定义系统中使用的核心数据结构
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class AnomalyLevel(str, Enum):
    """异常级别"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionCategory(str, Enum):
    """检测类别"""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    MARKET = "market"
    SEMANTIC = "semantic"


class ComponentStatus(str, Enum):
    """组件状态"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


class AnomalyResult(BaseModel):
    """异常检测结果数据模型"""
    
    # 基本结果
    overall_anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="总体异常分数"
    )
    
    overall_anomaly_level: AnomalyLevel = Field(
        ...,
        description="总体异常级别"
    )
    
    is_anomalous: bool = Field(
        ...,
        description="是否为异常"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="检测置信度"
    )
    
    # 详细分数
    detection_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="各检测器分数"
    )
    
    detection_categories: List[DetectionCategory] = Field(
        default_factory=list,
        description="触发的检测类别"
    )
    
    # 具体异常信息
    anomaly_details: Optional[Dict[str, Any]] = Field(
        None,
        description="异常详情"
    )
    
    risk_factors: Optional[List[str]] = Field(
        None,
        description="风险因素列表"
    )
    
    # 时间信息
    detection_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="检测时间戳"
    )
    
    processing_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="处理元数据"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "overall_anomaly_score": 0.78,
                "overall_anomaly_level": "HIGH",
                "is_anomalous": True,
                "confidence": 0.85,
                "detection_scores": {
                    "statistical": 0.65,
                    "behavioral": 0.82,
                    "market": 0.71,
                    "semantic": 0.89
                },
                "detection_categories": ["behavioral", "semantic"],
                "risk_factors": [
                    "现金流异常",
                    "收入确认时点疑似不当"
                ]
            }
        }


class ReportInfo(BaseModel):
    """报告信息数据模型"""
    
    report_id: str = Field(
        ...,
        description="报告ID"
    )
    
    report_type: str = Field(
        ...,
        description="报告类型"
    )
    
    title: Optional[str] = Field(
        None,
        description="报告标题"
    )
    
    generated_at: datetime = Field(
        ...,
        description="生成时间"
    )
    
    period_start: Optional[datetime] = Field(
        None,
        description="报告期间开始时间"
    )
    
    period_end: Optional[datetime] = Field(
        None,
        description="报告期间结束时间"
    )
    
    file_size: Optional[int] = Field(
        None,
        description="文件大小(字节)"
    )
    
    format: str = Field(
        "json",
        description="报告格式"
    )
    
    status: str = Field(
        "completed",
        description="报告状态"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="报告元数据"
    )


class SystemMetrics(BaseModel):
    """系统指标数据模型"""
    
    # 性能指标
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="检测准确率"
    )
    
    precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="精确率"
    )
    
    recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="召回率"
    )
    
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F1分数"
    )
    
    # 处理统计
    total_predictions: int = Field(
        ...,
        ge=0,
        description="总预测数"
    )
    
    total_errors: int = Field(
        ...,
        ge=0,
        description="总错误数"
    )
    
    average_processing_time: float = Field(
        ...,
        ge=0.0,
        description="平均处理时间(秒)"
    )
    
    # 系统资源
    cpu_usage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="CPU使用率(%)"
    )
    
    memory_usage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="内存使用率(%)"
    )
    
    disk_usage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="磁盘使用率(%)"
    )
    
    # 用户反馈
    user_satisfaction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="用户满意度"
    )
    
    feedback_count: int = Field(
        ...,
        ge=0,
        description="反馈数量"
    )
    
    # 时间信息
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="最后更新时间"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.87,
                "f1_score": 0.88,
                "total_predictions": 1250,
                "total_errors": 15,
                "average_processing_time": 2.3,
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "user_satisfaction": 0.85,
                "feedback_count": 89
            }
        }


class UserFeedbackInfo(BaseModel):
    """用户反馈信息数据模型"""
    
    feedback_id: str = Field(
        ...,
        description="反馈ID"
    )
    
    report_id: str = Field(
        ...,
        description="关联报告ID"
    )
    
    feedback_type: str = Field(
        ...,
        description="反馈类型"
    )
    
    is_correct: bool = Field(
        ...,
        description="检测结果是否正确"
    )
    
    confidence_rating: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="置信度评分"
    )
    
    explanation: Optional[str] = Field(
        None,
        description="反馈解释"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="用户ID"
    )
    
    user_expertise: Optional[str] = Field(
        None,
        description="用户专业程度"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    
    processed: bool = Field(
        False,
        description="是否已处理"
    )


class ModelConfiguration(BaseModel):
    """模型配置数据模型"""
    
    # 检测阈值
    detection_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="检测阈值配置"
    )
    
    # 集成权重
    ensemble_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="集成检测器权重"
    )
    
    # 特征权重
    feature_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="特征权重配置"
    )
    
    # 模型参数
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="模型参数"
    )
    
    # 配置信息
    version: str = Field(
        "1.0.0",
        description="配置版本"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="最后更新时间"
    )
    
    updated_by: Optional[str] = Field(
        None,
        description="更新者"
    )
    
    is_active: bool = Field(
        True,
        description="是否激活"
    )
    
    backup_config: Optional[Dict[str, Any]] = Field(
        None,
        description="备份配置"
    )
    
    @validator('detection_thresholds', 'ensemble_weights', 'feature_weights')
    def validate_weights_and_thresholds(cls, v):
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"权重/阈值 {key} 必须是数字")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"权重/阈值 {key} 必须在0-1之间")
        return v


class AlertInfo(BaseModel):
    """告警信息数据模型"""
    
    alert_id: str = Field(
        ...,
        description="告警ID"
    )
    
    alert_type: str = Field(
        ...,
        description="告警类型"
    )
    
    severity: str = Field(
        ...,
        description="严重程度"
    )
    
    title: str = Field(
        ...,
        description="告警标题"
    )
    
    message: str = Field(
        ...,
        description="告警消息"
    )
    
    source: str = Field(
        ...,
        description="告警来源"
    )
    
    current_value: Optional[float] = Field(
        None,
        description="当前值"
    )
    
    threshold_value: Optional[float] = Field(
        None,
        description="阈值"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    
    resolved: bool = Field(
        False,
        description="是否已解决"
    )
    
    resolved_at: Optional[datetime] = Field(
        None,
        description="解决时间"
    )
    
    acknowledged: bool = Field(
        False,
        description="是否已确认"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="告警元数据"
    )


class TaskInfo(BaseModel):
    """任务信息数据模型"""
    
    task_id: str = Field(
        ...,
        description="任务ID"
    )
    
    task_type: str = Field(
        ...,
        description="任务类型"
    )
    
    task_name: str = Field(
        ...,
        description="任务名称"
    )
    
    status: str = Field(
        ...,
        description="任务状态"
    )
    
    progress: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="进度(0-1)"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    
    started_at: Optional[datetime] = Field(
        None,
        description="开始时间"
    )
    
    completed_at: Optional[datetime] = Field(
        None,
        description="完成时间"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="错误消息"
    )
    
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="任务结果"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="任务参数"
    )


class ComponentInfo(BaseModel):
    """组件信息数据模型"""
    
    component_name: str = Field(
        ...,
        description="组件名称"
    )
    
    component_type: str = Field(
        ...,
        description="组件类型"
    )
    
    status: ComponentStatus = Field(
        ...,
        description="组件状态"
    )
    
    version: str = Field(
        ...,
        description="组件版本"
    )
    
    health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="健康分数"
    )
    
    last_check: datetime = Field(
        default_factory=datetime.now,
        description="最后检查时间"
    )
    
    uptime: float = Field(
        ...,
        ge=0.0,
        description="运行时间(小时)"
    )
    
    error_count: int = Field(
        0,
        ge=0,
        description="错误计数"
    )
    
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="性能指标"
    )
    
    configuration: Optional[Dict[str, Any]] = Field(
        None,
        description="组件配置"
    )
    
    dependencies: Optional[List[str]] = Field(
        None,
        description="依赖组件列表"
    )


class StatisticsInfo(BaseModel):
    """统计信息数据模型"""
    
    period: str = Field(
        ...,
        description="统计周期"
    )
    
    start_date: datetime = Field(
        ...,
        description="开始日期"
    )
    
    end_date: datetime = Field(
        ...,
        description="结束日期"
    )
    
    total_reports: int = Field(
        ...,
        ge=0,
        description="总报告数"
    )
    
    anomalous_reports: int = Field(
        ...,
        ge=0,
        description="异常报告数"
    )
    
    anomaly_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="异常率"
    )
    
    average_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="平均异常分数"
    )
    
    category_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="类别分布"
    )
    
    trend_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="趋势指标"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="生成时间"
    ) 
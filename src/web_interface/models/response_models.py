"""
Web API响应数据模型
定义所有API端点的响应数据结构
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from .data_models import AnomalyResult, SystemMetrics, ReportInfo


class ResponseStatus(str, Enum):
    """响应状态"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL_SUCCESS = "partial_success"


class BaseResponse(BaseModel):
    """基础响应模型"""
    
    status: ResponseStatus = Field(
        ...,
        description="响应状态"
    )
    
    message: str = Field(
        ...,
        description="响应消息"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="响应时间戳"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="请求ID"
    )


class SuccessResponse(BaseResponse):
    """成功响应模型"""
    
    status: ResponseStatus = Field(
        ResponseStatus.SUCCESS,
        description="成功状态"
    )
    
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="响应数据"
    )


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    
    status: ResponseStatus = Field(
        ResponseStatus.ERROR,
        description="错误状态"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="错误代码"
    )
    
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="错误详情"
    )
    
    suggestion: Optional[str] = Field(
        None,
        description="解决建议"
    )


class DetectionResponse(BaseResponse):
    """异常检测响应模型"""
    
    # 基本信息
    report_id: str = Field(
        ...,
        description="报告ID"
    )
    
    processing_time: float = Field(
        ...,
        description="处理耗时(秒)"
    )
    
    # 检测结果
    anomaly_result: AnomalyResult = Field(
        ...,
        description="异常检测结果"
    )
    
    # 详细分析
    detailed_scores: Optional[Dict[str, Any]] = Field(
        None,
        description="详细分数信息"
    )
    
    explanations: Optional[Dict[str, Any]] = Field(
        None,
        description="解释信息"
    )
    
    # 可信度和质量指标
    confidence_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="置信度指标"
    )
    
    quality_assessment: Optional[Dict[str, Any]] = Field(
        None,
        description="质量评估"
    )
    
    # 建议和下一步
    recommendations: Optional[List[str]] = Field(
        None,
        description="建议列表"
    )
    
    next_actions: Optional[List[str]] = Field(
        None,
        description="建议的下一步操作"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "检测完成",
                "report_id": "report_123",
                "processing_time": 2.5,
                "anomaly_result": {
                    "overall_anomaly_score": 0.75,
                    "overall_anomaly_level": "HIGH",
                    "is_anomalous": True,
                    "detection_categories": ["semantic", "behavioral"]
                },
                "recommendations": [
                    "建议进一步核实财务数据",
                    "关注现金流指标"
                ]
            }
        }


class FeedbackResponse(BaseResponse):
    """反馈提交响应模型"""
    
    feedback_id: str = Field(
        ...,
        description="反馈ID"
    )
    
    feedback_summary: Dict[str, Any] = Field(
        ...,
        description="反馈摘要"
    )
    
    impact_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="影响分析"
    )
    
    learning_triggered: bool = Field(
        False,
        description="是否触发了学习"
    )
    
    updated_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="更新的参数"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "反馈已收集",
                "feedback_id": "feedback_456",
                "feedback_summary": {
                    "type": "correct_detection",
                    "confidence": 8
                },
                "learning_triggered": True
            }
        }


class SystemStatusResponse(BaseResponse):
    """系统状态响应模型"""
    
    system_health: str = Field(
        ...,
        description="系统健康状态"
    )
    
    components_status: Dict[str, Any] = Field(
        ...,
        description="组件状态"
    )
    
    performance_metrics: SystemMetrics = Field(
        ...,
        description="性能指标"
    )
    
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="活跃告警"
    )
    
    system_info: Dict[str, Any] = Field(
        ...,
        description="系统信息"
    )
    
    uptime: float = Field(
        ...,
        description="运行时间(小时)"
    )
    
    last_health_check: datetime = Field(
        ...,
        description="最后健康检查时间"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "系统状态正常",
                "system_health": "healthy",
                "components_status": {
                    "model_monitor": "running",
                    "adaptive_learner": "running",
                    "feedback_collector": "running"
                },
                "uptime": 48.5
            }
        }


class PerformanceReportResponse(BaseResponse):
    """性能报告响应模型"""
    
    report_info: ReportInfo = Field(
        ...,
        description="报告基本信息"
    )
    
    performance_summary: Dict[str, Any] = Field(
        ...,
        description="性能摘要"
    )
    
    detailed_metrics: Dict[str, Any] = Field(
        ...,
        description="详细指标"
    )
    
    trend_analysis: Dict[str, Any] = Field(
        ...,
        description="趋势分析"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="建议列表"
    )
    
    charts: Optional[Dict[str, str]] = Field(
        None,
        description="图表数据(base64编码)"
    )
    
    download_links: Optional[Dict[str, str]] = Field(
        None,
        description="下载链接"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "报告生成完成",
                "report_info": {
                    "report_id": "weekly_20231201",
                    "report_type": "weekly",
                    "generated_at": "2023-12-01T10:00:00Z"
                },
                "performance_summary": {
                    "accuracy": 0.92,
                    "total_predictions": 1250
                }
            }
        }


class BatchProcessResponse(BaseResponse):
    """批量处理响应模型"""
    
    batch_id: str = Field(
        ...,
        description="批次ID"
    )
    
    total_items: int = Field(
        ...,
        description="总项目数"
    )
    
    processed_items: int = Field(
        ...,
        description="已处理项目数"
    )
    
    successful_items: int = Field(
        ...,
        description="成功处理项目数"
    )
    
    failed_items: int = Field(
        ...,
        description="失败项目数"
    )
    
    processing_time: float = Field(
        ...,
        description="总处理时间(秒)"
    )
    
    individual_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="单个结果列表"
    )
    
    batch_summary: Dict[str, Any] = Field(
        ...,
        description="批次摘要"
    )
    
    error_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="错误摘要"
    )
    
    download_links: Optional[Dict[str, str]] = Field(
        None,
        description="结果下载链接"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "批量处理完成",
                "batch_id": "batch_789",
                "total_items": 50,
                "processed_items": 50,
                "successful_items": 48,
                "failed_items": 2,
                "processing_time": 125.5,
                "batch_summary": {
                    "average_anomaly_score": 0.45,
                    "anomalous_reports": 12
                }
            }
        }


class ConfigUpdateResponse(BaseResponse):
    """配置更新响应模型"""
    
    updated_config: Dict[str, Any] = Field(
        ...,
        description="更新后的配置"
    )
    
    changes_applied: List[str] = Field(
        ...,
        description="应用的更改列表"
    )
    
    validation_results: Dict[str, Any] = Field(
        ...,
        description="验证结果"
    )
    
    restart_required: bool = Field(
        False,
        description="是否需要重启"
    )
    
    effective_immediately: bool = Field(
        True,
        description="是否立即生效"
    )
    
    rollback_info: Optional[Dict[str, Any]] = Field(
        None,
        description="回滚信息"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "配置更新成功",
                "changes_applied": [
                    "detection_thresholds.overall_anomaly: 0.7 -> 0.65"
                ],
                "restart_required": False,
                "effective_immediately": True
            }
        }


class ModelStatusResponse(BaseResponse):
    """模型状态响应模型"""
    
    model_info: Dict[str, Any] = Field(
        ...,
        description="模型信息"
    )
    
    model_status: str = Field(
        ...,
        description="模型状态(loaded/loading/error)"
    )
    
    performance_stats: Dict[str, Any] = Field(
        ...,
        description="性能统计"
    )
    
    memory_usage: Dict[str, Any] = Field(
        ...,
        description="内存使用情况"
    )
    
    last_updated: Optional[datetime] = Field(
        None,
        description="最后更新时间"
    )
    
    health_score: float = Field(
        ...,
        description="健康分数(0-1)"
    )


class DashboardResponse(BaseResponse):
    """仪表板数据响应模型"""
    
    overview: Dict[str, Any] = Field(
        ...,
        description="概览信息"
    )
    
    real_time_metrics: Dict[str, Any] = Field(
        ...,
        description="实时指标"
    )
    
    recent_activities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="最近活动"
    )
    
    alerts_summary: Dict[str, Any] = Field(
        ...,
        description="告警摘要"
    )
    
    performance_trends: Dict[str, Any] = Field(
        ...,
        description="性能趋势"
    )
    
    system_status: Dict[str, Any] = Field(
        ...,
        description="系统状态"
    )
    
    quick_stats: Dict[str, Any] = Field(
        ...,
        description="快速统计"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "仪表板数据获取成功",
                "overview": {
                    "total_reports_today": 150,
                    "anomaly_rate": 0.15,
                    "system_health": "healthy"
                },
                "quick_stats": {
                    "accuracy": 0.92,
                    "processing_speed": "2.3s/report"
                }
            }
        }


class SearchResponse(BaseResponse):
    """搜索响应模型"""
    
    query: str = Field(
        ...,
        description="搜索查询"
    )
    
    total_results: int = Field(
        ...,
        description="总结果数"
    )
    
    page: int = Field(
        ...,
        description="当前页码"
    )
    
    page_size: int = Field(
        ...,
        description="页面大小"
    )
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="搜索结果"
    )
    
    facets: Optional[Dict[str, Any]] = Field(
        None,
        description="分面统计"
    )
    
    suggestions: Optional[List[str]] = Field(
        None,
        description="搜索建议"
    )
    
    search_time: float = Field(
        ...,
        description="搜索耗时(秒)"
    )


class ExportResponse(BaseResponse):
    """导出响应模型"""
    
    export_id: str = Field(
        ...,
        description="导出任务ID"
    )
    
    export_type: str = Field(
        ...,
        description="导出类型"
    )
    
    file_info: Dict[str, Any] = Field(
        ...,
        description="文件信息"
    )
    
    download_url: str = Field(
        ...,
        description="下载URL"
    )
    
    expires_at: datetime = Field(
        ...,
        description="过期时间"
    )
    
    file_size: Optional[int] = Field(
        None,
        description="文件大小(字节)"
    )


class AsyncTaskResponse(BaseResponse):
    """异步任务响应模型"""
    
    task_id: str = Field(
        ...,
        description="任务ID"
    )
    
    task_status: str = Field(
        ...,
        description="任务状态(pending/running/completed/failed)"
    )
    
    progress: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="进度(0-1)"
    )
    
    estimated_completion: Optional[datetime] = Field(
        None,
        description="预计完成时间"
    )
    
    result_url: Optional[str] = Field(
        None,
        description="结果URL"
    )
    
    error_info: Optional[Dict[str, Any]] = Field(
        None,
        description="错误信息"
    )


# 便捷函数用于创建标准响应
def create_success_response(message: str = "操作成功", 
                          data: Optional[Dict[str, Any]] = None,
                          request_id: Optional[str] = None) -> SuccessResponse:
    """创建成功响应"""
    return SuccessResponse(
        message=message,
        data=data,
        request_id=request_id
    )


def create_error_response(message: str, 
                         error_code: Optional[str] = None,
                         error_details: Optional[Dict[str, Any]] = None,
                         suggestion: Optional[str] = None,
                         request_id: Optional[str] = None) -> ErrorResponse:
    """创建错误响应"""
    return ErrorResponse(
        message=message,
        error_code=error_code,
        error_details=error_details,
        suggestion=suggestion,
        request_id=request_id
    ) 
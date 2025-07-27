"""
Web界面数据模型
Web Interface Models

定义Web API的请求和响应数据模型，基于Pydantic提供数据验证和序列化功能
"""

from .request_models import (
    DetectionRequest,
    FeedbackRequest,
    ConfigUpdateRequest,
    ReportGenerationRequest,
    BatchProcessRequest
)

from .response_models import (
    DetectionResponse,
    FeedbackResponse,
    SystemStatusResponse,
    PerformanceReportResponse,
    ErrorResponse,
    SuccessResponse
)

from .data_models import (
    AnomalyResult,
    ReportInfo,
    SystemMetrics,
    UserFeedbackInfo,
    ModelConfiguration
)

__version__ = "1.0.0"

__all__ = [
    # 请求模型
    'DetectionRequest',
    'FeedbackRequest', 
    'ConfigUpdateRequest',
    'ReportGenerationRequest',
    'BatchProcessRequest',
    
    # 响应模型
    'DetectionResponse',
    'FeedbackResponse',
    'SystemStatusResponse', 
    'PerformanceReportResponse',
    'ErrorResponse',
    'SuccessResponse',
    
    # 数据模型
    'AnomalyResult',
    'ReportInfo',
    'SystemMetrics',
    'UserFeedbackInfo',
    'ModelConfiguration'
] 
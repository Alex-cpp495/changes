"""
Web服务层
Web Services Layer

提供Web API与后端系统之间的业务逻辑服务层，负责：
- 业务逻辑处理和数据转换
- 后端系统集成和调用
- 缓存和性能优化
- 异常处理和错误管理
"""

from .detection_service import DetectionService
from .feedback_service import FeedbackService
from .monitoring_service import MonitoringService
from .report_service import ReportService
from .config_service import ConfigService
from .auth_service import AuthService

__version__ = "1.0.0"

__all__ = [
    'DetectionService',
    'FeedbackService', 
    'MonitoringService',
    'ReportService',
    'ConfigService',
    'AuthService'
] 
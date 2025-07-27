"""
报告服务
负责处理性能报告相关的业务逻辑
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.request_models import ReportGenerationRequest
from ..models.response_models import PerformanceReportResponse
from ..models.data_models import ReportInfo
from ...continuous_learning import get_continuous_learning_system, ReportType
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ReportService:
    """
    报告服务
    
    提供性能报告相关功能：
    1. 生成性能报告
    2. 报告管理
    3. 导出功能
    4. 历史报告查询
    """
    
    def __init__(self):
        """初始化报告服务"""
        self.continuous_learning = get_continuous_learning_system()
        logger.info("报告服务初始化完成")
    
    async def generate_report(self, request: ReportGenerationRequest) -> PerformanceReportResponse:
        """生成性能报告"""
        try:
            # 转换报告类型
            report_type_map = {
                "daily": ReportType.DAILY,
                "weekly": ReportType.WEEKLY,
                "monthly": ReportType.MONTHLY,
                "quarterly": ReportType.QUARTERLY,
                "custom": ReportType.CUSTOM
            }
            
            report_type = report_type_map.get(request.report_type, ReportType.WEEKLY)
            
            # 使用持续学习系统生成报告
            performance_report = self.continuous_learning.performance_tracker.generate_performance_report(
                report_type=report_type,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # 创建报告信息
            report_info = ReportInfo(
                report_id=performance_report.report_id,
                report_type=performance_report.report_type.value,
                title=f"{performance_report.report_type.value.title()} Performance Report",
                generated_at=performance_report.generated_at,
                period_start=performance_report.period_start,
                period_end=performance_report.period_end,
                format=request.format,
                status="completed"
            )
            
            # 转换性能摘要
            performance_summary = {
                "period_start": performance_report.summary.period_start.isoformat(),
                "period_end": performance_report.summary.period_end.isoformat(),
                "total_predictions": performance_report.summary.total_predictions,
                "accuracy": performance_report.summary.accuracy,
                "precision": performance_report.summary.precision,
                "recall": performance_report.summary.recall,
                "f1_score": performance_report.summary.f1_score,
                "user_satisfaction": performance_report.summary.user_satisfaction,
                "accuracy_trend": performance_report.summary.accuracy_trend.value,
                "performance_change": performance_report.summary.performance_change
            }
            
            # 处理图表（如果需要）
            charts = None
            if request.include_charts:
                charts = performance_report.charts
            
            # 处理建议（如果需要）
            recommendations = []
            if request.include_recommendations:
                recommendations = performance_report.recommendations
            
            return PerformanceReportResponse(
                status="success",
                message="报告生成完成",
                report_info=report_info,
                performance_summary=performance_summary,
                detailed_metrics=performance_report.detailed_metrics,
                trend_analysis=performance_report.trend_analysis,
                recommendations=recommendations,
                charts=charts
            )
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise


def get_report_service() -> ReportService:
    """获取报告服务实例"""
    return ReportService() 
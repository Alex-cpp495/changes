"""
报告管理API路由
"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.request_models import ReportGenerationRequest
from ..models.response_models import PerformanceReportResponse
from ..services.report_service import get_report_service
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/generate", response_model=PerformanceReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    report_service = Depends(get_report_service)
):
    """生成性能报告"""
    try:
        result = await report_service.generate_report(request)
        return result
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")


@router.get("/list")
async def get_reports_list():
    """获取报告列表"""
    return {
        "status": "success",
        "data": []
    } 
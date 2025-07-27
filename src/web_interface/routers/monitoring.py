"""
系统监控API路由
"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.response_models import SystemStatusResponse, DashboardResponse
from ..services.monitoring_service import get_monitoring_service
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    monitoring_service = Depends(get_monitoring_service)
):
    """获取系统状态"""
    try:
        result = await monitoring_service.get_system_status()
        return result
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    monitoring_service = Depends(get_monitoring_service)
):
    """获取仪表板数据"""
    try:
        result = await monitoring_service.get_dashboard_data()
        return result
    except Exception as e:
        logger.error(f"获取仪表板数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取仪表板数据失败: {str(e)}") 
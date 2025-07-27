"""
系统配置API路由
"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.request_models import ConfigUpdateRequest
from ..models.response_models import ConfigUpdateResponse
from ..services.config_service import get_config_service
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/update", response_model=ConfigUpdateResponse)
async def update_config(
    request: ConfigUpdateRequest,
    config_service = Depends(get_config_service)
):
    """更新系统配置"""
    try:
        result = await config_service.update_config(request)
        return result
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.get("/current")
async def get_current_config(
    config_service = Depends(get_config_service)
):
    """获取当前配置"""
    try:
        result = await config_service.get_current_config()
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}") 
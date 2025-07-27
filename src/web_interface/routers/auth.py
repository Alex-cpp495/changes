"""
用户认证API路由
"""

from fastapi import APIRouter, HTTPException, Depends

from ..services.auth_service import get_auth_service
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/login")
async def login(
    username: str,
    password: str,
    auth_service = Depends(get_auth_service)
):
    """用户登录"""
    try:
        result = auth_service.authenticate_user(username, password)
        if result:
            return {
                "status": "success",
                "message": "登录成功",
                "data": result
            }
        else:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录失败: {e}")
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@router.post("/logout")
async def logout(token: str):
    """用户登出"""
    return {
        "status": "success",
        "message": "登出成功"
    } 
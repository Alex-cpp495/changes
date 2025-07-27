"""
用户反馈API路由
"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.request_models import FeedbackRequest
from ..models.response_models import FeedbackResponse
from ..services.feedback_service import get_feedback_service
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_service = Depends(get_feedback_service)
):
    """提交用户反馈"""
    try:
        result = await feedback_service.submit_feedback(request)
        return result
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.get("/list")
async def get_feedback_list(limit: int = 50, offset: int = 0):
    """获取反馈列表"""
    return {
        "status": "success",
        "data": [],
        "total": 0
    } 
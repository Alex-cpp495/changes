"""
反馈服务
负责处理用户反馈相关的业务逻辑
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.request_models import FeedbackRequest
from ..models.response_models import FeedbackResponse
from ...continuous_learning import get_continuous_learning_system, FeedbackType, FeedbackSource
from ...utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackService:
    """
    反馈服务
    
    提供用户反馈相关功能：
    1. 收集用户反馈
    2. 分析反馈影响
    3. 触发学习更新
    4. 反馈统计分析
    """
    
    def __init__(self):
        """初始化反馈服务"""
        self.continuous_learning = get_continuous_learning_system()
        logger.info("反馈服务初始化完成")
    
    async def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        提交用户反馈
        
        Args:
            request: 反馈请求
            
        Returns:
            FeedbackResponse: 反馈响应
        """
        try:
            # 转换反馈类型
            feedback_type_map = {
                "correct_detection": FeedbackType.CORRECT_DETECTION,
                "incorrect_detection": FeedbackType.INCORRECT_DETECTION,
                "missing_detection": FeedbackType.MISSING_DETECTION,
                "false_positive": FeedbackType.FALSE_POSITIVE,
                "severity_adjustment": FeedbackType.SEVERITY_ADJUSTMENT
            }
            
            feedback_type = feedback_type_map.get(
                request.feedback_type, 
                FeedbackType.CORRECT_DETECTION
            )
            
            # 提交反馈
            feedback_id = self.continuous_learning.record_prediction_feedback(
                report_id=request.report_id,
                original_prediction=request.original_prediction,
                is_correct=request.is_correct,
                feedback_type=feedback_type,
                corrected_label=request.corrected_label,
                confidence_rating=request.confidence_rating,
                explanation=request.explanation,
                feature_feedback=request.feature_feedback,
                severity_feedback=request.severity_feedback,
                additional_notes=request.additional_notes,
                user_id=request.user_id,
                user_expertise=request.user_expertise,
                feedback_source=FeedbackSource.USER_INTERFACE
            )
            
            # 分析反馈影响
            impact_analysis = self._analyze_feedback_impact(request)
            
            # 检查是否触发学习
            learning_triggered = self._check_learning_trigger(request)
            
            # 生成反馈摘要
            feedback_summary = {
                'type': request.feedback_type,
                'is_correct': request.is_correct,
                'confidence': request.confidence_rating,
                'has_explanation': bool(request.explanation),
                'user_expertise': request.user_expertise
            }
            
            response = FeedbackResponse(
                status="success",
                message="反馈提交成功",
                feedback_id=feedback_id,
                feedback_summary=feedback_summary,
                impact_analysis=impact_analysis,
                learning_triggered=learning_triggered
            )
            
            logger.info(f"反馈提交成功: {feedback_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"提交反馈失败: {e}")
            
            return FeedbackResponse(
                status="error",
                message=f"提交反馈失败: {str(e)}",
                feedback_id="",
                feedback_summary={}
            )
    
    def _analyze_feedback_impact(self, request: FeedbackRequest) -> Dict[str, Any]:
        """分析反馈影响"""
        return {
            'model_accuracy_impact': 'positive' if request.is_correct else 'negative',
            'confidence_adjustment_needed': request.confidence_rating is not None,
            'threshold_adjustment_suggested': request.feedback_type in ['false_positive', 'missing_detection']
        }
    
    def _check_learning_trigger(self, request: FeedbackRequest) -> bool:
        """检查是否触发学习"""
        # 简化的触发逻辑
        return request.feedback_type in ['incorrect_detection', 'false_positive', 'missing_detection']


def get_feedback_service() -> FeedbackService:
    """获取反馈服务实例"""
    return FeedbackService() 
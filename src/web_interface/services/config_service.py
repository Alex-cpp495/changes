"""
配置服务
负责处理系统配置相关的业务逻辑
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.request_models import ConfigUpdateRequest
from ..models.response_models import ConfigUpdateResponse
from ..models.data_models import ModelConfiguration
from ...continuous_learning import get_continuous_learning_system
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ConfigService:
    """
    配置服务
    
    提供系统配置相关功能：
    1. 配置读取和更新
    2. 配置验证
    3. 配置版本管理
    4. 配置回滚
    """
    
    def __init__(self):
        """初始化配置服务"""
        self.continuous_learning = get_continuous_learning_system()
        logger.info("配置服务初始化完成")
    
    async def update_config(self, request: ConfigUpdateRequest) -> ConfigUpdateResponse:
        """更新系统配置"""
        try:
            changes_applied = []
            validation_results = {"valid": True, "errors": []}
            
            # 获取当前配置
            current_params = self.continuous_learning.adaptive_learner.current_parameters.copy()
            
            # 更新检测阈值
            if request.detection_thresholds:
                current_params['detection_thresholds'].update(request.detection_thresholds)
                changes_applied.extend([f"detection_thresholds.{k}: {v}" for k, v in request.detection_thresholds.items()])
            
            # 更新集成权重
            if request.ensemble_weights:
                current_params['ensemble_weights'].update(request.ensemble_weights)
                changes_applied.extend([f"ensemble_weights.{k}: {v}" for k, v in request.ensemble_weights.items()])
            
            # 更新特征权重
            if request.feature_weights:
                current_params['feature_weights'].update(request.feature_weights)
                changes_applied.extend([f"feature_weights.{k}: {v}" for k, v in request.feature_weights.items()])
            
            # 更新模型参数
            if request.model_parameters:
                current_params['model_parameters'].update(request.model_parameters)
                changes_applied.extend([f"model_parameters.{k}: {v}" for k, v in request.model_parameters.items()])
            
            # 验证配置
            validation_results = self._validate_config(current_params)
            
            if validation_results["valid"]:
                # 应用配置
                self.continuous_learning.adaptive_learner.current_parameters = current_params
                self.continuous_learning.adaptive_learner._save_current_parameters()
                
                return ConfigUpdateResponse(
                    status="success",
                    message="配置更新成功",
                    updated_config=current_params,
                    changes_applied=changes_applied,
                    validation_results=validation_results,
                    restart_required=False,
                    effective_immediately=True
                )
            else:
                return ConfigUpdateResponse(
                    status="error",
                    message="配置验证失败",
                    updated_config={},
                    changes_applied=[],
                    validation_results=validation_results,
                    restart_required=False,
                    effective_immediately=False
                )
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            raise
    
    async def get_current_config(self) -> ModelConfiguration:
        """获取当前配置"""
        try:
            current_params = self.continuous_learning.adaptive_learner.current_parameters
            
            return ModelConfiguration(
                detection_thresholds=current_params.get('detection_thresholds', {}),
                ensemble_weights=current_params.get('ensemble_weights', {}),
                feature_weights=current_params.get('feature_weights', {}),
                model_parameters=current_params.get('model_parameters', {}),
                version="1.0.0",
                last_updated=datetime.now(),
                is_active=True
            )
            
        except Exception as e:
            logger.error(f"获取当前配置失败: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        validation_result = {"valid": True, "errors": []}
        
        try:
            # 验证检测阈值
            thresholds = config.get('detection_thresholds', {})
            for key, value in thresholds.items():
                if not isinstance(value, (int, float)):
                    validation_result["errors"].append(f"检测阈值 {key} 必须是数字")
                elif not 0.0 <= value <= 1.0:
                    validation_result["errors"].append(f"检测阈值 {key} 必须在0-1之间")
            
            # 验证集成权重
            weights = config.get('ensemble_weights', {})
            weight_sum = sum(weights.values()) if weights else 0
            if weights and abs(weight_sum - 1.0) > 0.01:
                validation_result["errors"].append("集成权重总和应该等于1.0")
            
            # 如果有错误，标记为无效
            if validation_result["errors"]:
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"配置验证异常: {str(e)}")
        
        return validation_result


def get_config_service() -> ConfigService:
    """获取配置服务实例"""
    return ConfigService() 
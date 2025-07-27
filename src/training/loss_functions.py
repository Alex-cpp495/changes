"""
损失函数
实现多种损失函数，支持多任务学习、异常检测、不平衡数据等场景
"""

import math
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from abc import ABC, abstractmethod
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)

# 尝试导入torch，如果不可用则使用numpy替代
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch不可用，将使用NumPy实现")

if not TORCH_AVAILABLE:
    import numpy as np


class LossType(Enum):
    """损失函数类型枚举"""
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    LABEL_SMOOTHING = "label_smoothing"
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    MULTI_TASK = "multi_task"
    ANOMALY_DETECTION = "anomaly_detection"
    WEIGHTED_CE = "weighted_cross_entropy"


class BaseLoss(ABC):
    """损失函数基类"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.history = []
    
    @abstractmethod
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算损失值"""
        pass
    
    def update_history(self, loss_value: float):
        """更新损失历史"""
        self.history.append(loss_value)
        if len(self.history) > 1000:  # 保持最近1000个值
            self.history = self.history[-1000:]
    
    def get_average_loss(self, window: int = 100) -> float:
        """获取平均损失"""
        if not self.history:
            return 0.0
        recent_losses = self.history[-window:] if len(self.history) > window else self.history
        return sum(recent_losses) / len(recent_losses)


class CrossEntropyLoss(BaseLoss):
    """交叉熵损失"""
    
    def __init__(self, weight: float = 1.0, ignore_index: int = -100, 
                 reduction: str = 'mean'):
        super().__init__("cross_entropy", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算交叉熵损失"""
        try:
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # PyTorch实现
                loss = F.cross_entropy(
                    predictions, 
                    targets, 
                    ignore_index=self.ignore_index,
                    reduction=self.reduction
                )
                loss_value = float(loss.item())
            else:
                # NumPy实现
                loss_value = self._numpy_cross_entropy(predictions, targets)
            
            self.update_history(loss_value)
            return loss_value * self.weight
            
        except Exception as e:
            logger.error(f"交叉熵损失计算失败: {e}")
            return 0.0
    
    def _numpy_cross_entropy(self, predictions: Any, targets: Any) -> float:
        """NumPy实现的交叉熵损失"""
        try:
            # 简化实现，实际使用中需要更完善的处理
            if hasattr(predictions, 'shape') and hasattr(targets, 'shape'):
                # 假设predictions是logits，targets是标签
                predictions = np.array(predictions)
                targets = np.array(targets)
                
                # Softmax
                exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                softmax_preds = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
                
                # 交叉熵
                epsilon = 1e-12
                softmax_preds = np.clip(softmax_preds, epsilon, 1.0 - epsilon)
                
                if targets.ndim == 1:  # 标签格式
                    loss = -np.mean(np.log(softmax_preds[np.arange(len(targets)), targets]))
                else:  # one-hot格式
                    loss = -np.mean(np.sum(targets * np.log(softmax_preds), axis=-1))
                
                return float(loss)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"NumPy交叉熵计算失败: {e}")
            return 1.0


class FocalLoss(BaseLoss):
    """Focal Loss - 用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__("focal_loss", weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算Focal损失"""
        try:
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # PyTorch实现
                ce_loss = F.cross_entropy(predictions, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    loss_value = float(focal_loss.mean().item())
                elif self.reduction == 'sum':
                    loss_value = float(focal_loss.sum().item())
                else:
                    loss_value = float(focal_loss.mean().item())
            else:
                # NumPy实现
                loss_value = self._numpy_focal_loss(predictions, targets)
            
            self.update_history(loss_value)
            return loss_value * self.weight
            
        except Exception as e:
            logger.error(f"Focal损失计算失败: {e}")
            return 0.0
    
    def _numpy_focal_loss(self, predictions: Any, targets: Any) -> float:
        """NumPy实现的Focal损失"""
        try:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # Softmax
            exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            softmax_preds = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
            
            # 交叉熵
            epsilon = 1e-12
            softmax_preds = np.clip(softmax_preds, epsilon, 1.0 - epsilon)
            
            # 计算pt
            if targets.ndim == 1:
                pt = softmax_preds[np.arange(len(targets)), targets]
                ce = -np.log(pt)
            else:
                pt = np.sum(targets * softmax_preds, axis=-1)
                ce = -np.log(pt)
            
            # Focal loss
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
            
            return float(np.mean(focal_loss))
            
        except Exception as e:
            logger.error(f"NumPy Focal损失计算失败: {e}")
            return 1.0


class LabelSmoothingLoss(BaseLoss):
    """标签平滑损失"""
    
    def __init__(self, smoothing: float = 0.1, weight: float = 1.0):
        super().__init__("label_smoothing", weight)
        self.smoothing = smoothing
    
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算标签平滑损失"""
        try:
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # PyTorch实现
                log_probs = F.log_softmax(predictions, dim=-1)
                
                if targets.dim() == 1:
                    # 标签格式，转换为one-hot
                    num_classes = predictions.size(-1)
                    targets_one_hot = torch.zeros_like(predictions).scatter(1, targets.unsqueeze(1), 1)
                else:
                    targets_one_hot = targets
                
                # 标签平滑
                smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / predictions.size(-1)
                
                loss = -torch.sum(smooth_targets * log_probs, dim=-1)
                loss_value = float(loss.mean().item())
            else:
                # NumPy实现
                loss_value = self._numpy_label_smoothing(predictions, targets)
            
            self.update_history(loss_value)
            return loss_value * self.weight
            
        except Exception as e:
            logger.error(f"标签平滑损失计算失败: {e}")
            return 0.0
    
    def _numpy_label_smoothing(self, predictions: Any, targets: Any) -> float:
        """NumPy实现的标签平滑损失"""
        try:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # Log softmax
            log_probs = predictions - np.log(np.sum(np.exp(predictions), axis=-1, keepdims=True))
            
            # 处理标签
            if targets.ndim == 1:
                num_classes = predictions.shape[-1]
                targets_one_hot = np.eye(num_classes)[targets]
            else:
                targets_one_hot = targets
            
            # 标签平滑
            smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / predictions.shape[-1]
            
            # 损失
            loss = -np.sum(smooth_targets * log_probs, axis=-1)
            
            return float(np.mean(loss))
            
        except Exception as e:
            logger.error(f"NumPy标签平滑损失计算失败: {e}")
            return 1.0


class WeightedCrossEntropyLoss(BaseLoss):
    """加权交叉熵损失 - 用于处理类别不平衡"""
    
    def __init__(self, class_weights: List[float], weight: float = 1.0):
        super().__init__("weighted_cross_entropy", weight)
        self.class_weights = class_weights
    
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算加权交叉熵损失"""
        try:
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # PyTorch实现
                weights = torch.tensor(self.class_weights, dtype=predictions.dtype, device=predictions.device)
                loss = F.cross_entropy(predictions, targets, weight=weights)
                loss_value = float(loss.item())
            else:
                # NumPy实现
                loss_value = self._numpy_weighted_cross_entropy(predictions, targets)
            
            self.update_history(loss_value)
            return loss_value * self.weight
            
        except Exception as e:
            logger.error(f"加权交叉熵损失计算失败: {e}")
            return 0.0
    
    def _numpy_weighted_cross_entropy(self, predictions: Any, targets: Any) -> float:
        """NumPy实现的加权交叉熵损失"""
        try:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # Softmax
            exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            softmax_preds = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
            
            # 加权交叉熵
            epsilon = 1e-12
            softmax_preds = np.clip(softmax_preds, epsilon, 1.0 - epsilon)
            
            if targets.ndim == 1:
                weights = np.array(self.class_weights)[targets]
                loss = -weights * np.log(softmax_preds[np.arange(len(targets)), targets])
            else:
                # one-hot格式
                class_weights_expanded = np.array(self.class_weights)[None, :]
                loss = -np.sum(targets * class_weights_expanded * np.log(softmax_preds), axis=-1)
            
            return float(np.mean(loss))
            
        except Exception as e:
            logger.error(f"NumPy加权交叉熵计算失败: {e}")
            return 1.0


class AnomalyDetectionLoss(BaseLoss):
    """异常检测专用损失函数"""
    
    def __init__(self, normal_weight: float = 1.0, anomaly_weight: float = 2.0, 
                 margin: float = 1.0, weight: float = 1.0):
        super().__init__("anomaly_detection", weight)
        self.normal_weight = normal_weight
        self.anomaly_weight = anomaly_weight
        self.margin = margin
    
    def compute_loss(self, predictions: Any, targets: Any, **kwargs) -> float:
        """计算异常检测损失"""
        try:
            # 组合多种损失：分类损失 + 对比损失
            classification_loss = self._compute_classification_loss(predictions, targets)
            contrastive_loss = self._compute_contrastive_loss(predictions, targets)
            
            total_loss = classification_loss + 0.1 * contrastive_loss
            
            self.update_history(total_loss)
            return total_loss * self.weight
            
        except Exception as e:
            logger.error(f"异常检测损失计算失败: {e}")
            return 0.0
    
    def _compute_classification_loss(self, predictions: Any, targets: Any) -> float:
        """计算分类损失"""
        try:
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # 使用加权交叉熵
                weights = torch.tensor([self.normal_weight, self.anomaly_weight], 
                                     dtype=predictions.dtype, device=predictions.device)
                loss = F.cross_entropy(predictions, targets, weight=weights)
                return float(loss.item())
            else:
                # NumPy实现
                weights = [self.normal_weight, self.anomaly_weight]
                weighted_ce = WeightedCrossEntropyLoss(weights)
                return weighted_ce._numpy_weighted_cross_entropy(predictions, targets)
                
        except Exception as e:
            logger.error(f"分类损失计算失败: {e}")
            return 1.0
    
    def _compute_contrastive_loss(self, predictions: Any, targets: Any) -> float:
        """计算对比损失"""
        try:
            # 简化的对比损失实现
            if TORCH_AVAILABLE and hasattr(predictions, 'device'):
                # 计算特征距离
                features = F.normalize(predictions, p=2, dim=1)
                
                # 构建正负样本对
                batch_size = features.size(0)
                loss = 0.0
                count = 0
                
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        distance = torch.norm(features[i] - features[j], p=2)
                        
                        if targets[i] == targets[j]:  # 同类样本
                            loss += distance ** 2
                        else:  # 异类样本
                            loss += torch.clamp(self.margin - distance, min=0.0) ** 2
                        count += 1
                
                return float(loss / max(count, 1))
            else:
                # NumPy简化实现
                return 0.1
                
        except Exception as e:
            logger.error(f"对比损失计算失败: {e}")
            return 0.0


class MultiTaskLoss(BaseLoss):
    """多任务学习损失函数"""
    
    def __init__(self, task_weights: Dict[str, float], 
                 uncertainty_weighting: bool = False, weight: float = 1.0):
        super().__init__("multi_task", weight)
        self.task_weights = task_weights
        self.uncertainty_weighting = uncertainty_weighting
        
        # 任务特定的损失函数
        self.task_losses = {}
        self._initialize_task_losses()
        
        # 不确定性权重（如果启用）
        self.log_vars = {}
        if uncertainty_weighting:
            for task in task_weights.keys():
                self.log_vars[task] = 0.0  # 初始化为0
    
    def _initialize_task_losses(self):
        """初始化任务特定的损失函数"""
        for task in self.task_weights.keys():
            if task == "sentiment_analysis":
                self.task_losses[task] = CrossEntropyLoss()
            elif task == "style_recognition":
                self.task_losses[task] = CrossEntropyLoss()
            elif task == "anomaly_detection":
                self.task_losses[task] = AnomalyDetectionLoss()
            else:
                self.task_losses[task] = CrossEntropyLoss()
    
    def compute_loss(self, predictions: Dict[str, Any], targets: Dict[str, Any], 
                    **kwargs) -> float:
        """计算多任务损失"""
        try:
            total_loss = 0.0
            task_losses = {}
            
            for task, pred in predictions.items():
                if task in targets and task in self.task_losses:
                    # 计算单任务损失
                    task_loss = self.task_losses[task].compute_loss(pred, targets[task])
                    task_losses[task] = task_loss
                    
                    # 应用权重
                    if self.uncertainty_weighting:
                        # 使用不确定性加权
                        log_var = self.log_vars.get(task, 0.0)
                        precision = math.exp(-log_var)
                        weighted_loss = precision * task_loss + log_var
                    else:
                        # 使用固定权重
                        weighted_loss = self.task_weights.get(task, 1.0) * task_loss
                    
                    total_loss += weighted_loss
            
            # 更新不确定性权重（简化版本）
            if self.uncertainty_weighting:
                self._update_uncertainty_weights(task_losses)
            
            self.update_history(total_loss)
            return total_loss * self.weight
            
        except Exception as e:
            logger.error(f"多任务损失计算失败: {e}")
            return 0.0
    
    def _update_uncertainty_weights(self, task_losses: Dict[str, float]):
        """更新不确定性权重"""
        # 简化的更新策略
        learning_rate = 0.01
        
        for task, loss in task_losses.items():
            if task in self.log_vars:
                # 基于损失大小调整不确定性
                if loss > 1.0:
                    self.log_vars[task] += learning_rate
                else:
                    self.log_vars[task] -= learning_rate
                
                # 限制范围
                self.log_vars[task] = max(-2.0, min(2.0, self.log_vars[task]))
    
    def get_task_weights(self) -> Dict[str, float]:
        """获取当前任务权重"""
        if self.uncertainty_weighting:
            # 返回基于不确定性的权重
            weights = {}
            for task in self.task_weights.keys():
                log_var = self.log_vars.get(task, 0.0)
                weights[task] = math.exp(-log_var)
            return weights
        else:
            return self.task_weights.copy()


class LossManager:
    """损失函数管理器"""
    
    def __init__(self):
        self.losses = {}
        self.history = []
    
    def register_loss(self, name: str, loss_function: BaseLoss):
        """注册损失函数"""
        self.losses[name] = loss_function
        logger.info(f"损失函数已注册: {name}")
    
    def compute_total_loss(self, predictions: Any, targets: Any, 
                          loss_names: Optional[List[str]] = None) -> float:
        """计算总损失"""
        if loss_names is None:
            loss_names = list(self.losses.keys())
        
        total_loss = 0.0
        loss_details = {}
        
        for name in loss_names:
            if name in self.losses:
                try:
                    loss_value = self.losses[name].compute_loss(predictions, targets)
                    loss_details[name] = loss_value
                    total_loss += loss_value
                except Exception as e:
                    logger.error(f"损失函数 {name} 计算失败: {e}")
        
        # 记录历史
        self.history.append({
            'total_loss': total_loss,
            'loss_details': loss_details,
            'timestamp': f"{len(self.history)}"
        })
        
        return total_loss
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """获取损失统计信息"""
        stats = {
            'registered_losses': list(self.losses.keys()),
            'total_computations': len(self.history)
        }
        
        if self.history:
            recent_losses = [h['total_loss'] for h in self.history[-100:]]
            stats['recent_average'] = sum(recent_losses) / len(recent_losses)
            stats['recent_min'] = min(recent_losses)
            stats['recent_max'] = max(recent_losses)
        
        # 各损失函数的统计
        for name, loss_func in self.losses.items():
            stats[f'{name}_average'] = loss_func.get_average_loss()
        
        return stats


# 工厂函数
def create_loss_function(loss_type: LossType, **kwargs) -> BaseLoss:
    """
    创建损失函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
        
    Returns:
        BaseLoss: 损失函数实例
    """
    if loss_type == LossType.CROSS_ENTROPY:
        return CrossEntropyLoss(**kwargs)
    elif loss_type == LossType.FOCAL_LOSS:
        return FocalLoss(**kwargs)
    elif loss_type == LossType.LABEL_SMOOTHING:
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == LossType.WEIGHTED_CE:
        return WeightedCrossEntropyLoss(**kwargs)
    elif loss_type == LossType.ANOMALY_DETECTION:
        return AnomalyDetectionLoss(**kwargs)
    elif loss_type == LossType.MULTI_TASK:
        return MultiTaskLoss(**kwargs)
    else:
        logger.warning(f"未知的损失函数类型: {loss_type}, 使用默认交叉熵损失")
        return CrossEntropyLoss()


if __name__ == "__main__":
    # 使用示例
    print("损失函数测试:")
    
    # 创建损失管理器
    manager = LossManager()
    
    # 注册不同的损失函数
    ce_loss = create_loss_function(LossType.CROSS_ENTROPY)
    focal_loss = create_loss_function(LossType.FOCAL_LOSS, alpha=1.0, gamma=2.0)
    
    manager.register_loss("cross_entropy", ce_loss)
    manager.register_loss("focal_loss", focal_loss)
    
    # 测试损失计算
    if TORCH_AVAILABLE:
        import torch
        predictions = torch.randn(4, 3)  # batch_size=4, num_classes=3
        targets = torch.tensor([0, 1, 2, 1])
        
        total_loss = manager.compute_total_loss(predictions, targets)
        print(f"总损失: {total_loss:.4f}")
    else:
        import numpy as np
        predictions = np.random.randn(4, 3)
        targets = np.array([0, 1, 2, 1])
        
        total_loss = manager.compute_total_loss(predictions, targets)
        print(f"总损失: {total_loss:.4f}")
    
    # 获取统计信息
    stats = manager.get_loss_statistics()
    print(f"损失统计: {stats}")
    
    # 测试多任务损失
    task_weights = {"task1": 0.5, "task2": 0.5}
    multi_task_loss = create_loss_function(LossType.MULTI_TASK, task_weights=task_weights)
    print(f"多任务损失函数创建成功: {multi_task_loss.name}") 
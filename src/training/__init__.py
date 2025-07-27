"""
训练系统模块
包含通用训练器、Qwen微调器、损失函数、训练回调等组件
"""

from .trainer import Trainer, TrainingConfig, TrainingPhase, create_trainer
from .qwen_fine_tuner import QwenFineTuner, QwenFineTuningConfig, create_qwen_fine_tuner
from .loss_functions import (
    BaseLoss, CrossEntropyLoss, FocalLoss, LabelSmoothingLoss, 
    WeightedCrossEntropyLoss, AnomalyDetectionLoss, MultiTaskLoss,
    LossManager, LossType, create_loss_function
)
from .callbacks import (
    BaseCallback, ModelCheckpointCallback, EarlyStoppingCallback,
    LearningRateSchedulerCallback, ProgressMonitorCallback,
    CallbackManager, CallbackEvent, create_default_callbacks
)

__all__ = [
    # 训练器
    'Trainer',
    'TrainingConfig', 
    'TrainingPhase',
    'create_trainer',
    
    # Qwen微调器
    'QwenFineTuner',
    'QwenFineTuningConfig',
    'create_qwen_fine_tuner',
    
    # 损失函数
    'BaseLoss',
    'CrossEntropyLoss',
    'FocalLoss', 
    'LabelSmoothingLoss',
    'WeightedCrossEntropyLoss',
    'AnomalyDetectionLoss',
    'MultiTaskLoss',
    'LossManager',
    'LossType',
    'create_loss_function',
    
    # 回调函数
    'BaseCallback',
    'ModelCheckpointCallback',
    'EarlyStoppingCallback',
    'LearningRateSchedulerCallback',
    'ProgressMonitorCallback',
    'CallbackManager',
    'CallbackEvent',
    'create_default_callbacks'
] 
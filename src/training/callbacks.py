"""
训练回调函数
提供训练过程中的各种回调功能，包括模型保存、早停、学习率调度、监控等
"""

import os
import json
import time
import pickle
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class CallbackEvent(Enum):
    """回调事件枚举"""
    TRAIN_BEGIN = "on_train_begin"
    TRAIN_END = "on_train_end"
    EPOCH_BEGIN = "on_epoch_begin"
    EPOCH_END = "on_epoch_end"
    BATCH_BEGIN = "on_batch_begin"
    BATCH_END = "on_batch_end"
    VALIDATION_BEGIN = "on_validation_begin"
    VALIDATION_END = "on_validation_end"
    ERROR = "on_train_error"


@dataclass
class CallbackState:
    """回调状态数据"""
    epoch: int = 0
    batch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class BaseCallback(ABC):
    """回调函数基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.state = CallbackState()
    
    def enable(self):
        """启用回调"""
        self.enabled = True
    
    def disable(self):
        """禁用回调"""
        self.enabled = False
    
    def update_state(self, **kwargs):
        """更新状态"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    # 回调方法（子类可选择性重写）
    def on_train_begin(self, data: Optional[Dict[str, Any]] = None):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, data: Optional[Dict[str, Any]] = None):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, data: Optional[Dict[str, Any]] = None):
        """Epoch开始时调用"""
        pass
    
    def on_epoch_end(self, data: Optional[Dict[str, Any]] = None):
        """Epoch结束时调用"""
        pass
    
    def on_batch_begin(self, data: Optional[Dict[str, Any]] = None):
        """Batch开始时调用"""
        pass
    
    def on_batch_end(self, data: Optional[Dict[str, Any]] = None):
        """Batch结束时调用"""
        pass
    
    def on_validation_begin(self, data: Optional[Dict[str, Any]] = None):
        """验证开始时调用"""
        pass
    
    def on_validation_end(self, data: Optional[Dict[str, Any]] = None):
        """验证结束时调用"""
        pass
    
    def on_train_error(self, data: Optional[Dict[str, Any]] = None):
        """训练错误时调用"""
        pass


class ModelCheckpointCallback(BaseCallback):
    """模型检查点保存回调"""
    
    def __init__(self, save_dir: str = "checkpoints", 
                 save_best_only: bool = True,
                 monitor: str = "eval_loss",
                 mode: str = "min",
                 save_freq: str = "epoch",
                 save_top_k: int = 3):
        super().__init__("model_checkpoint")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode  # 'min' or 'max'
        self.save_freq = save_freq  # 'epoch' or 'batch'
        self.save_top_k = save_top_k
        
        # 跟踪最佳模型
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        
        # 保存的模型列表
        self.saved_models = []
        
        self.file_manager = get_file_manager()
        
        logger.info(f"模型检查点回调初始化: 监控{monitor}, 模式{mode}")
    
    def on_epoch_end(self, data: Optional[Dict[str, Any]] = None):
        """Epoch结束时保存模型"""
        if not self.enabled or self.save_freq != "epoch":
            return
        
        try:
            if data and 'eval_metrics' in data and data['eval_metrics']:
                eval_metrics = data['eval_metrics']
                current_metric = getattr(eval_metrics, self.monitor, None)
                
                if current_metric is not None:
                    should_save = self._should_save_model(current_metric)
                    
                    if should_save:
                        self._save_model(data, current_metric)
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def on_batch_end(self, data: Optional[Dict[str, Any]] = None):
        """Batch结束时保存模型"""
        if not self.enabled or self.save_freq != "batch":
            return
        
        try:
            # 每N个batch保存一次（可配置）
            if data and data.get('global_step', 0) % 1000 == 0:
                self._save_model(data, None)
                
        except Exception as e:
            logger.error(f"批次模型保存失败: {e}")
    
    def _should_save_model(self, current_metric: float) -> bool:
        """判断是否应该保存模型"""
        if not self.save_best_only:
            return True
        
        if self.mode == 'min':
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def _save_model(self, data: Dict[str, Any], metric_value: Optional[float]):
        """保存模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epoch = data.get('epoch', 0)
            
            # 创建模型文件名
            if metric_value is not None:
                model_name = f"model_epoch_{epoch}_{self.monitor}_{metric_value:.4f}_{timestamp}"
            else:
                step = data.get('global_step', 0)
                model_name = f"model_step_{step}_{timestamp}"
            
            model_path = self.save_dir / f"{model_name}.pth"
            
            # 保存模型（需要从训练器获取模型）
            model = data.get('model')
            if model:
                if hasattr(model, 'state_dict'):
                    # PyTorch模型
                    try:
                        import torch
                        torch.save(model.state_dict(), model_path)
                    except ImportError:
                        # 使用pickle备用
                        with open(model_path.with_suffix('.pkl'), 'wb') as f:
                            pickle.dump(model, f)
                        model_path = model_path.with_suffix('.pkl')
                else:
                    # 其他类型模型
                    with open(model_path.with_suffix('.pkl'), 'wb') as f:
                        pickle.dump(model, f)
                    model_path = model_path.with_suffix('.pkl')
                
                # 保存元数据
                metadata = {
                    'epoch': epoch,
                    'metric_value': metric_value,
                    'monitor': self.monitor,
                    'save_time': timestamp,
                    'model_path': str(model_path)
                }
                
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # 更新记录
                self.saved_models.append({
                    'path': str(model_path),
                    'metric_value': metric_value,
                    'epoch': epoch,
                    'timestamp': timestamp
                })
                
                # 更新最佳模型
                if metric_value is not None and self._should_save_model(metric_value):
                    self.best_metric = metric_value
                    self.best_model_path = str(model_path)
                
                # 清理旧模型
                self._cleanup_old_models()
                
                logger.info(f"模型已保存: {model_path}")
            
        except Exception as e:
            logger.error(f"模型保存过程失败: {e}")
    
    def _cleanup_old_models(self):
        """清理旧模型，保持top-k"""
        if len(self.saved_models) <= self.save_top_k:
            return
        
        # 按指标排序
        if self.mode == 'min':
            self.saved_models.sort(key=lambda x: x['metric_value'] if x['metric_value'] is not None else float('inf'))
        else:
            self.saved_models.sort(key=lambda x: x['metric_value'] if x['metric_value'] is not None else float('-inf'), reverse=True)
        
        # 删除多余的模型
        models_to_remove = self.saved_models[self.save_top_k:]
        self.saved_models = self.saved_models[:self.save_top_k]
        
        for model_info in models_to_remove:
            try:
                model_path = Path(model_info['path'])
                if model_path.exists():
                    model_path.unlink()
                
                # 删除元数据文件
                metadata_path = model_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"删除旧模型: {model_path}")
                
            except Exception as e:
                logger.error(f"删除模型失败: {e}")


class EarlyStoppingCallback(BaseCallback):
    """早停回调"""
    
    def __init__(self, monitor: str = "eval_loss", 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = "min",
                 restore_best_weights: bool = True):
        super().__init__("early_stopping")
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # 状态跟踪
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        logger.info(f"早停回调初始化: 监控{monitor}, 耐心值{patience}, 模式{mode}")
    
    def on_epoch_end(self, data: Optional[Dict[str, Any]] = None):
        """Epoch结束时检查早停条件"""
        if not self.enabled:
            return
        
        try:
            if data and 'eval_metrics' in data and data['eval_metrics']:
                eval_metrics = data['eval_metrics']
                current_metric = getattr(eval_metrics, self.monitor, None)
                
                if current_metric is not None:
                    if self._is_improvement(current_metric):
                        self.best_metric = current_metric
                        self.wait = 0
                        
                        # 保存最佳权重
                        if self.restore_best_weights and 'model' in data:
                            self._save_best_weights(data['model'])
                    else:
                        self.wait += 1
                        
                        if self.wait >= self.patience:
                            self.stopped_epoch = data.get('epoch', 0)
                            logger.info(f"早停触发: epoch {self.stopped_epoch}, 最佳{self.monitor}: {self.best_metric}")
                            
                            # 恢复最佳权重
                            if self.restore_best_weights and self.best_weights and 'model' in data:
                                self._restore_best_weights(data['model'])
                            
                            # 停止训练
                            trainer = data.get('trainer')
                            if trainer and hasattr(trainer, 'stop_training'):
                                trainer.stop_training()
            
        except Exception as e:
            logger.error(f"早停检查失败: {e}")
    
    def _is_improvement(self, current_metric: float) -> bool:
        """检查是否有改进"""
        if self.mode == 'min':
            return current_metric < self.best_metric - self.min_delta
        else:
            return current_metric > self.best_metric + self.min_delta
    
    def _save_best_weights(self, model: Any):
        """保存最佳权重"""
        try:
            if hasattr(model, 'state_dict'):
                import torch
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                import copy
                self.best_weights = copy.deepcopy(model)
        except Exception as e:
            logger.error(f"保存最佳权重失败: {e}")
    
    def _restore_best_weights(self, model: Any):
        """恢复最佳权重"""
        try:
            if hasattr(model, 'load_state_dict') and isinstance(self.best_weights, dict):
                model.load_state_dict(self.best_weights)
                logger.info("已恢复最佳模型权重")
            elif not hasattr(model, 'load_state_dict'):
                # 对于非PyTorch模型，需要重新设置
                logger.warning("无法恢复最佳权重：不支持的模型类型")
        except Exception as e:
            logger.error(f"恢复最佳权重失败: {e}")


class LearningRateSchedulerCallback(BaseCallback):
    """学习率调度回调"""
    
    def __init__(self, scheduler_type: str = "step",
                 step_size: int = 10,
                 gamma: float = 0.1,
                 patience: int = 5,
                 factor: float = 0.5,
                 min_lr: float = 1e-8):
        super().__init__("lr_scheduler")
        
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        # 状态跟踪
        self.current_lr = None
        self.wait = 0
        self.best_metric = float('inf')
        
        logger.info(f"学习率调度回调初始化: 类型{scheduler_type}")
    
    def on_epoch_end(self, data: Optional[Dict[str, Any]] = None):
        """Epoch结束时调整学习率"""
        if not self.enabled:
            return
        
        try:
            optimizer = data.get('optimizer') if data else None
            if not optimizer:
                return
            
            # 获取当前学习率
            if hasattr(optimizer, 'param_groups'):
                self.current_lr = optimizer.param_groups[0]['lr']
            
            if self.scheduler_type == "step":
                self._step_scheduler(data, optimizer)
            elif self.scheduler_type == "reduce_on_plateau":
                self._reduce_on_plateau_scheduler(data, optimizer)
            elif self.scheduler_type == "cosine":
                self._cosine_scheduler(data, optimizer)
            
        except Exception as e:
            logger.error(f"学习率调度失败: {e}")
    
    def _step_scheduler(self, data: Dict[str, Any], optimizer: Any):
        """步进式学习率调度"""
        epoch = data.get('epoch', 0)
        
        if epoch > 0 and epoch % self.step_size == 0:
            new_lr = max(self.current_lr * self.gamma, self.min_lr)
            self._update_learning_rate(optimizer, new_lr)
    
    def _reduce_on_plateau_scheduler(self, data: Dict[str, Any], optimizer: Any):
        """基于平台的学习率调度"""
        if 'eval_metrics' not in data or not data['eval_metrics']:
            return
        
        eval_metrics = data['eval_metrics']
        current_metric = getattr(eval_metrics, 'eval_loss', None)
        
        if current_metric is not None:
            if current_metric < self.best_metric - 1e-4:
                self.best_metric = current_metric
                self.wait = 0
            else:
                self.wait += 1
                
                if self.wait >= self.patience:
                    new_lr = max(self.current_lr * self.factor, self.min_lr)
                    self._update_learning_rate(optimizer, new_lr)
                    self.wait = 0
    
    def _cosine_scheduler(self, data: Dict[str, Any], optimizer: Any):
        """余弦学习率调度"""
        epoch = data.get('epoch', 0)
        total_epochs = data.get('total_epochs', 100)
        
        if total_epochs > 0:
            import math
            cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
            new_lr = max(self.min_lr + (self.current_lr - self.min_lr) * cosine_factor, self.min_lr)
            self._update_learning_rate(optimizer, new_lr)
    
    def _update_learning_rate(self, optimizer: Any, new_lr: float):
        """更新学习率"""
        try:
            if hasattr(optimizer, 'param_groups'):
                old_lr = optimizer.param_groups[0]['lr']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                logger.info(f"学习率更新: {old_lr:.2e} -> {new_lr:.2e}")
                self.current_lr = new_lr
            
        except Exception as e:
            logger.error(f"学习率更新失败: {e}")


class ProgressMonitorCallback(BaseCallback):
    """训练进度监控回调"""
    
    def __init__(self, log_interval: int = 10,
                 save_logs: bool = True,
                 log_file: str = "training_logs.json"):
        super().__init__("progress_monitor")
        
        self.log_interval = log_interval
        self.save_logs = save_logs
        self.log_file = Path(log_file)
        
        # 训练记录
        self.training_logs = []
        self.start_time = None
        
        logger.info(f"进度监控回调初始化: 日志间隔{log_interval}")
    
    def on_train_begin(self, data: Optional[Dict[str, Any]] = None):
        """训练开始时记录"""
        self.start_time = time.time()
        log_entry = {
            'event': 'train_begin',
            'timestamp': datetime.now().isoformat(),
            'message': '训练开始'
        }
        
        self._add_log_entry(log_entry)
        logger.info("训练开始")
    
    def on_train_end(self, data: Optional[Dict[str, Any]] = None):
        """训练结束时记录"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        log_entry = {
            'event': 'train_end',
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'message': f'训练完成，总耗时: {total_time:.2f}秒'
        }
        
        self._add_log_entry(log_entry)
        logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        
        # 保存日志
        if self.save_logs:
            self._save_logs()
    
    def on_epoch_end(self, data: Optional[Dict[str, Any]] = None):
        """Epoch结束时记录"""
        if not data:
            return
        
        epoch = data.get('epoch', 0)
        train_metrics = data.get('train_metrics')
        eval_metrics = data.get('eval_metrics')
        
        log_entry = {
            'event': 'epoch_end',
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        if train_metrics:
            log_entry['train_loss'] = getattr(train_metrics, 'train_loss', None)
            log_entry['learning_rate'] = getattr(train_metrics, 'learning_rate', None)
        
        if eval_metrics:
            log_entry['eval_loss'] = getattr(eval_metrics, 'eval_loss', None)
        
        self._add_log_entry(log_entry)
        
        # 记录进度
        if epoch % self.log_interval == 0:
            message = f"Epoch {epoch}"
            if train_metrics:
                message += f", Train Loss: {getattr(train_metrics, 'train_loss', 'N/A'):.4f}"
            if eval_metrics:
                message += f", Eval Loss: {getattr(eval_metrics, 'eval_loss', 'N/A'):.4f}"
            
            logger.info(message)
    
    def on_train_error(self, data: Optional[Dict[str, Any]] = None):
        """训练错误时记录"""
        error_msg = data.get('error', 'Unknown error') if data else 'Unknown error'
        
        log_entry = {
            'event': 'train_error',
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'message': f'训练出错: {error_msg}'
        }
        
        self._add_log_entry(log_entry)
        logger.error(f"训练出错: {error_msg}")
    
    def _add_log_entry(self, entry: Dict[str, Any]):
        """添加日志条目"""
        self.training_logs.append(entry)
        
        # 限制日志大小
        if len(self.training_logs) > 10000:
            self.training_logs = self.training_logs[-5000:]
    
    def _save_logs(self):
        """保存训练日志"""
        try:
            log_data = {
                'training_logs': self.training_logs,
                'summary': {
                    'total_entries': len(self.training_logs),
                    'start_time': self.training_logs[0]['timestamp'] if self.training_logs else None,
                    'end_time': self.training_logs[-1]['timestamp'] if self.training_logs else None
                }
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"训练日志已保存: {self.log_file}")
            
        except Exception as e:
            logger.error(f"保存训练日志失败: {e}")


class CallbackManager:
    """回调管理器"""
    
    def __init__(self):
        self.callbacks: List[BaseCallback] = []
    
    def add_callback(self, callback: BaseCallback):
        """添加回调"""
        self.callbacks.append(callback)
        logger.info(f"回调已添加: {callback.name}")
    
    def remove_callback(self, callback_name: str):
        """移除回调"""
        self.callbacks = [cb for cb in self.callbacks if cb.name != callback_name]
        logger.info(f"回调已移除: {callback_name}")
    
    def trigger_event(self, event: CallbackEvent, data: Optional[Dict[str, Any]] = None):
        """触发回调事件"""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    method = getattr(callback, event.value, None)
                    if method and callable(method):
                        method(data)
                except Exception as e:
                    logger.error(f"回调 {callback.name} 执行失败: {e}")
    
    def get_callback_status(self) -> Dict[str, Any]:
        """获取回调状态"""
        return {
            'total_callbacks': len(self.callbacks),
            'enabled_callbacks': [cb.name for cb in self.callbacks if cb.enabled],
            'disabled_callbacks': [cb.name for cb in self.callbacks if not cb.enabled]
        }


# 预设回调组合
def create_default_callbacks(save_dir: str = "outputs") -> CallbackManager:
    """
    创建默认回调组合
    
    Args:
        save_dir: 保存目录
        
    Returns:
        CallbackManager: 配置好的回调管理器
    """
    manager = CallbackManager()
    
    # 模型检查点
    checkpoint_callback = ModelCheckpointCallback(
        save_dir=f"{save_dir}/checkpoints",
        save_best_only=True,
        monitor="eval_loss",
        mode="min"
    )
    manager.add_callback(checkpoint_callback)
    
    # 早停
    early_stopping_callback = EarlyStoppingCallback(
        monitor="eval_loss",
        patience=10,
        mode="min"
    )
    manager.add_callback(early_stopping_callback)
    
    # 学习率调度
    lr_scheduler_callback = LearningRateSchedulerCallback(
        scheduler_type="reduce_on_plateau",
        patience=5,
        factor=0.5
    )
    manager.add_callback(lr_scheduler_callback)
    
    # 进度监控
    progress_callback = ProgressMonitorCallback(
        log_interval=10,
        log_file=f"{save_dir}/training_logs.json"
    )
    manager.add_callback(progress_callback)
    
    return manager


if __name__ == "__main__":
    # 使用示例
    print("训练回调测试:")
    
    # 创建回调管理器
    manager = create_default_callbacks("test_outputs")
    
    print(f"回调状态: {manager.get_callback_status()}")
    
    # 模拟训练过程
    print("\n模拟训练过程:")
    
    # 训练开始
    manager.trigger_event(CallbackEvent.TRAIN_BEGIN)
    
    # 模拟几个epoch
    for epoch in range(3):
        # Epoch开始
        manager.trigger_event(CallbackEvent.EPOCH_BEGIN, {'epoch': epoch})
        
        # 模拟训练指标
        from dataclasses import dataclass
        
        @dataclass
        class MockMetrics:
            train_loss: float
            eval_loss: float
            learning_rate: float
        
        train_metrics = MockMetrics(
            train_loss=1.0 - epoch * 0.1,
            eval_loss=0.8 - epoch * 0.05,
            learning_rate=0.001
        )
        
        eval_metrics = MockMetrics(
            train_loss=0.0,
            eval_loss=0.8 - epoch * 0.05,
            learning_rate=0.001
        )
        
        # Epoch结束
        manager.trigger_event(CallbackEvent.EPOCH_END, {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        })
    
    # 训练结束
    manager.trigger_event(CallbackEvent.TRAIN_END)
    
    print("回调测试完成") 
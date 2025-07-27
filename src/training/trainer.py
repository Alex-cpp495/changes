"""
通用训练器
支持多种模型的训练、验证、超参数调优和模型保存
"""

import os
import json
import time
import gc
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import pickle

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class TrainingPhase(Enum):
    """训练阶段枚举"""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizerType(Enum):
    """优化器类型枚举"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


@dataclass
class TrainingConfig:
    """训练配置"""
    model_type: str
    model_config: Dict[str, Any]
    
    # 训练参数
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # 优化器设置
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: str = "linear"
    
    # 验证设置
    validation_split: float = 0.2
    validation_strategy: str = "epoch"  # epoch, steps
    validation_steps: int = 500
    
    # 保存设置
    save_strategy: str = "epoch"  # epoch, steps, no
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # 早停设置
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 1e-4
    
    # 设备设置
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 日志设置
    logging_steps: int = 10
    eval_steps: int = 100
    
    # 其他设置
    seed: int = 42
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    
    # 自定义指标
    custom_metrics: Dict[str, float] = None
    
    # 时间统计
    epoch_time: float = 0.0
    training_time: float = 0.0
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class TrainingState:
    """训练状态管理"""
    
    def __init__(self):
        self.phase = TrainingPhase.INITIALIZATION
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_model_checkpoint = None
        self.patience_counter = 0
        self.is_training = False
        self.should_stop = False
        
        # 历史记录
        self.train_history: List[TrainingMetrics] = []
        self.eval_history: List[TrainingMetrics] = []
        
        # 时间统计
        self.start_time = None
        self.end_time = None
        
        # 线程安全锁
        self.lock = threading.Lock()
    
    def update_metrics(self, metrics: TrainingMetrics, is_eval: bool = False):
        """更新训练指标"""
        with self.lock:
            if is_eval:
                self.eval_history.append(metrics)
            else:
                self.train_history.append(metrics)
    
    def should_early_stop(self, current_metric: float, patience: int, min_delta: float) -> bool:
        """检查是否应该早停"""
        with self.lock:
            if current_metric < self.best_metric - min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
                return self.patience_counter >= patience


class Trainer:
    """
    通用训练器
    
    提供灵活的模型训练功能：
    1. 模型训练管理 - 训练循环、验证、测试
    2. 优化器配置 - 多种优化器、学习率调度
    3. 训练监控 - 指标跟踪、日志记录、可视化
    4. 模型保存 - 检查点保存、最佳模型保存
    5. 早停机制 - 防止过拟合、节省计算资源
    6. 分布式训练 - 多GPU、数据并行（可扩展）
    
    Args:
        config: 训练配置
        model: 要训练的模型
        
    Attributes:
        config: 训练配置
        model: 训练模型
        state: 训练状态
        callbacks: 回调函数列表
    """
    
    def __init__(self, config: TrainingConfig, model: Any = None):
        """初始化训练器"""
        self.config = config
        self.model = model
        self.state = TrainingState()
        
        self.file_manager = get_file_manager()
        
        # 训练组件
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        
        # 数据加载器
        self.train_dataloader = None
        self.eval_dataloader = None
        self.test_dataloader = None
        
        # 回调函数
        self.callbacks: List[Callable] = []
        
        # 输出目录
        self.output_dir = Path(f"data/training/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        logger.info("训练器初始化完成")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def set_model(self, model: Any):
        """设置训练模型"""
        self.model = model
        logger.info(f"设置训练模型: {type(model).__name__}")
    
    def set_optimizer(self, optimizer: Any):
        """设置优化器"""
        self.optimizer = optimizer
        logger.info(f"设置优化器: {type(optimizer).__name__}")
    
    def set_loss_function(self, loss_function: Any):
        """设置损失函数"""
        self.loss_function = loss_function
        logger.info(f"设置损失函数: {type(loss_function).__name__}")
    
    def set_data_loaders(self, train_dataloader: Any, eval_dataloader: Any = None, test_dataloader: Any = None):
        """设置数据加载器"""
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        
        logger.info(f"设置数据加载器 - 训练: {len(train_dataloader) if train_dataloader else 0}, "
                   f"验证: {len(eval_dataloader) if eval_dataloader else 0}, "
                   f"测试: {len(test_dataloader) if test_dataloader else 0}")
    
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)
        logger.info(f"添加回调函数: {callback.__name__ if hasattr(callback, '__name__') else str(callback)}")
    
    def train(self) -> Dict[str, Any]:
        """
        开始训练
        
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            logger.info("开始训练...")
            
            # 检查必要组件
            self._validate_training_setup()
            
            # 初始化训练状态
            self.state.phase = TrainingPhase.TRAINING
            self.state.is_training = True
            self.state.start_time = time.time()
            
            # 调用训练开始回调
            self._call_callbacks('on_train_begin')
            
            # 训练循环
            for epoch in range(self.config.epochs):
                if self.state.should_stop:
                    logger.info("训练被手动停止")
                    break
                
                self.state.epoch = epoch
                
                # 训练一个epoch
                train_metrics = self._train_epoch()
                
                # 验证
                eval_metrics = None
                if self.eval_dataloader and self._should_evaluate():
                    eval_metrics = self._evaluate()
                
                # 保存模型
                if self._should_save():
                    self._save_checkpoint()
                
                # 早停检查
                if self._check_early_stopping(eval_metrics):
                    logger.info(f"早停触发，在epoch {epoch}")
                    break
                
                # 调用epoch结束回调
                self._call_callbacks('on_epoch_end', {
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics
                })
            
            # 训练完成
            self.state.phase = TrainingPhase.COMPLETED
            self.state.is_training = False
            self.state.end_time = time.time()
            
            # 调用训练结束回调
            self._call_callbacks('on_train_end')
            
            # 生成训练报告
            training_result = self._generate_training_report()
            
            logger.info("训练完成")
            return training_result
            
        except Exception as e:
            self.state.phase = TrainingPhase.FAILED
            self.state.is_training = False
            logger.error(f"训练失败: {e}")
            
            # 调用失败回调
            self._call_callbacks('on_train_error', {'error': str(e)})
            
            raise Exception(f"训练失败: {e}")
    
    def _validate_training_setup(self):
        """验证训练设置"""
        if self.model is None:
            raise ValueError("未设置训练模型")
        
        if self.train_dataloader is None:
            raise ValueError("未设置训练数据加载器")
        
        if self.optimizer is None:
            logger.warning("未设置优化器，将使用默认优化器")
            self._create_default_optimizer()
        
        if self.loss_function is None:
            logger.warning("未设置损失函数，将使用默认损失函数")
            self._create_default_loss_function()
    
    def _create_default_optimizer(self):
        """创建默认优化器"""
        try:
            import torch
            
            if hasattr(self.model, 'parameters'):
                if self.config.optimizer == OptimizerType.ADAM:
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                elif self.config.optimizer == OptimizerType.ADAMW:
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                else:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
                
                logger.info(f"创建默认优化器: {type(self.optimizer).__name__}")
            
        except ImportError:
            logger.warning("无法创建默认优化器，请手动设置")
    
    def _create_default_loss_function(self):
        """创建默认损失函数"""
        try:
            import torch.nn as nn
            self.loss_function = nn.CrossEntropyLoss()
            logger.info("创建默认损失函数: CrossEntropyLoss")
        except ImportError:
            logger.warning("无法创建默认损失函数，请手动设置")
    
    def _train_epoch(self) -> TrainingMetrics:
        """训练一个epoch"""
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        # 设置模型为训练模式
        if hasattr(self.model, 'train'):
            self.model.train()
        
        # 调用epoch开始回调
        self._call_callbacks('on_epoch_begin', {'epoch': self.state.epoch})
        
        try:
            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.state.should_stop:
                    break
                
                # 调用batch开始回调
                self._call_callbacks('on_batch_begin', {
                    'batch_idx': batch_idx,
                    'batch': batch
                })
                
                # 前向传播
                loss = self._training_step(batch)
                
                if loss is not None:
                    total_loss += float(loss)
                    num_batches += 1
                
                self.state.global_step += 1
                
                # 调用batch结束回调
                self._call_callbacks('on_batch_end', {
                    'batch_idx': batch_idx,
                    'loss': loss,
                    'global_step': self.state.global_step
                })
                
                # 日志记录
                if self.state.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    logger.info(f"Epoch {self.state.epoch}, Step {self.state.global_step}, Loss: {avg_loss:.4f}")
        
        except Exception as e:
            logger.error(f"训练epoch时发生错误: {e}")
            raise
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        # 创建训练指标
        train_metrics = TrainingMetrics(
            epoch=self.state.epoch,
            step=self.state.global_step,
            train_loss=avg_loss,
            learning_rate=self._get_current_learning_rate(),
            epoch_time=epoch_time,
            training_time=time.time() - self.state.start_time
        )
        
        # 更新状态
        self.state.update_metrics(train_metrics, is_eval=False)
        
        return train_metrics
    
    def _training_step(self, batch: Any) -> Optional[float]:
        """执行一个训练步骤"""
        try:
            # 这里是简化的训练步骤，实际实现需要根据具体模型调整
            if hasattr(self.model, 'forward') and self.loss_function:
                # 假设batch是(inputs, targets)的形式
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    
                    # 清零梯度
                    if self.optimizer:
                        self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    
                    # 反向传播
                    if hasattr(loss, 'backward'):
                        loss.backward()
                    
                    # 梯度裁剪
                    if self.config.max_grad_norm > 0 and hasattr(self.model, 'parameters'):
                        try:
                            import torch.nn.utils
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        except ImportError:
                            pass
                    
                    # 更新参数
                    if self.optimizer:
                        self.optimizer.step()
                    
                    # 更新学习率
                    if self.scheduler:
                        self.scheduler.step()
                    
                    return float(loss) if hasattr(loss, 'item') else float(loss)
            
            return None
            
        except Exception as e:
            logger.error(f"训练步骤执行失败: {e}")
            return None
    
    def _evaluate(self) -> TrainingMetrics:
        """执行验证"""
        if not self.eval_dataloader:
            return None
        
        logger.info("开始验证...")
        eval_start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        # 设置模型为评估模式
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        try:
            # 禁用梯度计算
            import torch
            with torch.no_grad() if 'torch' in globals() else nullcontext():
                for batch in self.eval_dataloader:
                    if self.state.should_stop:
                        break
                    
                    loss = self._evaluation_step(batch)
                    if loss is not None:
                        total_loss += float(loss)
                        num_batches += 1
        
        except ImportError:
            # 如果没有torch，直接执行评估
            for batch in self.eval_dataloader:
                if self.state.should_stop:
                    break
                
                loss = self._evaluation_step(batch)
                if loss is not None:
                    total_loss += float(loss)
                    num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        eval_time = time.time() - eval_start_time
        
        # 创建验证指标
        eval_metrics = TrainingMetrics(
            epoch=self.state.epoch,
            step=self.state.global_step,
            train_loss=0.0,  # 验证时没有训练损失
            eval_loss=avg_loss,
            learning_rate=self._get_current_learning_rate(),
            epoch_time=eval_time,
            training_time=time.time() - self.state.start_time
        )
        
        # 更新状态
        self.state.update_metrics(eval_metrics, is_eval=True)
        
        logger.info(f"验证完成，损失: {avg_loss:.4f}")
        return eval_metrics
    
    def _evaluation_step(self, batch: Any) -> Optional[float]:
        """执行一个验证步骤"""
        try:
            if hasattr(self.model, 'forward') and self.loss_function:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    
                    return float(loss) if hasattr(loss, 'item') else float(loss)
            
            return None
            
        except Exception as e:
            logger.error(f"验证步骤执行失败: {e}")
            return None
    
    def _should_evaluate(self) -> bool:
        """判断是否应该进行验证"""
        if self.config.validation_strategy == "epoch":
            return True
        elif self.config.validation_strategy == "steps":
            return self.state.global_step % self.config.validation_steps == 0
        return False
    
    def _should_save(self) -> bool:
        """判断是否应该保存模型"""
        if self.config.save_strategy == "no":
            return False
        elif self.config.save_strategy == "epoch":
            return True
        elif self.config.save_strategy == "steps":
            return self.state.global_step % self.config.save_steps == 0
        return False
    
    def _save_checkpoint(self):
        """保存检查点"""
        try:
            checkpoint_dir = self.output_dir / "checkpoints" / f"epoch_{self.state.epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型状态
            if hasattr(self.model, 'state_dict'):
                import torch
                model_path = checkpoint_dir / "model.pth"
                torch.save(self.model.state_dict(), model_path)
            else:
                # 使用pickle保存其他类型的模型
                model_path = checkpoint_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            
            # 保存优化器状态
            if self.optimizer and hasattr(self.optimizer, 'state_dict'):
                optimizer_path = checkpoint_dir / "optimizer.pth"
                torch.save(self.optimizer.state_dict(), optimizer_path)
            
            # 保存训练状态
            state_path = checkpoint_dir / "training_state.json"
            state_data = {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_metric': self.state.best_metric,
                'patience_counter': self.state.patience_counter
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # 保存配置
            config_path = checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"检查点已保存: {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _check_early_stopping(self, eval_metrics: Optional[TrainingMetrics]) -> bool:
        """检查早停条件"""
        if not self.config.early_stopping or not eval_metrics:
            return False
        
        current_metric = eval_metrics.eval_loss
        return self.state.should_early_stop(
            current_metric,
            self.config.patience,
            self.config.min_delta
        )
    
    def _get_current_learning_rate(self) -> float:
        """获取当前学习率"""
        if self.optimizer and hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        return self.config.learning_rate
    
    def _call_callbacks(self, event: str, data: Optional[Dict[str, Any]] = None):
        """调用回调函数"""
        for callback in self.callbacks:
            try:
                if hasattr(callback, event):
                    method = getattr(callback, event)
                    if data:
                        method(data)
                    else:
                        method()
            except Exception as e:
                logger.error(f"回调函数 {event} 执行失败: {e}")
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        total_time = self.state.end_time - self.state.start_time if self.state.end_time else 0
        
        report = {
            'training_summary': {
                'status': self.state.phase.value,
                'epochs_completed': self.state.epoch + 1,
                'total_steps': self.state.global_step,
                'total_time_seconds': total_time,
                'average_time_per_epoch': total_time / (self.state.epoch + 1) if self.state.epoch >= 0 else 0
            },
            'final_metrics': {
                'best_metric': self.state.best_metric,
                'final_train_loss': self.state.train_history[-1].train_loss if self.state.train_history else None,
                'final_eval_loss': self.state.eval_history[-1].eval_loss if self.state.eval_history else None
            },
            'config': asdict(self.config),
            'training_history': [asdict(m) for m in self.state.train_history],
            'eval_history': [asdict(m) for m in self.state.eval_history],
            'output_directory': str(self.output_dir),
            'generated_at': datetime.now().isoformat()
        }
        
        # 保存训练报告
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"训练报告已保存: {report_path}")
        return report
    
    def stop_training(self):
        """停止训练"""
        self.state.should_stop = True
        logger.info("训练停止信号已发送")
    
    def get_training_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'phase': self.state.phase.value,
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'is_training': self.state.is_training,
            'best_metric': self.state.best_metric,
            'patience_counter': self.state.patience_counter,
            'train_history_count': len(self.state.train_history),
            'eval_history_count': len(self.state.eval_history)
        }


# 上下文管理器，用于没有torch时的情况
class nullcontext:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# 工厂函数
def create_trainer(config_dict: Dict[str, Any], model: Any = None) -> Trainer:
    """
    创建训练器实例
    
    Args:
        config_dict: 配置字典
        model: 训练模型
        
    Returns:
        Trainer: 训练器实例
    """
    config = TrainingConfig(**config_dict)
    return Trainer(config, model)


if __name__ == "__main__":
    # 使用示例
    print("训练器测试:")
    
    # 创建训练配置
    config = TrainingConfig(
        model_type="test_model",
        model_config={},
        epochs=3,
        batch_size=16,
        learning_rate=1e-4
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    print(f"训练器创建成功")
    print(f"配置: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    print(f"输出目录: {trainer.output_dir}")
    
    # 获取训练状态
    state = trainer.get_training_state()
    print(f"训练状态: {state}")
    
    # 测试训练报告生成
    report = trainer._generate_training_report()
    print(f"训练报告生成成功，包含 {len(report)} 个部分") 
"""
模型加载器
统一管理各种模型的加载、初始化、缓存和生命周期管理
"""

import gc
import psutil
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import weakref
from enum import Enum
from dataclasses import dataclass

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    model_type: str
    model_path: str
    config: Dict[str, Any]
    memory_usage: int = 0
    load_time: float = 0.0
    last_used: Optional[datetime] = None
    use_count: int = 0
    status: ModelStatus = ModelStatus.UNLOADED
    instance: Any = None


class ModelLoader:
    """
    模型加载器
    
    提供统一的模型管理功能：
    1. 模型注册和发现 - 支持多种模型类型
    2. 智能加载策略 - 延迟加载、预加载、缓存管理
    3. 内存管理 - 内存监控、自动卸载、垃圾回收
    4. 生命周期管理 - 加载状态跟踪、错误处理
    5. 并发安全 - 线程安全的模型访问
    6. 性能优化 - 模型复用、批量处理优化
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 模型加载配置
        models: 注册的模型信息
        loaded_models: 已加载的模型实例
        loading_lock: 加载锁，确保线程安全
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化模型加载器"""
        self.config_path = config_path or "configs/model_config.yaml"
        self.config = self._load_config()
        
        self.file_manager = get_file_manager()
        
        # 模型注册表
        self.models: Dict[str, ModelInfo] = {}
        
        # 加载器注册表
        self.loaders: Dict[str, Callable] = {}
        
        # 线程安全锁
        self.loading_lock = threading.RLock()
        self.access_lock = threading.RLock()
        
        # 内存监控
        self.memory_threshold = self.config.get('memory_management', {}).get('max_memory_mb', 4096)
        self.auto_unload_enabled = self.config.get('memory_management', {}).get('auto_unload', True)
        
        # 统计信息
        self.stats = {
            'total_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'auto_unloads': 0,
            'manual_unloads': 0
        }
        
        # 注册默认加载器
        self._register_default_loaders()
        
        # 自动发现和注册模型
        self._discover_models()
        
        logger.info("模型加载器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('model_loader', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_discovery': {
                'auto_discover': True,
                'search_paths': ['src/models', 'models'],
                'supported_types': ['qwen', 'transformer', 'sklearn', 'custom']
            },
            'loading_strategy': {
                'lazy_loading': True,
                'preload_critical': True,
                'cache_models': True,
                'max_concurrent_loads': 2
            },
            'memory_management': {
                'max_memory_mb': 4096,
                'auto_unload': True,
                'unload_timeout_minutes': 30,
                'gc_after_unload': True
            },
            'performance': {
                'enable_model_sharing': True,
                'batch_optimization': True,
                'prefetch_next_model': False
            }
        }
    
    def _register_default_loaders(self):
        """注册默认的模型加载器"""
        
        def load_qwen_model(model_info: ModelInfo) -> Any:
            """加载Qwen模型"""
            try:
                from ..models.qwen_wrapper import QwenWrapper
                
                # 从配置创建实例
                model_config = model_info.config.copy()
                model_config['model_path'] = model_info.model_path
                
                wrapper = QwenWrapper(model_config)
                success = wrapper.load_model(enable_lora=model_config.get('enable_lora', False))
                
                if success:
                    return wrapper
                else:
                    raise Exception("Qwen模型加载失败")
                    
            except Exception as e:
                raise Exception(f"Qwen模型加载异常: {e}")
        
        def load_sklearn_model(model_info: ModelInfo) -> Any:
            """加载sklearn模型"""
            try:
                import pickle
                
                model_path = Path(model_info.model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                return model
                
            except Exception as e:
                raise Exception(f"sklearn模型加载异常: {e}")
        
        def load_anomaly_detector(model_info: ModelInfo) -> Any:
            """加载异常检测器"""
            try:
                detector_type = model_info.config.get('detector_type', 'ensemble')
                
                if detector_type == 'statistical':
                    from ..anomaly_detection.statistical_detector import get_statistical_detector
                    return get_statistical_detector()
                elif detector_type == 'behavioral':
                    from ..anomaly_detection.behavioral_detector import get_behavioral_detector
                    return get_behavioral_detector()
                elif detector_type == 'market':
                    from ..anomaly_detection.market_detector import get_market_detector
                    return get_market_detector()
                elif detector_type == 'semantic':
                    from ..anomaly_detection.semantic_detector import get_semantic_detector
                    return get_semantic_detector()
                elif detector_type == 'ensemble':
                    from ..anomaly_detection.ensemble_detector import get_ensemble_detector
                    return get_ensemble_detector()
                else:
                    raise ValueError(f"不支持的检测器类型: {detector_type}")
                    
            except Exception as e:
                raise Exception(f"异常检测器加载异常: {e}")
        
        # 注册加载器
        self.loaders['qwen'] = load_qwen_model
        self.loaders['sklearn'] = load_sklearn_model
        self.loaders['anomaly_detector'] = load_anomaly_detector
        self.loaders['transformer'] = load_qwen_model  # 复用Qwen加载器
    
    def _discover_models(self):
        """自动发现和注册模型"""
        if not self.config.get('model_discovery', {}).get('auto_discover', True):
            return
        
        # 从配置文件中加载预定义模型
        try:
            model_config = load_config("configs/model_config.yaml")
            predefined_models = model_config.get('models', {})
            
            for model_name, model_config in predefined_models.items():
                self.register_model(
                    name=model_name,
                    model_type=model_config.get('type', 'unknown'),
                    model_path=model_config.get('path', ''),
                    config=model_config
                )
                
        except Exception as e:
            logger.debug(f"预定义模型加载失败: {e}")
        
        # 注册内置的异常检测器
        builtin_detectors = [
            ('statistical_detector', 'anomaly_detector', '', {'detector_type': 'statistical'}),
            ('behavioral_detector', 'anomaly_detector', '', {'detector_type': 'behavioral'}),
            ('market_detector', 'anomaly_detector', '', {'detector_type': 'market'}),
            ('semantic_detector', 'anomaly_detector', '', {'detector_type': 'semantic'}),
            ('ensemble_detector', 'anomaly_detector', '', {'detector_type': 'ensemble'})
        ]
        
        for name, model_type, path, config in builtin_detectors:
            self.register_model(name, model_type, path, config)
    
    def register_model(self, name: str, model_type: str, model_path: str, 
                      config: Dict[str, Any]) -> bool:
        """
        注册模型
        
        Args:
            name: 模型名称
            model_type: 模型类型
            model_path: 模型路径
            config: 模型配置
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self.loading_lock:
                if name in self.models:
                    logger.warning(f"模型 {name} 已存在，将被覆盖")
                
                model_info = ModelInfo(
                    name=name,
                    model_type=model_type,
                    model_path=model_path,
                    config=config
                )
                
                self.models[name] = model_info
                logger.info(f"模型 {name} 注册成功 (类型: {model_type})")
                
                return True
                
        except Exception as e:
            logger.error(f"模型 {name} 注册失败: {e}")
            return False
    
    def register_loader(self, model_type: str, loader_func: Callable) -> bool:
        """
        注册模型加载器
        
        Args:
            model_type: 模型类型
            loader_func: 加载函数
            
        Returns:
            bool: 注册是否成功
        """
        try:
            self.loaders[model_type] = loader_func
            logger.info(f"加载器 {model_type} 注册成功")
            return True
        except Exception as e:
            logger.error(f"加载器 {model_type} 注册失败: {e}")
            return False
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            force_reload: 是否强制重新加载
            
        Returns:
            Any: 模型实例
        """
        with self.loading_lock:
            if model_name not in self.models:
                raise ValueError(f"未注册的模型: {model_name}")
            
            model_info = self.models[model_name]
            
            # 检查是否已加载且不需要重新加载
            if not force_reload and model_info.status == ModelStatus.LOADED and model_info.instance is not None:
                self._update_model_usage(model_name)
                self.stats['cache_hits'] += 1
                return model_info.instance
            
            # 检查内存限制
            if not self._check_memory_availability():
                self._auto_unload_models()
            
            try:
                model_info.status = ModelStatus.LOADING
                self.stats['total_loads'] += 1
                
                start_time = time.time()
                
                # 获取对应的加载器
                if model_info.model_type not in self.loaders:
                    raise ValueError(f"不支持的模型类型: {model_info.model_type}")
                
                loader_func = self.loaders[model_info.model_type]
                
                # 加载模型
                logger.info(f"开始加载模型: {model_name}")
                model_instance = loader_func(model_info)
                
                # 更新模型信息
                load_time = time.time() - start_time
                model_info.instance = model_instance
                model_info.load_time = load_time
                model_info.status = ModelStatus.LOADED
                model_info.memory_usage = self._estimate_model_memory(model_instance)
                
                self._update_model_usage(model_name)
                self.stats['successful_loads'] += 1
                
                logger.info(f"模型 {model_name} 加载成功 (耗时: {load_time:.2f}s, 内存: {model_info.memory_usage}MB)")
                
                return model_instance
                
            except Exception as e:
                model_info.status = ModelStatus.ERROR
                self.stats['failed_loads'] += 1
                
                logger.error(f"模型 {model_name} 加载失败: {e}")
                raise Exception(f"模型加载失败: {e}")
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 卸载是否成功
        """
        with self.loading_lock:
            if model_name not in self.models:
                logger.warning(f"模型 {model_name} 不存在")
                return False
            
            model_info = self.models[model_name]
            
            if model_info.status != ModelStatus.LOADED:
                logger.warning(f"模型 {model_name} 未加载")
                return False
            
            try:
                model_info.status = ModelStatus.UNLOADING
                
                # 清理模型实例
                if hasattr(model_info.instance, 'cleanup'):
                    model_info.instance.cleanup()
                
                model_info.instance = None
                model_info.status = ModelStatus.UNLOADED
                model_info.memory_usage = 0
                
                # 强制垃圾回收
                if self.config.get('memory_management', {}).get('gc_after_unload', True):
                    gc.collect()
                
                self.stats['manual_unloads'] += 1
                logger.info(f"模型 {model_name} 卸载成功")
                
                return True
                
            except Exception as e:
                model_info.status = ModelStatus.ERROR
                logger.error(f"模型 {model_name} 卸载失败: {e}")
                return False
    
    def get_model(self, model_name: str, auto_load: bool = True) -> Optional[Any]:
        """
        获取模型实例
        
        Args:
            model_name: 模型名称
            auto_load: 是否自动加载
            
        Returns:
            Optional[Any]: 模型实例
        """
        with self.access_lock:
            if model_name not in self.models:
                if auto_load:
                    logger.warning(f"模型 {model_name} 未注册")
                return None
            
            model_info = self.models[model_name]
            
            if model_info.status == ModelStatus.LOADED and model_info.instance is not None:
                self._update_model_usage(model_name)
                return model_info.instance
            elif auto_load:
                try:
                    return self.load_model(model_name)
                except Exception as e:
                    logger.error(f"自动加载模型 {model_name} 失败: {e}")
                    return None
            
            return None
    
    def _update_model_usage(self, model_name: str):
        """更新模型使用统计"""
        if model_name in self.models:
            model_info = self.models[model_name]
            model_info.last_used = datetime.now()
            model_info.use_count += 1
    
    def _check_memory_availability(self) -> bool:
        """检查内存可用性"""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            return available_mb > self.memory_threshold * 0.1  # 至少保留10%的阈值
            
        except Exception:
            return True  # 如果无法检查，假设可用
    
    def _estimate_model_memory(self, model_instance: Any) -> int:
        """估算模型内存使用"""
        try:
            if hasattr(model_instance, 'get_memory_footprint'):
                return model_instance.get_memory_footprint()
            
            # 简单的估算方法
            import sys
            return sys.getsizeof(model_instance) // (1024 * 1024)
            
        except Exception:
            return 100  # 默认估算值
    
    def _auto_unload_models(self):
        """自动卸载长时间未使用的模型"""
        if not self.auto_unload_enabled:
            return
        
        current_time = datetime.now()
        timeout_minutes = self.config.get('memory_management', {}).get('unload_timeout_minutes', 30)
        timeout_delta = timedelta(minutes=timeout_minutes)
        
        models_to_unload = []
        
        for model_name, model_info in self.models.items():
            if (model_info.status == ModelStatus.LOADED and 
                model_info.last_used and 
                current_time - model_info.last_used > timeout_delta):
                models_to_unload.append(model_name)
        
        for model_name in models_to_unload:
            if self.unload_model(model_name):
                self.stats['auto_unloads'] += 1
                logger.info(f"自动卸载模型: {model_name}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出所有注册的模型
        
        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """
        models_info = []
        
        for model_name, model_info in self.models.items():
            info = {
                'name': model_name,
                'type': model_info.model_type,
                'status': model_info.status.value,
                'memory_usage_mb': model_info.memory_usage,
                'load_time_seconds': model_info.load_time,
                'use_count': model_info.use_count,
                'last_used': model_info.last_used.isoformat() if model_info.last_used else None
            }
            models_info.append(info)
        
        return models_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取加载器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_memory = sum(info.memory_usage for info in self.models.values())
        loaded_count = sum(1 for info in self.models.values() if info.status == ModelStatus.LOADED)
        
        return {
            'total_models': len(self.models),
            'loaded_models': loaded_count,
            'total_memory_usage_mb': total_memory,
            'loading_statistics': self.stats.copy(),
            'success_rate': self.stats['successful_loads'] / max(self.stats['total_loads'], 1),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_loads'], 1)
        }
    
    def cleanup(self):
        """清理所有模型和资源"""
        logger.info("开始清理模型加载器...")
        
        model_names = list(self.models.keys())
        for model_name in model_names:
            try:
                self.unload_model(model_name)
            except Exception as e:
                logger.error(f"清理模型 {model_name} 时出错: {e}")
        
        gc.collect()
        logger.info("模型加载器清理完成")


# 全局模型加载器实例
_global_model_loader = None


def get_model_loader() -> ModelLoader:
    """
    获取全局模型加载器实例
    
    Returns:
        ModelLoader: 加载器实例
    """
    global _global_model_loader
    
    if _global_model_loader is None:
        _global_model_loader = ModelLoader()
    
    return _global_model_loader


if __name__ == "__main__":
    # 使用示例
    loader = ModelLoader()
    
    print("模型加载器测试:")
    
    # 列出注册的模型
    models = loader.list_models()
    print(f"注册的模型数量: {len(models)}")
    
    for model_info in models:
        print(f"  - {model_info['name']} ({model_info['type']}) - {model_info['status']}")
    
    # 测试加载内置检测器
    try:
        detector = loader.get_model('ensemble_detector')
        if detector:
            print("✅ 集成检测器加载成功")
        else:
            print("❌ 集成检测器加载失败")
    except Exception as e:
        print(f"❌ 检测器加载异常: {e}")
    
    # 获取统计信息
    stats = loader.get_statistics()
    print(f"\n统计信息: {stats}")
    
    # 清理
    loader.cleanup() 
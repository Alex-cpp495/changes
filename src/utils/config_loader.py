"""
配置加载器模块

提供统一的配置文件加载和管理接口，支持多种配置文件格式
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

try:
    from .logger import get_logger
except ImportError:
    # 当直接运行时的备用导入
    from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    配置加载器类
    
    提供配置文件的加载、缓存和验证功能
    
    Attributes:
        _cache: 配置文件缓存
        _base_path: 配置文件基础路径
    """
    
    def __init__(self, base_path: str = "configs"):
        """
        初始化配置加载器
        
        Args:
            base_path: 配置文件基础路径
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._base_path = Path(base_path)
        
        if not self._base_path.exists():
            logger.warning(f"配置目录不存在: {self._base_path}")
            
    def load_config(self, config_file: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件名或路径
            use_cache: 是否使用缓存
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        # 如果使用缓存且配置已缓存，直接返回
        if use_cache and config_file in self._cache:
            logger.debug(f"从缓存加载配置: {config_file}")
            return self._cache[config_file]
        
        # 构建完整的配置文件路径
        if not config_file.endswith('.yaml') and not config_file.endswith('.yml'):
            config_file += '.yaml'
            
        config_path = self._base_path / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            logger.info(f"加载配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # 缓存配置
            if use_cache:
                self._cache[config_file] = config
                
            logger.info(f"配置文件加载成功: {config_file}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML格式错误: {config_file}, 错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"配置文件加载失败: {config_file}, 错误: {str(e)}")
            raise
    
    def get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config: 配置字典
            key_path: 键路径，用点分隔，如 'model.lora_config.rank'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"配置键不存在: {key_path}, 使用默认值: {default}")
            return default
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """
        验证配置文件是否包含必需的键
        
        Args:
            config: 配置字典
            required_keys: 必需的键列表
            
        Returns:
            是否验证通过
        """
        missing_keys = []
        
        for key in required_keys:
            if self.get_nested_value(config, key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"配置验证失败，缺少必需的键: {missing_keys}")
            return False
        
        logger.info("配置验证通过")
        return True
    
    def clear_cache(self):
        """清空配置缓存"""
        self._cache.clear()
        logger.info("配置缓存已清空")
    
    def reload_config(self, config_file: str) -> Dict[str, Any]:
        """
        重新加载配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            配置字典
        """
        # 从缓存中移除
        if config_file in self._cache:
            del self._cache[config_file]
        
        return self.load_config(config_file)


# 全局配置加载器实例
_global_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """
    获取全局配置加载器实例
    
    Returns:
        配置加载器实例
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader


def load_config(config_file: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件
    
    Args:
        config_file: 配置文件名
        use_cache: 是否使用缓存
        
    Returns:
        配置字典
    """
    loader = get_config_loader()
    return loader.load_config(config_file, use_cache)


def get_config_value(config_file: str, key_path: str, default: Any = None) -> Any:
    """
    便捷函数：获取配置值
    
    Args:
        config_file: 配置文件名
        key_path: 键路径
        default: 默认值
        
    Returns:
        配置值
    """
    loader = get_config_loader()
    config = loader.load_config(config_file)
    return loader.get_nested_value(config, key_path, default)


# 预定义的配置文件路径常量
MODEL_CONFIG = "model_config"
TRAINING_CONFIG = "training_config"
ANOMALY_THRESHOLDS = "anomaly_thresholds"
WEB_CONFIG = "web_config" 
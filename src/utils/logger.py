"""
日志工具模块

提供统一的日志配置和获取接口，支持文件日志和控制台输出
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog

# 创建日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 日志格式
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
CONSOLE_LOG_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"

# 颜色配置
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    设置日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件名（不含路径）
        console: 是否输出到控制台

    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    if log_file:
        # 使用更完整的日志目录路径
        full_log_dir = Path.cwd() / LOG_DIR
        full_log_dir.mkdir(exist_ok=True)
        file_path = full_log_dir / f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(FILE_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 控制台处理器
    if console:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = colorlog.ColoredFormatter(
            CONSOLE_LOG_FORMAT,
            log_colors=LOG_COLORS
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 防止日志重复传播
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器（使用默认配置）

    Args:
        name: 日志器名称，通常使用 __name__

    Returns:
        配置好的日志器
    """
    # 从模块名提取简短名称
    short_name = name.split('.')[-1] if '.' in name else name

    return setup_logger(
        name=name,
        level=logging.INFO,
        log_file=short_name,
        console=True
    )


# 全局异常日志记录
def log_exception(logger: logging.Logger):
    """
    装饰器：自动记录函数异常

    Args:
        logger: 日志器实例
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


# 性能日志记录
def log_performance(logger: logging.Logger):
    """
    装饰器：记录函数执行时间

    Args:
        logger: 日志器实例
    """
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"函数 {func.__name__} 执行耗时: {elapsed_time:.4f}秒")
            return result
        return wrapper
    return decorator

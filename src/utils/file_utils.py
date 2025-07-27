"""
文件操作工具模块

提供统一的文件操作接口，包括读写、目录管理、批量处理等功能
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
import hashlib
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """
    文件管理器类
    
    提供文件和目录的统一管理接口
    
    Attributes:
        base_path: 基础路径
    """
    
    def __init__(self, base_path: str = "."):
        """
        初始化文件管理器
        
        Args:
            base_path: 基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def read_text_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        读取文本文件
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 编码错误
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        try:
            logger.debug(f"读取文件: {full_path}")
            with open(full_path, 'r', encoding=encoding) as file:
                content = file.read()
            logger.debug(f"文件读取成功，内容长度: {len(content)}")
            return content
            
        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误: {full_path}, 错误: {str(e)}")
            # 尝试其他编码
            for alt_encoding in ['gbk', 'gb2312', 'latin-1']:
                try:
                    with open(full_path, 'r', encoding=alt_encoding) as file:
                        content = file.read()
                    logger.warning(f"使用 {alt_encoding} 编码成功读取文件")
                    return content
                except UnicodeDecodeError:
                    continue
            raise
        except Exception as e:
            logger.error(f"文件读取失败: {full_path}, 错误: {str(e)}")
            raise
    
    def write_text_file(self, file_path: Union[str, Path], content: str, 
                       encoding: str = 'utf-8', backup: bool = False) -> bool:
        """
        写入文本文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码
            backup: 是否备份原文件
            
        Returns:
            是否写入成功
        """
        full_path = self.base_path / file_path
        
        try:
            # 创建目录
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 备份原文件
            if backup and full_path.exists():
                backup_path = full_path.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copy2(full_path, backup_path)
                logger.info(f"文件已备份: {backup_path}")
            
            logger.debug(f"写入文件: {full_path}")
            with open(full_path, 'w', encoding=encoding) as file:
                file.write(content)
            
            logger.debug(f"文件写入成功，内容长度: {len(content)}")
            return True
            
        except Exception as e:
            logger.error(f"文件写入失败: {full_path}, 错误: {str(e)}")
            return False
    
    def read_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        读取JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            JSON数据
        """
        content = self.read_text_file(file_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {file_path}, 错误: {str(e)}")
            raise
    
    def write_json_file(self, file_path: Union[str, Path], data: Dict[str, Any], 
                       indent: int = 2, ensure_ascii: bool = False) -> bool:
        """
        写入JSON文件
        
        Args:
            file_path: 文件路径
            data: JSON数据
            indent: 缩进
            ensure_ascii: 是否确保ASCII编码
            
        Returns:
            是否写入成功
        """
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
            return self.write_text_file(file_path, content)
        except Exception as e:
            logger.error(f"JSON序列化失败: {str(e)}")
            return False
    
    def save_pickle(self, file_path: Union[str, Path], data: Any) -> bool:
        """
        保存Pickle文件
        
        Args:
            file_path: 文件路径
            data: 要保存的数据
            
        Returns:
            是否保存成功
        """
        full_path = self.base_path / file_path
        
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"保存Pickle文件: {full_path}")
            with open(full_path, 'wb') as file:
                pickle.dump(data, file)
            
            logger.debug("Pickle文件保存成功")
            return True
            
        except Exception as e:
            logger.error(f"Pickle保存失败: {full_path}, 错误: {str(e)}")
            return False
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        加载Pickle文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的数据
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Pickle文件不存在: {full_path}")
        
        try:
            logger.debug(f"加载Pickle文件: {full_path}")
            with open(full_path, 'rb') as file:
                data = pickle.load(file)
            logger.debug("Pickle文件加载成功")
            return data
            
        except Exception as e:
            logger.error(f"Pickle加载失败: {full_path}, 错误: {str(e)}")
            raise
    
    def list_files(self, directory: Union[str, Path] = ".", 
                   pattern: str = "*", recursive: bool = False) -> List[Path]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件模式
            recursive: 是否递归搜索
            
        Returns:
            文件路径列表
        """
        dir_path = self.base_path / directory
        
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return []
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        # 只返回文件，不包括目录
        files = [f for f in files if f.is_file()]
        
        logger.debug(f"找到 {len(files)} 个文件，模式: {pattern}")
        return files
    
    def get_file_hash(self, file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法
            
        Returns:
            文件哈希值
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(full_path, 'rb') as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_func.update(chunk)
            
            hash_value = hash_func.hexdigest()
            logger.debug(f"文件哈希 ({algorithm}): {hash_value}")
            return hash_value
            
        except Exception as e:
            logger.error(f"计算文件哈希失败: {full_path}, 错误: {str(e)}")
            raise
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        stat = full_path.stat()
        
        return {
            'path': str(full_path),
            'name': full_path.name,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': full_path.is_file(),
            'is_dir': full_path.is_dir(),
            'extension': full_path.suffix
        }
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            目录路径对象
        """
        dir_path = self.base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"目录已确保存在: {dir_path}")
        return dir_path
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """
        复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            
        Returns:
            是否复制成功
        """
        src_path = self.base_path / src
        dst_path = self.base_path / dst
        
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            logger.info(f"文件复制成功: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件复制失败: {src_path} -> {dst_path}, 错误: {str(e)}")
            return False
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """
        移动文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            
        Returns:
            是否移动成功
        """
        src_path = self.base_path / src
        dst_path = self.base_path / dst
        
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            logger.info(f"文件移动成功: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件移动失败: {src_path} -> {dst_path}, 错误: {str(e)}")
            return False


def batch_process_files(file_list: List[Path], 
                       processor_func, 
                       output_dir: Optional[Path] = None,
                       max_workers: int = 4) -> List[Tuple[Path, bool]]:
    """
    批量处理文件
    
    Args:
        file_list: 文件列表
        processor_func: 处理函数
        output_dir: 输出目录
        max_workers: 最大工作线程数
        
    Returns:
        处理结果列表 (文件路径, 是否成功)
    """
    import concurrent.futures
    
    logger.info(f"开始批量处理 {len(file_list)} 个文件")
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_file = {
            executor.submit(processor_func, file_path, output_dir): file_path 
            for file_path in file_list
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append((file_path, True))
                logger.debug(f"文件处理成功: {file_path}")
            except Exception as e:
                results.append((file_path, False))
                logger.error(f"文件处理失败: {file_path}, 错误: {str(e)}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"批量处理完成: {success_count}/{len(file_list)} 成功")
    
    return results


# 全局文件管理器实例
_global_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """
    获取全局文件管理器实例
    
    Returns:
        文件管理器实例
    """
    global _global_file_manager
    if _global_file_manager is None:
        _global_file_manager = FileManager()
    return _global_file_manager 
"""
批量处理器
高效处理大量研报数据的异常检测、特征提取、质量检查等任务
"""

import asyncio
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union, Callable, Iterator, Tuple
from datetime import datetime, timedelta
import logging
import time
import traceback
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager
from .model_loader import get_model_loader

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """处理模式枚举"""
    SEQUENTIAL = "sequential"      # 顺序处理
    THREADING = "threading"       # 多线程处理
    MULTIPROCESSING = "multiprocessing"  # 多进程处理
    ASYNC = "async"              # 异步处理


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchTask:
    """批处理任务数据类"""
    task_id: str
    data: Any
    processor_type: str
    config: Dict[str, Any]
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BatchResult:
    """批处理结果数据类"""
    batch_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    processing_time: float
    results: List[Any]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BatchProcessor:
    """
    批量处理器
    
    提供高效的批量数据处理功能：
    1. 多种处理模式 - 顺序、多线程、多进程、异步
    2. 任务管理 - 优先级队列、状态跟踪、重试机制
    3. 负载均衡 - 智能调度、资源监控、动态调整
    4. 错误处理 - 容错机制、错误重试、部分失败处理
    5. 进度监控 - 实时进度、性能统计、日志记录
    6. 结果管理 - 结果收集、格式化、持久化
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 批处理配置
        model_loader: 模型加载器
        task_queue: 任务队列
        result_cache: 结果缓存
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化批量处理器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.model_loader = get_model_loader()
        self.file_manager = get_file_manager()
        
        # 任务队列
        self.task_queue = queue.PriorityQueue()
        self.result_cache: Dict[str, Any] = {}
        
        # 处理器注册表
        self.processors: Dict[str, Callable] = {}
        
        # 线程池和进程池
        self.thread_pool = None
        self.process_pool = None
        
        # 状态监控
        self.active_tasks: Dict[str, BatchTask] = {}
        self.completed_tasks: Dict[str, BatchTask] = {}
        
        # 统计信息
        self.stats = {
            'total_batches': 0,
            'completed_batches': 0,
            'failed_batches': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0
        }
        
        # 注册默认处理器
        self._register_default_processors()
        
        logger.info("批量处理器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('batch_processing', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'processing': {
                'default_mode': 'threading',
                'max_workers': min(32, (mp.cpu_count() or 1) + 4),
                'max_queue_size': 10000,
                'chunk_size': 100,
                'timeout_seconds': 300
            },
            'retry': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'backoff_factor': 2.0
            },
            'memory': {
                'max_memory_mb': 8192,
                'gc_interval': 100,
                'clear_cache_interval': 1000
            },
            'monitoring': {
                'log_progress_interval': 50,
                'save_intermediate_results': True,
                'collect_detailed_stats': True
            },
            'output': {
                'save_results': True,
                'output_format': 'json',
                'compress_results': False
            }
        }
    
    def _register_default_processors(self):
        """注册默认的数据处理器"""
        
        def anomaly_detection_processor(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
            """异常检测处理器"""
            try:
                # 获取集成检测器
                detector = self.model_loader.get_model('ensemble_detector')
                if not detector:
                    raise Exception("集成检测器加载失败")
                
                # 执行异常检测
                result = detector.detect_anomalies(data)
                
                return {
                    'success': True,
                    'result': result,
                    'processor': 'anomaly_detection',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processor': 'anomaly_detection',
                    'timestamp': datetime.now().isoformat()
                }
        
        def quality_check_processor(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
            """质量检查处理器"""
            try:
                from ..preprocessing.quality_checker import get_quality_checker
                
                checker = get_quality_checker()
                result = checker.check_data_quality(data)
                
                return {
                    'success': True,
                    'result': result,
                    'processor': 'quality_check',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processor': 'quality_check',
                    'timestamp': datetime.now().isoformat()
                }
        
        def feature_extraction_processor(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
            """特征提取处理器"""
            try:
                from ..preprocessing.feature_extractor import get_feature_extractor
                
                extractor = get_feature_extractor()
                result = extractor.extract_features(data)
                
                return {
                    'success': True,
                    'result': result,
                    'processor': 'feature_extraction',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processor': 'feature_extraction',
                    'timestamp': datetime.now().isoformat()
                }
        
        def text_cleaning_processor(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
            """文本清洗处理器"""
            try:
                from ..preprocessing.text_cleaner import get_text_cleaner
                
                cleaner = get_text_cleaner()
                content = data.get('content', '')
                result = cleaner.clean_text(content)
                
                # 更新数据中的内容
                processed_data = data.copy()
                processed_data['content'] = result['cleaned_text']
                processed_data['cleaning_metadata'] = {
                    'quality_score': result['quality_score'],
                    'cleaning_applied': result['cleaning_applied'],
                    'issues_found': result['issues_found']
                }
                
                return {
                    'success': True,
                    'result': processed_data,
                    'processor': 'text_cleaning',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processor': 'text_cleaning',
                    'timestamp': datetime.now().isoformat()
                }
        
        # 注册处理器
        self.processors['anomaly_detection'] = anomaly_detection_processor
        self.processors['quality_check'] = quality_check_processor
        self.processors['feature_extraction'] = feature_extraction_processor
        self.processors['text_cleaning'] = text_cleaning_processor
    
    def register_processor(self, name: str, processor_func: Callable) -> bool:
        """
        注册自定义处理器
        
        Args:
            name: 处理器名称
            processor_func: 处理函数
            
        Returns:
            bool: 注册是否成功
        """
        try:
            self.processors[name] = processor_func
            logger.info(f"处理器 {name} 注册成功")
            return True
        except Exception as e:
            logger.error(f"处理器 {name} 注册失败: {e}")
            return False
    
    def process_batch(self, data_list: List[Dict[str, Any]], 
                     processor_type: str,
                     mode: ProcessingMode = ProcessingMode.THREADING,
                     config: Optional[Dict[str, Any]] = None) -> BatchResult:
        """
        批量处理数据
        
        Args:
            data_list: 数据列表
            processor_type: 处理器类型
            mode: 处理模式
            config: 处理配置
            
        Returns:
            BatchResult: 批处理结果
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data_list)}"
        start_time = time.time()
        
        logger.info(f"开始批量处理: {batch_id} (数据量: {len(data_list)}, 模式: {mode.value})")
        
        try:
            self.stats['total_batches'] += 1
            self.stats['total_tasks'] += len(data_list)
            
            # 检查处理器是否存在
            if processor_type not in self.processors:
                raise ValueError(f"未注册的处理器: {processor_type}")
            
            # 创建任务列表
            tasks = []
            for i, data in enumerate(data_list):
                task = BatchTask(
                    task_id=f"{batch_id}_task_{i}",
                    data=data,
                    processor_type=processor_type,
                    config=config or {}
                )
                tasks.append(task)
            
            # 根据模式选择处理方法
            if mode == ProcessingMode.SEQUENTIAL:
                results = self._process_sequential(tasks)
            elif mode == ProcessingMode.THREADING:
                results = self._process_threading(tasks)
            elif mode == ProcessingMode.MULTIPROCESSING:
                results = self._process_multiprocessing(tasks)
            elif mode == ProcessingMode.ASYNC:
                results = self._process_async(tasks)
            else:
                raise ValueError(f"不支持的处理模式: {mode}")
            
            # 统计结果
            processing_time = time.time() - start_time
            completed_tasks = sum(1 for r in results if r.get('success', False))
            failed_tasks = len(results) - completed_tasks
            
            # 收集错误信息
            errors = []
            for i, result in enumerate(results):
                if not result.get('success', False):
                    errors.append({
                        'task_id': tasks[i].task_id,
                        'error': result.get('error', 'Unknown error'),
                        'data_index': i
                    })
            
            # 创建批处理结果
            batch_result = BatchResult(
                batch_id=batch_id,
                total_tasks=len(data_list),
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                processing_time=processing_time,
                results=results,
                errors=errors,
                metadata={
                    'processor_type': processor_type,
                    'processing_mode': mode.value,
                    'config': config,
                    'start_time': start_time,
                    'end_time': time.time(),
                    'avg_task_time': processing_time / len(data_list) if data_list else 0
                }
            )
            
            # 更新统计信息
            self.stats['completed_batches'] += 1
            self.stats['completed_tasks'] += completed_tasks
            self.stats['failed_tasks'] += failed_tasks
            self.stats['total_processing_time'] += processing_time
            self.stats['average_task_time'] = self.stats['total_processing_time'] / max(self.stats['total_tasks'], 1)
            
            logger.info(f"批量处理完成: {batch_id} (成功: {completed_tasks}/{len(data_list)}, 耗时: {processing_time:.2f}s)")
            
            # 保存结果
            if self.config.get('output', {}).get('save_results', True):
                self._save_batch_result(batch_result)
            
            return batch_result
            
        except Exception as e:
            self.stats['failed_batches'] += 1
            logger.error(f"批量处理失败: {batch_id} - {e}")
            
            # 返回失败结果
            return BatchResult(
                batch_id=batch_id,
                total_tasks=len(data_list),
                completed_tasks=0,
                failed_tasks=len(data_list),
                processing_time=time.time() - start_time,
                results=[],
                errors=[{'batch_error': str(e)}],
                metadata={'error': str(e)}
            )
    
    def _process_sequential(self, tasks: List[BatchTask]) -> List[Dict[str, Any]]:
        """顺序处理任务"""
        results = []
        
        for i, task in enumerate(tasks):
            try:
                result = self._process_single_task(task)
                results.append(result)
                
                if (i + 1) % self.config.get('monitoring', {}).get('log_progress_interval', 50) == 0:
                    logger.info(f"顺序处理进度: {i + 1}/{len(tasks)}")
                    
            except Exception as e:
                logger.error(f"任务 {task.task_id} 处理失败: {e}")
                results.append({'success': False, 'error': str(e)})
        
        return results
    
    def _process_threading(self, tasks: List[BatchTask]) -> List[Dict[str, Any]]:
        """多线程处理任务"""
        max_workers = self.config.get('processing', {}).get('max_workers', 4)
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._process_single_task, task): i 
                for i, task in enumerate(tasks)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=self.config.get('processing', {}).get('timeout_seconds', 300))
                    results[index] = result
                except Exception as e:
                    logger.error(f"任务 {tasks[index].task_id} 处理失败: {e}")
                    results[index] = {'success': False, 'error': str(e)}
                
                completed += 1
                if completed % self.config.get('monitoring', {}).get('log_progress_interval', 50) == 0:
                    logger.info(f"多线程处理进度: {completed}/{len(tasks)}")
        
        return results
    
    def _process_multiprocessing(self, tasks: List[BatchTask]) -> List[Dict[str, Any]]:
        """多进程处理任务"""
        max_workers = min(self.config.get('processing', {}).get('max_workers', 4), mp.cpu_count())
        
        # 准备任务数据（序列化友好）
        task_data = []
        for task in tasks:
            task_data.append({
                'task_id': task.task_id,
                'data': task.data,
                'processor_type': task.processor_type,
                'config': task.config
            })
        
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 分块处理以避免内存问题
            chunk_size = self.config.get('processing', {}).get('chunk_size', 100)
            chunks = [task_data[i:i + chunk_size] for i in range(0, len(task_data), chunk_size)]
            
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    logger.info(f"多进程处理进度: {len(results)}/{len(tasks)}")
                    
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"块处理失败: {e}")
                    # 为失败的块添加错误结果
                    for _ in chunk:
                        results.append({'success': False, 'error': str(e)})
        
        return results
    
    def _process_async(self, tasks: List[BatchTask]) -> List[Dict[str, Any]]:
        """异步处理任务"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._async_process_tasks(tasks))
    
    async def _async_process_tasks(self, tasks: List[BatchTask]) -> List[Dict[str, Any]]:
        """异步处理任务的协程"""
        semaphore = asyncio.Semaphore(self.config.get('processing', {}).get('max_workers', 10))
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await self._async_process_single_task(task)
        
        # 创建所有任务的协程
        coroutines = [process_with_semaphore(task) for task in tasks]
        
        # 并发执行所有任务
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"异步任务 {tasks[i].task_id} 失败: {result}")
                processed_results.append({'success': False, 'error': str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _async_process_single_task(self, task: BatchTask) -> Dict[str, Any]:
        """异步处理单个任务"""
        loop = asyncio.get_event_loop()
        # 在线程池中运行同步处理函数
        return await loop.run_in_executor(None, self._process_single_task, task)
    
    def _process_single_task(self, task: BatchTask) -> Dict[str, Any]:
        """处理单个任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            processor_func = self.processors[task.processor_type]
            result = processor_func(task.data, task.config)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"任务 {task.task_id} 处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _process_chunk(chunk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理数据块（用于多进程）"""
        results = []
        
        # 在新进程中重新创建处理器实例
        try:
            from ..anomaly_detection.ensemble_detector import get_ensemble_detector
            from ..preprocessing.quality_checker import get_quality_checker
            from ..preprocessing.feature_extractor import get_feature_extractor
            from ..preprocessing.text_cleaner import get_text_cleaner
            
            processors = {
                'anomaly_detection': lambda data, config: {
                    'success': True,
                    'result': get_ensemble_detector().detect_anomalies(data),
                    'processor': 'anomaly_detection',
                    'timestamp': datetime.now().isoformat()
                },
                'quality_check': lambda data, config: {
                    'success': True,
                    'result': get_quality_checker().check_data_quality(data),
                    'processor': 'quality_check',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            for task_data in chunk_data:
                try:
                    processor_type = task_data['processor_type']
                    if processor_type in processors:
                        result = processors[processor_type](task_data['data'], task_data['config'])
                    else:
                        result = {'success': False, 'error': f'Unknown processor: {processor_type}'}
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
            
        except Exception as e:
            # 如果整个块处理失败，为所有任务返回错误
            for _ in chunk_data:
                results.append({'success': False, 'error': f'Chunk processing failed: {str(e)}'})
        
        return results
    
    def _save_batch_result(self, batch_result: BatchResult):
        """保存批处理结果"""
        try:
            output_dir = Path('data/batch_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / f"{batch_result.batch_id}.json"
            
            # 准备保存的数据
            save_data = {
                'batch_id': batch_result.batch_id,
                'summary': {
                    'total_tasks': batch_result.total_tasks,
                    'completed_tasks': batch_result.completed_tasks,
                    'failed_tasks': batch_result.failed_tasks,
                    'processing_time': batch_result.processing_time,
                    'success_rate': batch_result.completed_tasks / batch_result.total_tasks if batch_result.total_tasks > 0 else 0
                },
                'metadata': batch_result.metadata,
                'errors': batch_result.errors,
                'results': batch_result.results if len(batch_result.results) < 1000 else f"Results too large, saved separately"
            }
            
            # 保存主结果文件
            self.file_manager.write_json_file(result_file, save_data)
            
            # 如果结果太大，单独保存
            if len(batch_result.results) >= 1000:
                results_file = output_dir / f"{batch_result.batch_id}_results.json"
                self.file_manager.write_json_file(results_file, batch_result.results)
            
            logger.info(f"批处理结果已保存: {result_file}")
            
        except Exception as e:
            logger.error(f"保存批处理结果失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取批处理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'batch_statistics': self.stats.copy(),
            'processor_count': len(self.processors),
            'registered_processors': list(self.processors.keys()),
            'active_tasks': len(self.active_tasks),
            'completed_tasks_count': len(self.completed_tasks),
            'config': self.config
        }


# 全局批量处理器实例
_global_batch_processor = None


def get_batch_processor() -> BatchProcessor:
    """
    获取全局批量处理器实例
    
    Returns:
        BatchProcessor: 处理器实例
    """
    global _global_batch_processor
    
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    
    return _global_batch_processor


if __name__ == "__main__":
    # 使用示例
    processor = BatchProcessor()
    
    print("批量处理器测试:")
    
    # 准备测试数据
    test_data = []
    for i in range(10):
        test_data.append({
            'title': f'测试报告{i+1}',
            'content': f'这是第{i+1}份测试报告的内容。' * 20,
            'author': f'作者{i+1}',
            'stocks': [f'00000{i+1}.SZ']
        })
    
    print(f"准备测试数据: {len(test_data)}份")
    
    # 测试质量检查处理
    print("\n开始质量检查批处理...")
    result = processor.process_batch(
        data_list=test_data,
        processor_type='quality_check',
        mode=ProcessingMode.THREADING
    )
    
    print(f"批处理结果:")
    print(f"  批次ID: {result.batch_id}")
    print(f"  总任务数: {result.total_tasks}")
    print(f"  成功任务: {result.completed_tasks}")
    print(f"  失败任务: {result.failed_tasks}")
    print(f"  处理时间: {result.processing_time:.2f}秒")
    print(f"  平均任务时间: {result.metadata['avg_task_time']:.3f}秒")
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"\n统计信息: {stats['batch_statistics']}") 
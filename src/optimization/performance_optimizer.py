"""
性能优化器
提供系统性能监控和自动优化功能
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import gc
import weakref

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceOptimizer:
    """
    性能优化器
    
    功能：
    1. 实时性能监控
    2. 自动内存管理
    3. 缓存优化
    4. 并发控制
    5. 性能瓶颈分析
    6. 自动调优建议
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化性能优化器"""
        self.config = config or {}
        
        # 性能指标收集
        self.metrics = defaultdict(deque)
        self.metrics_lock = threading.Lock()
        
        # 性能阈值
        self.thresholds = {
            'cpu_warning': self.config.get('cpu_warning', 80.0),
            'cpu_critical': self.config.get('cpu_critical', 95.0),
            'memory_warning': self.config.get('memory_warning', 80.0),
            'memory_critical': self.config.get('memory_critical', 95.0),
            'response_time_warning': self.config.get('response_time_warning', 2.0),
            'response_time_critical': self.config.get('response_time_critical', 5.0)
        }
        
        # 优化器状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.optimization_callbacks = []
        
        # 缓存管理
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'evictions': 0
        }
        
        # 内存管理
        self.memory_monitor = MemoryMonitor()
        
        # 并发控制
        self.concurrency_limiter = ConcurrencyLimiter(
            max_concurrent=self.config.get('max_concurrent', 50)
        )
        
        logger.info("性能优化器初始化完成")
    
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        if self.is_monitoring:
            logger.warning("性能监控已在运行")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("性能监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                
                with self.metrics_lock:
                    for key, value in metrics.items():
                        self.metrics[key].append((time.time(), value))
                        # 保留最近1000个数据点
                        if len(self.metrics[key]) > 1000:
                            self.metrics[key].popleft()
                
                # 检查阈值并触发优化
                self._check_thresholds(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"性能监控异常: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available,
                'disk_percent': disk_percent,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'process_memory_mb': process_memory,
                'process_cpu_percent': process_cpu
            }
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return {}
    
    def _check_thresholds(self, metrics: Dict[str, float]):
        """检查性能阈值"""
        alerts = []
        
        # CPU检查
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append(('critical', 'cpu', f'CPU使用率过高: {cpu_percent:.1f}%'))
            self._trigger_cpu_optimization()
        elif cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append(('warning', 'cpu', f'CPU使用率较高: {cpu_percent:.1f}%'))
        
        # 内存检查
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent >= self.thresholds['memory_critical']:
            alerts.append(('critical', 'memory', f'内存使用率过高: {memory_percent:.1f}%'))
            self._trigger_memory_optimization()
        elif memory_percent >= self.thresholds['memory_warning']:
            alerts.append(('warning', 'memory', f'内存使用率较高: {memory_percent:.1f}%'))
        
        # 记录告警
        for level, category, message in alerts:
            logger.warning(f"性能告警 [{level}] {category}: {message}")
    
    def _trigger_cpu_optimization(self):
        """触发CPU优化"""
        logger.info("执行CPU优化措施")
        
        # 降低并发数
        current_max = self.concurrency_limiter.max_concurrent
        new_max = max(1, int(current_max * 0.8))
        self.concurrency_limiter.set_max_concurrent(new_max)
        
        # 触发垃圾回收
        gc.collect()
        
        # 通知优化回调
        self._notify_optimization_callbacks('cpu_optimization', {
            'action': 'reduce_concurrency',
            'old_max': current_max,
            'new_max': new_max
        })
    
    def _trigger_memory_optimization(self):
        """触发内存优化"""
        logger.info("执行内存优化措施")
        
        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收释放了 {collected} 个对象")
        
        # 清理缓存
        self.memory_monitor.cleanup_weak_references()
        
        # 通知优化回调
        self._notify_optimization_callbacks('memory_optimization', {
            'action': 'garbage_collection',
            'objects_collected': collected
        })
    
    def register_optimization_callback(self, callback: Callable):
        """注册优化回调"""
        self.optimization_callbacks.append(callback)
    
    def _notify_optimization_callbacks(self, event_type: str, data: Dict[str, Any]):
        """通知优化回调"""
        for callback in self.optimization_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"优化回调执行失败: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.metrics_lock:
            # 计算平均值和趋势
            report = {}
            
            for metric_name, metric_data in self.metrics.items():
                if not metric_data:
                    continue
                
                values = [value for _, value in metric_data]
                recent_values = values[-60:]  # 最近60个数据点
                
                report[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values),
                    'recent_average': sum(recent_values) / len(recent_values) if recent_values else 0,
                    'min': min(values),
                    'max': max(values),
                    'trend': self._calculate_trend(recent_values)
                }
            
            # 添加缓存统计
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
            
            report['cache_performance'] = {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                **self.cache_stats
            }
            
            # 添加并发统计
            report['concurrency'] = {
                'max_concurrent': self.concurrency_limiter.max_concurrent,
                'current_concurrent': self.concurrency_limiter.current_count,
                'total_requests': self.concurrency_limiter.total_requests,
                'rejected_requests': self.concurrency_limiter.rejected_requests
            }
            
            return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单线性趋势计算
        mid_point = len(values) // 2
        first_half_avg = sum(values[:mid_point]) / mid_point
        second_half_avg = sum(values[mid_point:]) / (len(values) - mid_point)
        
        diff_percent = (second_half_avg - first_half_avg) / first_half_avg * 100
        
        if diff_percent > 10:
            return 'increasing'
        elif diff_percent < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def optimize_for_batch_processing(self):
        """针对批量处理进行优化"""
        logger.info("启用批量处理优化")
        
        # 调整并发数
        cpu_count = psutil.cpu_count()
        optimal_concurrent = min(cpu_count * 2, 20)
        self.concurrency_limiter.set_max_concurrent(optimal_concurrent)
        
        # 预分配内存
        gc.set_threshold(700, 10, 10)  # 调整垃圾回收阈值
        
        return {
            'max_concurrent': optimal_concurrent,
            'memory_optimization': True,
            'gc_tuned': True
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        report = self.get_performance_report()
        
        # CPU优化建议
        cpu_current = report.get('cpu_percent', {}).get('current', 0)
        if cpu_current > 80:
            recommendations.append("CPU使用率较高，建议减少并发处理数量")
        
        # 内存优化建议
        memory_current = report.get('memory_percent', {}).get('current', 0)
        if memory_current > 80:
            recommendations.append("内存使用率较高，建议启用更频繁的垃圾回收")
        
        # 缓存优化建议
        cache_hit_rate = report.get('cache_performance', {}).get('hit_rate', 0)
        if cache_hit_rate < 0.5:
            recommendations.append("缓存命中率较低，建议调整缓存策略")
        
        # 并发优化建议
        concurrency = report.get('concurrency', {})
        if concurrency.get('rejected_requests', 0) > 0:
            recommendations.append("存在请求被拒绝，建议增加并发处理能力")
        
        return recommendations


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.weak_refs = set()
    
    def track_object(self, obj):
        """跟踪对象"""
        self.weak_refs.add(weakref.ref(obj))
    
    def cleanup_weak_references(self):
        """清理弱引用"""
        dead_refs = {ref for ref in self.weak_refs if ref() is None}
        self.weak_refs -= dead_refs
        return len(dead_refs)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024**2),
            'vms_mb': memory_info.vms / (1024**2),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024**2)
        }


class ConcurrencyLimiter:
    """并发限制器"""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self.total_requests = 0
        self.rejected_requests = 0
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
    
    def set_max_concurrent(self, new_max: int):
        """设置最大并发数"""
        with self._lock:
            if new_max != self.max_concurrent:
                self.max_concurrent = new_max
                self._semaphore = threading.Semaphore(new_max)
                logger.info(f"并发限制调整为: {new_max}")
    
    async def acquire(self):
        """获取并发许可"""
        with self._lock:
            self.total_requests += 1
            
            if self.current_count >= self.max_concurrent:
                self.rejected_requests += 1
                raise RuntimeError("超过最大并发限制")
            
            self.current_count += 1
    
    def release(self):
        """释放并发许可"""
        with self._lock:
            if self.current_count > 0:
                self.current_count -= 1


def get_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """获取性能优化器实例"""
    return PerformanceOptimizer(config) 
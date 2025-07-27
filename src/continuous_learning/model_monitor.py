"""
模型性能监控器
实时监控模型性能指标、系统资源使用、预测质量和异常检测效果
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque, defaultdict
import warnings

from ..utils.logger import get_logger
from ..utils.config_loader import load_config
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    PERFORMANCE = "performance"     # 性能指标
    ACCURACY = "accuracy"          # 准确性指标
    RESOURCE = "resource"          # 资源使用
    LATENCY = "latency"           # 延迟指标
    QUALITY = "quality"           # 质量指标


@dataclass
class PerformanceMetric:
    """性能指标数据结构"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    alert_level: Optional[AlertLevel] = None
    context: Optional[Dict[str, Any]] = None
    
    def is_within_threshold(self) -> bool:
        """检查是否在阈值范围内"""
        if self.threshold_low is not None and self.value < self.threshold_low:
            return False
        if self.threshold_high is not None and self.value > self.threshold_high:
            return False
        return True


@dataclass
class SystemStatus:
    """系统状态"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    disk_usage: float
    network_io: Dict[str, float]
    model_status: str
    active_processes: int
    queue_size: int
    error_count: int


@dataclass
class ModelAlert:
    """模型告警"""
    alert_id: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ModelMonitor:
    """
    模型性能监控器
    
    功能：
    1. 实时性能监控 - CPU、内存、GPU使用率监控
    2. 模型质量监控 - 准确率、F1分数、AUC等指标跟踪
    3. 延迟监控 - 推理时间、响应时间监控
    4. 资源监控 - 系统资源使用情况监控
    5. 告警系统 - 异常情况自动告警
    6. 趋势分析 - 性能趋势分析和预测
    7. 健康检查 - 模型健康状态评估
    8. 数据漂移检测 - 输入数据分布变化检测
    
    Args:
        config: 监控配置
        
    Attributes:
        metrics_buffer: 指标缓冲区
        alerts: 告警列表
        thresholds: 阈值配置
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化监控器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 数据存储
        self.data_dir = Path("data/monitoring")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "monitoring.db"
        
        # 指标缓冲区
        self.metrics_buffer = deque(maxlen=1000)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 告警管理
        self.alerts: List[ModelAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # 阈值配置
        self.thresholds = self._load_thresholds()
        
        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitor_lock = threading.RLock()
        
        # 统计信息
        self.start_time = datetime.now()
        self.total_predictions = 0
        self.error_count = 0
        
        # 初始化数据库
        self._initialize_database()
        
        logger.info("模型监控器初始化完成")
    
    def _load_thresholds(self) -> Dict[str, Dict[str, float]]:
        """加载阈值配置"""
        default_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'gpu_usage': {'warning': 90.0, 'critical': 98.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'inference_time': {'warning': 5.0, 'critical': 10.0},
            'accuracy': {'warning': 0.8, 'critical': 0.7},
            'f1_score': {'warning': 0.75, 'critical': 0.65},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'queue_size': {'warning': 100, 'critical': 500}
        }
        
        try:
            config_path = Path("configs/monitoring_thresholds.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_thresholds = yaml.safe_load(f)
                    default_thresholds.update(custom_thresholds.get('thresholds', {}))
        except Exception as e:
            logger.warning(f"加载阈值配置失败，使用默认配置: {e}")
        
        return default_thresholds
    
    def _initialize_database(self):
        """初始化监控数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        threshold_low REAL,
                        threshold_high REAL,
                        alert_level TEXT,
                        context TEXT
                    )
                ''')
                
                # 创建告警表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        alert_level TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_time TEXT
                    )
                ''')
                
                # 创建系统状态表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_usage REAL NOT NULL,
                        memory_usage REAL NOT NULL,
                        gpu_usage REAL,
                        disk_usage REAL NOT NULL,
                        network_io TEXT,
                        model_status TEXT NOT NULL,
                        active_processes INTEGER,
                        queue_size INTEGER,
                        error_count INTEGER
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_timestamp ON system_status(timestamp)')
                
                conn.commit()
                
            logger.info("监控数据库初始化完成")
            
        except Exception as e:
            logger.error(f"监控数据库初始化失败: {e}")
            raise
    
    def start_monitoring(self, interval: float = 30.0):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            logger.warning("监控已经在运行中")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"开始模型监控，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("模型监控已停止")
    
    def _monitoring_loop(self, interval: float):
        """监控主循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                self._collect_system_metrics()
                
                # 检查告警
                self._check_alerts()
                
                # 清理旧数据
                self._cleanup_old_data()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            timestamp = datetime.now()
            
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_usage, MetricType.RESOURCE, timestamp)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self.record_metric("memory_usage", memory_usage, MetricType.RESOURCE, timestamp)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self.record_metric("disk_usage", disk_usage, MetricType.RESOURCE, timestamp)
            
            # GPU使用率（如果可用）
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                self.record_metric("gpu_usage", gpu_usage, MetricType.RESOURCE, timestamp)
            
            # 网络IO
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # 进程信息
            active_processes = len(psutil.pids())
            
            # 保存系统状态
            status = SystemStatus(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                model_status="running",
                active_processes=active_processes,
                queue_size=0,  # 需要从实际队列获取
                error_count=self.error_count
            )
            
            self._save_system_status(status)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def _get_gpu_usage(self) -> Optional[float]:
        """获取GPU使用率"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"获取GPU使用率失败: {e}")
        
        return None
    
    def record_metric(self, metric_name: str, value: float, 
                     metric_type: MetricType, timestamp: Optional[datetime] = None,
                     context: Optional[Dict[str, Any]] = None):
        """
        记录指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            metric_type: 指标类型
            timestamp: 时间戳
            context: 上下文信息
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # 获取阈值
            thresholds = self.thresholds.get(metric_name, {})
            threshold_low = thresholds.get('low')
            threshold_high = thresholds.get('high', thresholds.get('warning'))
            
            # 创建指标对象
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                context=context
            )
            
            # 检查是否超出阈值
            if not metric.is_within_threshold():
                self._handle_threshold_violation(metric)
            
            # 添加到缓冲区
            with self.monitor_lock:
                self.metrics_buffer.append(metric)
                self.metrics_history[metric_name].append((timestamp, value))
            
            # 保存到数据库
            self._save_metric_to_db(metric)
            
        except Exception as e:
            logger.error(f"记录指标失败: {e}")
    
    def _handle_threshold_violation(self, metric: PerformanceMetric):
        """处理阈值违规"""
        try:
            thresholds = self.thresholds.get(metric.metric_name, {})
            
            # 确定告警级别
            if 'critical' in thresholds and metric.value >= thresholds['critical']:
                alert_level = AlertLevel.CRITICAL
                threshold_value = thresholds['critical']
            elif 'warning' in thresholds and metric.value >= thresholds['warning']:
                alert_level = AlertLevel.WARNING
                threshold_value = thresholds['warning']
            else:
                return
            
            # 创建告警
            alert = ModelAlert(
                alert_id=f"{metric.metric_name}_{metric.timestamp.timestamp()}",
                alert_level=alert_level,
                metric_name=metric.metric_name,
                current_value=metric.value,
                threshold_value=threshold_value,
                message=f"{metric.metric_name} 超出阈值: {metric.value:.2f} > {threshold_value:.2f}",
                timestamp=metric.timestamp
            )
            
            self.alerts.append(alert)
            self._save_alert_to_db(alert)
            
            # 触发告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调执行失败: {e}")
            
            logger.warning(f"阈值告警: {alert.message}")
            
        except Exception as e:
            logger.error(f"处理阈值违规失败: {e}")
    
    def record_prediction_metrics(self, prediction_time: float, 
                                accuracy: Optional[float] = None,
                                confidence: Optional[float] = None):
        """
        记录预测指标
        
        Args:
            prediction_time: 预测耗时
            accuracy: 准确率
            confidence: 置信度
        """
        try:
            timestamp = datetime.now()
            
            # 记录预测时间
            self.record_metric("inference_time", prediction_time, MetricType.LATENCY, timestamp)
            
            # 记录准确率
            if accuracy is not None:
                self.record_metric("accuracy", accuracy, MetricType.ACCURACY, timestamp)
            
            # 记录置信度
            if confidence is not None:
                self.record_metric("confidence", confidence, MetricType.QUALITY, timestamp)
            
            # 更新统计
            with self.monitor_lock:
                self.total_predictions += 1
            
        except Exception as e:
            logger.error(f"记录预测指标失败: {e}")
    
    def record_error(self, error_type: str, error_message: str):
        """记录错误"""
        try:
            with self.monitor_lock:
                self.error_count += 1
            
            # 记录错误率
            if self.total_predictions > 0:
                error_rate = self.error_count / self.total_predictions
                self.record_metric("error_rate", error_rate, MetricType.QUALITY)
            
            logger.error(f"记录错误: {error_type} - {error_message}")
            
        except Exception as e:
            logger.error(f"记录错误失败: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        try:
            with self.monitor_lock:
                latest_metrics = {}
                
                # 获取最新指标
                for metric in reversed(self.metrics_buffer):
                    if metric.metric_name not in latest_metrics:
                        latest_metrics[metric.metric_name] = {
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat(),
                            'type': metric.metric_type.value,
                            'within_threshold': metric.is_within_threshold()
                        }
                
                return {
                    'metrics': latest_metrics,
                    'system_health': self._calculate_system_health(),
                    'total_predictions': self.total_predictions,
                    'error_count': self.error_count,
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                    'active_alerts': len([a for a in self.alerts if not a.resolved])
                }
                
        except Exception as e:
            logger.error(f"获取当前指标失败: {e}")
            return {}
    
    def _calculate_system_health(self) -> str:
        """计算系统健康状态"""
        try:
            # 检查关键指标
            latest_metrics = {}
            for metric in reversed(self.metrics_buffer):
                if metric.metric_name not in latest_metrics:
                    latest_metrics[metric.metric_name] = metric
                if len(latest_metrics) >= 5:  # 检查主要指标
                    break
            
            critical_issues = 0
            warning_issues = 0
            
            for metric in latest_metrics.values():
                if not metric.is_within_threshold():
                    thresholds = self.thresholds.get(metric.metric_name, {})
                    if 'critical' in thresholds and metric.value >= thresholds['critical']:
                        critical_issues += 1
                    else:
                        warning_issues += 1
            
            # 判断健康状态
            if critical_issues > 0:
                return "critical"
            elif warning_issues > 2:
                return "warning"
            elif warning_issues > 0:
                return "caution"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error(f"计算系统健康状态失败: {e}")
            return "unknown"
    
    def get_performance_trends(self, metric_name: str, 
                             hours: int = 24) -> Dict[str, Any]:
        """
        获取性能趋势
        
        Args:
            metric_name: 指标名称
            hours: 时间范围（小时）
            
        Returns:
            Dict: 趋势数据
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, value FROM metrics 
                    WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(metric_name, start_time.isoformat(), end_time.isoformat())
                )
            
            if df.empty:
                return {'error': '没有找到数据'}
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 计算统计信息
            stats = {
                'mean': df['value'].mean(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max(),
                'current': df['value'].iloc[-1] if not df.empty else None,
                'trend': 'stable'
            }
            
            # 计算趋势
            if len(df) > 10:
                recent_mean = df['value'].tail(10).mean()
                earlier_mean = df['value'].head(10).mean()
                
                if recent_mean > earlier_mean * 1.1:
                    stats['trend'] = 'increasing'
                elif recent_mean < earlier_mean * 0.9:
                    stats['trend'] = 'decreasing'
            
            return {
                'metric_name': metric_name,
                'time_range_hours': hours,
                'data_points': len(df),
                'statistics': stats,
                'data': df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {'error': str(e)}
    
    def _save_metric_to_db(self, metric: PerformanceMetric):
        """保存指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (
                        metric_name, metric_type, value, timestamp,
                        threshold_low, threshold_high, alert_level, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.metric_name,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    metric.threshold_low,
                    metric.threshold_high,
                    metric.alert_level.value if metric.alert_level else None,
                    json.dumps(metric.context) if metric.context else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存指标到数据库失败: {e}")
    
    def _save_alert_to_db(self, alert: ModelAlert):
        """保存告警到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts (
                        alert_id, alert_level, metric_name, current_value,
                        threshold_value, message, timestamp, resolved, resolution_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.alert_level.value,
                    alert.metric_name,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.resolved,
                    alert.resolution_time.isoformat() if alert.resolution_time else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存告警到数据库失败: {e}")
    
    def _save_system_status(self, status: SystemStatus):
        """保存系统状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_status (
                        timestamp, cpu_usage, memory_usage, gpu_usage, disk_usage,
                        network_io, model_status, active_processes, queue_size, error_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    status.timestamp.isoformat(),
                    status.cpu_usage,
                    status.memory_usage,
                    status.gpu_usage,
                    status.disk_usage,
                    json.dumps(status.network_io),
                    status.model_status,
                    status.active_processes,
                    status.queue_size,
                    status.error_count
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")
    
    def _check_alerts(self):
        """检查告警状态"""
        try:
            # 检查是否有告警需要自动解决
            current_time = datetime.now()
            
            for alert in self.alerts:
                if not alert.resolved:
                    # 检查告警是否应该自动解决
                    # 例如：如果指标恢复正常超过5分钟
                    if (current_time - alert.timestamp).total_seconds() > 300:
                        recent_metrics = [
                            m for m in self.metrics_buffer
                            if (m.metric_name == alert.metric_name and 
                                m.timestamp > alert.timestamp)
                        ]
                        
                        if recent_metrics and all(m.is_within_threshold() for m in recent_metrics[-5:]):
                            alert.resolved = True
                            alert.resolution_time = current_time
                            self._save_alert_to_db(alert)
                            logger.info(f"告警自动解决: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            # 清理超过30天的数据
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理旧指标
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_date.isoformat(),))
                
                # 清理旧告警
                cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE', (cutoff_date.isoformat(),))
                
                # 清理旧状态
                cursor.execute('DELETE FROM system_status WHERE timestamp < ?', (cutoff_date.isoformat(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[ModelAlert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'start_time': self.start_time.isoformat(),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_predictions': self.total_predictions,
                'error_count': self.error_count,
                'metrics_in_buffer': len(self.metrics_buffer),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'system_health': self._calculate_system_health(),
                'database_path': str(self.db_path),
                'thresholds_configured': len(self.thresholds)
            }
            
        except Exception as e:
            logger.error(f"获取监控摘要失败: {e}")
            return {'error': str(e)}


# 全局监控器实例
_global_monitor = None


def get_model_monitor() -> ModelMonitor:
    """
    获取全局模型监控器实例
    
    Returns:
        ModelMonitor: 监控器实例
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = ModelMonitor()
    
    return _global_monitor


if __name__ == "__main__":
    # 使用示例
    print("模型监控器测试:")
    
    # 创建监控器
    monitor = ModelMonitor()
    
    # 开始监控
    monitor.start_monitoring(interval=5.0)
    
    # 模拟记录一些指标
    monitor.record_metric("test_accuracy", 0.95, MetricType.ACCURACY)
    monitor.record_prediction_metrics(prediction_time=1.2, accuracy=0.92, confidence=0.88)
    
    # 获取当前指标
    current = monitor.get_current_metrics()
    print(f"当前指标: {current}")
    
    # 获取摘要
    summary = monitor.get_monitoring_summary()
    print(f"监控摘要: {summary}")
    
    # 停止监控
    time.sleep(10)
    monitor.stop_monitoring() 
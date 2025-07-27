"""
性能跟踪器
长期跟踪模型性能趋势、生成性能报告、提供决策支持
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import warnings
import base64
import io

from .feedback_collector import get_feedback_collector
from .model_monitor import get_model_monitor
from .adaptive_learner import get_adaptive_learner
from ..utils.logger import get_logger
from ..utils.config_loader import load_config
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportType(Enum):
    """报告类型"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class MetricTrend(Enum):
    """指标趋势"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class PerformanceSummary:
    """性能摘要"""
    period_start: datetime
    period_end: datetime
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    average_confidence: float
    error_count: int
    user_satisfaction: float
    
    # 趋势信息
    accuracy_trend: MetricTrend
    performance_change: float
    
    # 对比基线
    baseline_comparison: Dict[str, float]
    
    # 关键事件
    adaptations_count: int
    alerts_count: int


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    summary: PerformanceSummary
    detailed_metrics: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    
    # 可视化图表
    charts: Dict[str, str]  # base64编码的图片
    
    # 原始数据
    raw_data: Optional[Dict[str, Any]] = None


class PerformanceTracker:
    """
    性能跟踪器
    
    功能：
    1. 长期性能跟踪 - 收集和存储历史性能数据
    2. 趋势分析 - 识别性能变化趋势和模式
    3. 基线管理 - 建立和维护性能基线
    4. 报告生成 - 自动生成性能报告
    5. 可视化分析 - 生成性能趋势图表
    6. 异常检测 - 识别性能异常和退化
    7. 对比分析 - 不同时期性能对比
    8. 预测分析 - 性能趋势预测
    
    Args:
        config: 跟踪器配置
        
    Attributes:
        performance_data: 性能数据缓存
        reports: 生成的报告
        baselines: 性能基线
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化性能跟踪器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 获取其他组件
        self.feedback_collector = get_feedback_collector()
        self.model_monitor = get_model_monitor()
        self.adaptive_learner = get_adaptive_learner()
        
        # 数据存储
        self.data_dir = Path("data/performance_tracking")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "performance_tracking.db"
        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.data_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.reports: List[PerformanceReport] = []
        
        # 性能基线
        self.baselines = self._load_baselines()
        
        # 初始化数据库
        self._initialize_database()
        
        # 加载历史数据
        self._load_recent_data()
        
        logger.info("性能跟踪器初始化完成")
    
    def _initialize_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 性能快照表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        false_positive_rate REAL,
                        confidence REAL,
                        prediction_count INTEGER,
                        error_count INTEGER,
                        user_satisfaction REAL,
                        context TEXT
                    )
                ''')
                
                # 性能报告表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_reports (
                        report_id TEXT PRIMARY KEY,
                        report_type TEXT NOT NULL,
                        generated_at TEXT NOT NULL,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        summary_data TEXT NOT NULL,
                        detailed_data TEXT,
                        charts_data TEXT,
                        file_path TEXT
                    )
                ''')
                
                # 性能基线表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_baselines (
                        baseline_id TEXT PRIMARY KEY,
                        metric_name TEXT NOT NULL,
                        baseline_value REAL NOT NULL,
                        confidence_interval TEXT,
                        established_date TEXT NOT NULL,
                        data_points INTEGER,
                        context TEXT
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON performance_snapshots(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reports_generated ON performance_reports(generated_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_baselines_metric ON performance_baselines(metric_name)')
                
                conn.commit()
                
            logger.info("性能跟踪数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _load_baselines(self) -> Dict[str, float]:
        """加载性能基线"""
        try:
            baselines_file = self.data_dir / "baselines.json"
            if baselines_file.exists():
                with open(baselines_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 默认基线
            return {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77,
                'false_positive_rate': 0.15,
                'user_satisfaction': 0.70
            }
            
        except Exception as e:
            logger.error(f"加载基线失败: {e}")
            return {}
    
    def _load_recent_data(self):
        """加载最近的性能数据"""
        try:
            # 加载最近30天的数据
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM performance_snapshots 
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                '''
                
                cursor = conn.cursor()
                cursor.execute(query, (cutoff_date.isoformat(),))
                rows = cursor.fetchall()
                
                for row in rows:
                    timestamp = datetime.fromisoformat(row[1])
                    
                    # 将数据添加到缓存
                    snapshot = {
                        'timestamp': timestamp,
                        'accuracy': row[2],
                        'precision': row[3],
                        'recall': row[4],
                        'f1_score': row[5],
                        'false_positive_rate': row[6],
                        'confidence': row[7],
                        'prediction_count': row[8],
                        'error_count': row[9],
                        'user_satisfaction': row[10]
                    }
                    
                    for metric, value in snapshot.items():
                        if metric != 'timestamp' and value is not None:
                            self.performance_data[metric].append((timestamp, value))
            
            logger.info(f"加载了 {len(rows)} 条历史性能数据")
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
    
    def capture_performance_snapshot(self, context: Optional[Dict[str, Any]] = None):
        """
        捕获性能快照
        
        Args:
            context: 上下文信息
        """
        try:
            timestamp = datetime.now()
            
            # 从监控器获取当前指标
            current_metrics = self.model_monitor.get_current_metrics()
            
            # 从反馈收集器获取用户满意度
            feedback_stats = self.feedback_collector.calculate_statistics()
            
            # 构建快照
            snapshot = {
                'timestamp': timestamp,
                'accuracy': self._extract_metric_value(current_metrics, 'accuracy'),
                'precision': self._extract_metric_value(current_metrics, 'precision'),
                'recall': self._extract_metric_value(current_metrics, 'recall'),
                'f1_score': self._extract_metric_value(current_metrics, 'f1_score'),
                'false_positive_rate': self._extract_metric_value(current_metrics, 'error_rate'),
                'confidence': self._extract_metric_value(current_metrics, 'confidence'),
                'prediction_count': current_metrics.get('total_predictions', 0),
                'error_count': current_metrics.get('error_count', 0),
                'user_satisfaction': feedback_stats.user_satisfaction
            }
            
            # 添加到缓存
            for metric, value in snapshot.items():
                if metric != 'timestamp' and value is not None:
                    self.performance_data[metric].append((timestamp, value))
            
            # 保存到数据库
            self._save_snapshot_to_db(snapshot, context)
            
            logger.debug(f"性能快照已捕获: {timestamp}")
            
        except Exception as e:
            logger.error(f"捕获性能快照失败: {e}")
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """从指标数据中提取值"""
        try:
            metric_data = metrics.get('metrics', {}).get(metric_name)
            if metric_data:
                return metric_data.get('value')
            return None
        except Exception:
            return None
    
    def _save_snapshot_to_db(self, snapshot: Dict[str, Any], context: Optional[Dict[str, Any]]):
        """保存快照到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_snapshots (
                        timestamp, accuracy, precision_score, recall, f1_score,
                        false_positive_rate, confidence, prediction_count,
                        error_count, user_satisfaction, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot['timestamp'].isoformat(),
                    snapshot.get('accuracy'),
                    snapshot.get('precision'),
                    snapshot.get('recall'),
                    snapshot.get('f1_score'),
                    snapshot.get('false_positive_rate'),
                    snapshot.get('confidence'),
                    snapshot.get('prediction_count'),
                    snapshot.get('error_count'),
                    snapshot.get('user_satisfaction'),
                    json.dumps(context) if context else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存快照到数据库失败: {e}")
    
    def analyze_performance_trend(self, metric_name: str, 
                                 days: int = 30) -> Dict[str, Any]:
        """
        分析性能趋势
        
        Args:
            metric_name: 指标名称
            days: 分析天数
            
        Returns:
            Dict: 趋势分析结果
        """
        try:
            # 获取指定时间范围的数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            metric_data = self.performance_data.get(metric_name, deque())
            
            # 过滤时间范围
            filtered_data = [
                (timestamp, value) for timestamp, value in metric_data
                if start_time <= timestamp <= end_time
            ]
            
            if len(filtered_data) < 5:
                return {
                    'trend': MetricTrend.UNKNOWN,
                    'message': '数据点不足，无法分析趋势',
                    'data_points': len(filtered_data)
                }
            
            # 提取值和时间
            values = [value for _, value in filtered_data]
            timestamps = [timestamp for timestamp, _ in filtered_data]
            
            # 计算趋势
            trend = self._calculate_trend(values)
            
            # 统计信息
            stats = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'current': values[-1],
                'change_from_start': values[-1] - values[0],
                'percent_change': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
            }
            
            # 与基线对比
            baseline_value = self.baselines.get(metric_name)
            baseline_comparison = None
            if baseline_value:
                baseline_comparison = {
                    'baseline_value': baseline_value,
                    'current_vs_baseline': values[-1] - baseline_value,
                    'percent_vs_baseline': ((values[-1] - baseline_value) / baseline_value) * 100
                }
            
            return {
                'metric_name': metric_name,
                'trend': trend,
                'time_range_days': days,
                'data_points': len(filtered_data),
                'statistics': stats,
                'baseline_comparison': baseline_comparison,
                'data': filtered_data
            }
            
        except Exception as e:
            logger.error(f"分析性能趋势失败: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> MetricTrend:
        """计算趋势"""
        try:
            if len(values) < 3:
                return MetricTrend.UNKNOWN
            
            # 计算线性回归斜率
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # 计算变异系数
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # 判断趋势
            if cv > 0.2:  # 变异系数大于20%
                return MetricTrend.VOLATILE
            elif slope > 0.01:  # 正向趋势
                return MetricTrend.IMPROVING
            elif slope < -0.01:  # 负向趋势
                return MetricTrend.DECLINING
            else:
                return MetricTrend.STABLE
                
        except Exception as e:
            logger.error(f"计算趋势失败: {e}")
            return MetricTrend.UNKNOWN
    
    def generate_performance_report(self, report_type: ReportType,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> PerformanceReport:
        """
        生成性能报告
        
        Args:
            report_type: 报告类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            PerformanceReport: 性能报告
        """
        try:
            # 确定时间范围
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                if report_type == ReportType.DAILY:
                    start_date = end_date - timedelta(days=1)
                elif report_type == ReportType.WEEKLY:
                    start_date = end_date - timedelta(days=7)
                elif report_type == ReportType.MONTHLY:
                    start_date = end_date - timedelta(days=30)
                elif report_type == ReportType.QUARTERLY:
                    start_date = end_date - timedelta(days=90)
                else:
                    start_date = end_date - timedelta(days=7)
            
            # 生成报告ID
            report_id = f"{report_type.value}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # 收集数据
            summary = self._generate_performance_summary(start_date, end_date)
            detailed_metrics = self._collect_detailed_metrics(start_date, end_date)
            trend_analysis = self._analyze_period_trends(start_date, end_date)
            
            # 生成建议
            recommendations = self._generate_recommendations(summary, trend_analysis)
            
            # 生成图表
            charts = self._generate_performance_charts(start_date, end_date)
            
            # 创建报告
            report = PerformanceReport(
                report_id=report_id,
                report_type=report_type,
                generated_at=datetime.now(),
                period_start=start_date,
                period_end=end_date,
                summary=summary,
                detailed_metrics=detailed_metrics,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                charts=charts
            )
            
            # 保存报告
            self._save_report(report)
            
            logger.info(f"性能报告已生成: {report_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            raise
    
    def _generate_performance_summary(self, start_date: datetime, 
                                    end_date: datetime) -> PerformanceSummary:
        """生成性能摘要"""
        try:
            # 收集期间内的所有数据点
            period_data = {}
            total_predictions = 0
            error_count = 0
            
            for metric_name, data in self.performance_data.items():
                period_values = [
                    value for timestamp, value in data
                    if start_date <= timestamp <= end_date
                ]
                
                if period_values:
                    period_data[metric_name] = {
                        'values': period_values,
                        'mean': np.mean(period_values),
                        'latest': period_values[-1]
                    }
                    
                    if metric_name == 'prediction_count':
                        total_predictions = int(np.sum(period_values))
                    elif metric_name == 'error_count':
                        error_count = int(np.sum(period_values))
            
            # 计算主要指标
            accuracy = period_data.get('accuracy', {}).get('mean', 0.0)
            precision = period_data.get('precision', {}).get('mean', 0.0)
            recall = period_data.get('recall', {}).get('mean', 0.0)
            f1_score = period_data.get('f1_score', {}).get('mean', 0.0)
            false_positive_rate = period_data.get('false_positive_rate', {}).get('mean', 0.0)
            average_confidence = period_data.get('confidence', {}).get('mean', 0.0)
            user_satisfaction = period_data.get('user_satisfaction', {}).get('mean', 0.0)
            
            # 计算趋势
            accuracy_trend = self._calculate_trend(period_data.get('accuracy', {}).get('values', []))
            
            # 计算性能变化
            performance_change = 0.0
            if period_data.get('accuracy', {}).get('values'):
                values = period_data['accuracy']['values']
                if len(values) >= 2:
                    performance_change = values[-1] - values[0]
            
            # 与基线对比
            baseline_comparison = {}
            for metric_name, baseline_value in self.baselines.items():
                current_value = period_data.get(metric_name, {}).get('mean', 0.0)
                baseline_comparison[metric_name] = current_value - baseline_value
            
            # 获取适应和告警数量
            adaptations_count = 0
            alerts_count = 0
            
            # 这里需要从其他组件获取实际数据
            try:
                learner_summary = self.adaptive_learner.get_learning_summary()
                adaptations_count = learner_summary.get('recent_adaptations', 0)
                
                monitor_summary = self.model_monitor.get_monitoring_summary()
                alerts_count = monitor_summary.get('active_alerts', 0)
            except Exception:
                pass
            
            return PerformanceSummary(
                period_start=start_date,
                period_end=end_date,
                total_predictions=total_predictions,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                false_positive_rate=false_positive_rate,
                average_confidence=average_confidence,
                error_count=error_count,
                user_satisfaction=user_satisfaction,
                accuracy_trend=accuracy_trend,
                performance_change=performance_change,
                baseline_comparison=baseline_comparison,
                adaptations_count=adaptations_count,
                alerts_count=alerts_count
            )
            
        except Exception as e:
            logger.error(f"生成性能摘要失败: {e}")
            raise
    
    def _collect_detailed_metrics(self, start_date: datetime, 
                                end_date: datetime) -> Dict[str, Any]:
        """收集详细指标"""
        try:
            detailed = {}
            
            # 收集所有指标的详细统计
            for metric_name, data in self.performance_data.items():
                period_values = [
                    value for timestamp, value in data
                    if start_date <= timestamp <= end_date
                ]
                
                if period_values:
                    detailed[metric_name] = {
                        'count': len(period_values),
                        'mean': float(np.mean(period_values)),
                        'std': float(np.std(period_values)),
                        'min': float(np.min(period_values)),
                        'max': float(np.max(period_values)),
                        'median': float(np.median(period_values)),
                        'q25': float(np.percentile(period_values, 25)),
                        'q75': float(np.percentile(period_values, 75)),
                        'trend': self._calculate_trend(period_values).value
                    }
            
            return detailed
            
        except Exception as e:
            logger.error(f"收集详细指标失败: {e}")
            return {}
    
    def _analyze_period_trends(self, start_date: datetime, 
                             end_date: datetime) -> Dict[str, Any]:
        """分析期间趋势"""
        try:
            trends = {}
            
            # 分析每个指标的趋势
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                trend_result = self.analyze_performance_trend(
                    metric_name, 
                    days=(end_date - start_date).days
                )
                trends[metric_name] = trend_result
            
            return trends
            
        except Exception as e:
            logger.error(f"分析期间趋势失败: {e}")
            return {}
    
    def _generate_recommendations(self, summary: PerformanceSummary, 
                                trends: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        try:
            # 基于准确率趋势的建议
            if summary.accuracy_trend == MetricTrend.DECLINING:
                recommendations.append("⚠️ 检测准确率呈下降趋势，建议重新训练模型或调整检测阈值")
            
            # 基于误报率的建议
            if summary.false_positive_rate > 0.2:
                recommendations.append("🔍 误报率较高，建议提高检测阈值或优化特征工程")
            
            # 基于用户满意度的建议
            if summary.user_satisfaction < 0.7:
                recommendations.append("👥 用户满意度较低，建议收集更多用户反馈并改进模型")
            
            # 基于基线对比的建议
            accuracy_diff = summary.baseline_comparison.get('accuracy', 0)
            if accuracy_diff < -0.05:
                recommendations.append("📊 准确率低于基线，建议进行模型诊断和优化")
            
            # 基于自适应学习的建议
            if summary.adaptations_count == 0:
                recommendations.append("🔄 建议启用自适应学习功能以持续优化模型性能")
            
            # 基于告警的建议
            if summary.alerts_count > 0:
                recommendations.append(f"🚨 当前有 {summary.alerts_count} 个活跃告警，建议及时处理")
            
            # 如果没有具体建议，给出一般性建议
            if not recommendations:
                recommendations.append("✅ 模型性能表现良好，建议继续监控并定期评估")
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append("❗ 建议生成过程中出现错误，请手动检查系统状态")
        
        return recommendations
    
    def _generate_performance_charts(self, start_date: datetime, 
                                   end_date: datetime) -> Dict[str, str]:
        """生成性能图表"""
        charts = {}
        
        try:
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            
            # 1. 准确率趋势图
            charts['accuracy_trend'] = self._create_trend_chart(
                'accuracy', start_date, end_date, '准确率趋势'
            )
            
            # 2. 多指标对比图
            charts['metrics_comparison'] = self._create_multi_metrics_chart(
                start_date, end_date
            )
            
            # 3. 用户满意度图
            charts['user_satisfaction'] = self._create_trend_chart(
                'user_satisfaction', start_date, end_date, '用户满意度趋势'
            )
            
            # 4. 性能分布图
            charts['performance_distribution'] = self._create_distribution_chart(
                start_date, end_date
            )
            
        except Exception as e:
            logger.error(f"生成性能图表失败: {e}")
        
        return charts
    
    def _create_trend_chart(self, metric_name: str, start_date: datetime,
                          end_date: datetime, title: str) -> str:
        """创建趋势图表"""
        try:
            # 获取数据
            data = self.performance_data.get(metric_name, deque())
            period_data = [
                (timestamp, value) for timestamp, value in data
                if start_date <= timestamp <= end_date
            ]
            
            if not period_data:
                return ""
            
            timestamps, values = zip(*period_data)
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, values, linewidth=2, marker='o', markersize=4)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 添加基线
            if metric_name in self.baselines:
                baseline = self.baselines[metric_name]
                ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, label='基线')
                ax.legend()
            
            # 格式化x轴
            fig.autofmt_xdate()
            
            # 转换为base64
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"创建趋势图表失败: {e}")
            return ""
    
    def _create_multi_metrics_chart(self, start_date: datetime, 
                                  end_date: datetime) -> str:
        """创建多指标对比图"""
        try:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for metric in metrics:
                data = self.performance_data.get(metric, deque())
                period_data = [
                    (timestamp, value) for timestamp, value in data
                    if start_date <= timestamp <= end_date
                ]
                
                if period_data:
                    timestamps, values = zip(*period_data)
                    ax.plot(timestamps, values, label=metric.replace('_', ' ').title(), 
                           linewidth=2, marker='o', markersize=3)
            
            ax.set_title('性能指标对比', fontsize=16, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('分数', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            fig.autofmt_xdate()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"创建多指标对比图失败: {e}")
            return ""
    
    def _create_distribution_chart(self, start_date: datetime, 
                                 end_date: datetime) -> str:
        """创建性能分布图"""
        try:
            # 获取准确率数据
            data = self.performance_data.get('accuracy', deque())
            period_values = [
                value for timestamp, value in data
                if start_date <= timestamp <= end_date
            ]
            
            if not period_values:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 直方图
            ax1.hist(period_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('准确率分布', fontsize=14, fontweight='bold')
            ax1.set_xlabel('准确率', fontsize=12)
            ax1.set_ylabel('频次', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 箱线图
            ax2.boxplot(period_values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax2.set_title('准确率箱线图', fontsize=14, fontweight='bold')
            ax2.set_ylabel('准确率', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"创建分布图失败: {e}")
            return ""
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图表转换为base64字符串"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            # 转换为base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)  # 关闭图表释放内存
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"图表转换失败: {e}")
            plt.close(fig)
            return ""
    
    def _save_report(self, report: PerformanceReport):
        """保存报告"""
        try:
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_reports (
                        report_id, report_type, generated_at, period_start, period_end,
                        summary_data, detailed_data, charts_data, file_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.report_id,
                    report.report_type.value,
                    report.generated_at.isoformat(),
                    report.period_start.isoformat(),
                    report.period_end.isoformat(),
                    json.dumps(asdict(report.summary), default=str),
                    json.dumps(report.detailed_metrics),
                    json.dumps(report.charts),
                    None  # 文件路径暂时为空
                ))
                conn.commit()
            
            # 添加到报告列表
            self.reports.append(report)
            
            # 保持最近50个报告
            if len(self.reports) > 50:
                self.reports = self.reports[-50:]
                
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """获取性能仪表板数据"""
        try:
            # 获取最近7天的数据
            recent_summary = self._generate_performance_summary(
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            
            # 获取趋势数据
            trends = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                trends[metric] = self.analyze_performance_trend(metric, days=7)
            
            # 获取最近的报告
            recent_reports = sorted(self.reports, key=lambda x: x.generated_at, reverse=True)[:5]
            
            return {
                'summary': asdict(recent_summary),
                'trends': trends,
                'recent_reports': [
                    {
                        'report_id': r.report_id,
                        'report_type': r.report_type.value,
                        'generated_at': r.generated_at.isoformat(),
                        'period_start': r.period_start.isoformat(),
                        'period_end': r.period_end.isoformat()
                    }
                    for r in recent_reports
                ],
                'system_health': self.model_monitor.get_current_metrics().get('system_health', 'unknown'),
                'data_freshness': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取仪表板数据失败: {e}")
            return {'error': str(e)}
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """获取跟踪摘要"""
        try:
            return {
                'data_points_cached': sum(len(data) for data in self.performance_data.values()),
                'metrics_tracked': list(self.performance_data.keys()),
                'reports_generated': len(self.reports),
                'baselines_configured': len(self.baselines),
                'database_path': str(self.db_path),
                'charts_directory': str(self.charts_dir),
                'last_snapshot': max([
                    max(data, key=lambda x: x[0])[0] if data else datetime.min
                    for data in self.performance_data.values()
                ], default=datetime.min).isoformat() if any(self.performance_data.values()) else None
            }
            
        except Exception as e:
            logger.error(f"获取跟踪摘要失败: {e}")
            return {'error': str(e)}


# 全局性能跟踪器实例
_global_performance_tracker = None


def get_performance_tracker() -> PerformanceTracker:
    """
    获取全局性能跟踪器实例
    
    Returns:
        PerformanceTracker: 跟踪器实例
    """
    global _global_performance_tracker
    
    if _global_performance_tracker is None:
        _global_performance_tracker = PerformanceTracker()
    
    return _global_performance_tracker


if __name__ == "__main__":
    # 使用示例
    print("性能跟踪器测试:")
    
    # 创建跟踪器
    tracker = PerformanceTracker()
    
    # 捕获性能快照
    tracker.capture_performance_snapshot()
    
    # 分析趋势
    trend = tracker.analyze_performance_trend('accuracy', days=7)
    print(f"准确率趋势: {trend}")
    
    # 生成报告
    report = tracker.generate_performance_report(ReportType.WEEKLY)
    print(f"报告生成: {report.report_id}")
    
    # 获取仪表板数据
    dashboard = tracker.get_performance_dashboard_data()
    print(f"仪表板数据: {dashboard}")
    
    # 获取摘要
    summary = tracker.get_tracking_summary()
    print(f"跟踪摘要: {summary}") 
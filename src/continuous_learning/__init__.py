"""
持续学习模块
Continuous Learning Module

提供完整的持续学习功能：
- 用户反馈收集和管理
- 模型性能实时监控  
- 自适应学习和参数优化
- 长期性能跟踪和报告

主要组件：
- FeedbackCollector: 用户反馈收集器
- ModelMonitor: 模型性能监控器
- AdaptiveLearner: 自适应学习器
- PerformanceTracker: 性能跟踪器
"""

from .feedback_collector import (
    FeedbackCollector,
    get_feedback_collector,
    UserFeedback,
    FeedbackType,
    FeedbackSource,
    FeedbackStatistics
)

from .model_monitor import (
    ModelMonitor,
    get_model_monitor,
    PerformanceMetric,
    SystemStatus,
    ModelAlert,
    AlertLevel,
    MetricType
)

from .adaptive_learner import (
    AdaptiveLearner,
    get_adaptive_learner,
    AdaptationResult,
    LearningObjective,
    AdaptationStrategy,
    LearningTrigger
)

from .performance_tracker import (
    PerformanceTracker,
    get_performance_tracker,
    PerformanceReport,
    PerformanceSummary,
    ReportType,
    MetricTrend
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Anomaly Detection Team"

# 导出的主要类和函数
__all__ = [
    # 反馈收集相关
    'FeedbackCollector',
    'get_feedback_collector',
    'UserFeedback',
    'FeedbackType',
    'FeedbackSource',
    'FeedbackStatistics',
    
    # 模型监控相关
    'ModelMonitor',
    'get_model_monitor',
    'PerformanceMetric',
    'SystemStatus',
    'ModelAlert',
    'AlertLevel',
    'MetricType',
    
    # 自适应学习相关
    'AdaptiveLearner',
    'get_adaptive_learner',
    'AdaptationResult',
    'LearningObjective',
    'AdaptationStrategy',
    'LearningTrigger',
    
    # 性能跟踪相关
    'PerformanceTracker',
    'get_performance_tracker',
    'PerformanceReport',
    'PerformanceSummary',
    'ReportType',
    'MetricTrend',
    
    # 便捷函数
    'get_continuous_learning_system',
    'initialize_continuous_learning'
]


class ContinuousLearningSystem:
    """
    持续学习系统整合类
    
    整合所有持续学习组件，提供统一的接口和协调管理
    """
    
    def __init__(self):
        """初始化持续学习系统"""
        self.feedback_collector = get_feedback_collector()
        self.model_monitor = get_model_monitor()
        self.adaptive_learner = get_adaptive_learner()
        self.performance_tracker = get_performance_tracker()
        
        self._is_running = False
    
    def start_system(self, monitor_interval: float = 30.0, 
                    learning_interval: float = 3600.0):
        """
        启动持续学习系统
        
        Args:
            monitor_interval: 监控间隔（秒）
            learning_interval: 学习检查间隔（秒）
        """
        if self._is_running:
            return
        
        # 启动监控
        self.model_monitor.start_monitoring(interval=monitor_interval)
        
        # 启动自适应学习
        self.adaptive_learner.start_adaptive_learning(check_interval=learning_interval)
        
        self._is_running = True
    
    def stop_system(self):
        """停止持续学习系统"""
        if not self._is_running:
            return
        
        # 停止监控
        self.model_monitor.stop_monitoring()
        
        # 停止自适应学习
        self.adaptive_learner.stop_adaptive_learning()
        
        self._is_running = False
    
    def record_prediction_feedback(self, report_id: str, 
                                 original_prediction: dict,
                                 is_correct: bool,
                                 feedback_type: FeedbackType = FeedbackType.CORRECT_DETECTION,
                                 **kwargs) -> str:
        """
        记录预测反馈
        
        Args:
            report_id: 报告ID
            original_prediction: 原始预测结果
            is_correct: 是否正确
            feedback_type: 反馈类型
            **kwargs: 其他参数
            
        Returns:
            str: 反馈ID
        """
        return self.feedback_collector.collect_feedback(
            report_id=report_id,
            original_prediction=original_prediction,
            feedback_type=feedback_type,
            is_correct=is_correct,
            **kwargs
        )
    
    def record_performance_metrics(self, prediction_time: float,
                                 accuracy: float = None,
                                 confidence: float = None):
        """
        记录性能指标
        
        Args:
            prediction_time: 预测耗时
            accuracy: 准确率
            confidence: 置信度
        """
        # 记录到监控器
        self.model_monitor.record_prediction_metrics(
            prediction_time=prediction_time,
            accuracy=accuracy,
            confidence=confidence
        )
        
        # 捕获性能快照
        self.performance_tracker.capture_performance_snapshot()
    
    def generate_system_report(self, report_type: ReportType = ReportType.WEEKLY) -> PerformanceReport:
        """
        生成系统报告
        
        Args:
            report_type: 报告类型
            
        Returns:
            PerformanceReport: 系统报告
        """
        return self.performance_tracker.generate_performance_report(report_type)
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        return {
            'is_running': self._is_running,
            'feedback_summary': self.feedback_collector.get_feedback_summary(),
            'monitoring_summary': self.model_monitor.get_monitoring_summary(),
            'learning_summary': self.adaptive_learner.get_learning_summary(),
            'tracking_summary': self.performance_tracker.get_tracking_summary()
        }
    
    def trigger_adaptive_learning(self, trigger: LearningTrigger = LearningTrigger.MANUAL_TRIGGER):
        """手动触发自适应学习"""
        self.adaptive_learner._execute_adaptive_learning(trigger)


# 全局系统实例
_global_continuous_learning_system = None


def get_continuous_learning_system() -> ContinuousLearningSystem:
    """
    获取全局持续学习系统实例
    
    Returns:
        ContinuousLearningSystem: 系统实例
    """
    global _global_continuous_learning_system
    
    if _global_continuous_learning_system is None:
        _global_continuous_learning_system = ContinuousLearningSystem()
    
    return _global_continuous_learning_system


def initialize_continuous_learning(auto_start: bool = True,
                                 monitor_interval: float = 30.0,
                                 learning_interval: float = 3600.0) -> ContinuousLearningSystem:
    """
    初始化持续学习系统
    
    Args:
        auto_start: 是否自动启动
        monitor_interval: 监控间隔（秒）
        learning_interval: 学习检查间隔（秒）
        
    Returns:
        ContinuousLearningSystem: 初始化的系统实例
    """
    system = get_continuous_learning_system()
    
    if auto_start:
        system.start_system(
            monitor_interval=monitor_interval,
            learning_interval=learning_interval
        )
    
    return system


# 便捷函数
def record_feedback(report_id: str, original_prediction: dict, 
                   is_correct: bool, **kwargs) -> str:
    """便捷的反馈记录函数"""
    system = get_continuous_learning_system()
    return system.record_prediction_feedback(
        report_id=report_id,
        original_prediction=original_prediction,
        is_correct=is_correct,
        **kwargs
    )


def record_metrics(prediction_time: float, accuracy: float = None, 
                  confidence: float = None):
    """便捷的指标记录函数"""
    system = get_continuous_learning_system()
    system.record_performance_metrics(
        prediction_time=prediction_time,
        accuracy=accuracy,
        confidence=confidence
    )


def get_system_dashboard() -> dict:
    """获取系统仪表板数据"""
    system = get_continuous_learning_system()
    
    # 组合所有组件的数据
    dashboard_data = {
        'system_status': system.get_system_status(),
        'performance_dashboard': system.performance_tracker.get_performance_dashboard_data(),
        'current_metrics': system.model_monitor.get_current_metrics(),
        'feedback_stats': system.feedback_collector.calculate_statistics(),
        'learning_summary': system.adaptive_learner.get_learning_summary()
    }
    
    return dashboard_data 
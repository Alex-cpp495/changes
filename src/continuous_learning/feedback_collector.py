"""
用户反馈收集器
收集、存储和管理用户对异常检测结果的反馈，用于持续学习和模型改进
"""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class FeedbackType(Enum):
    """反馈类型枚举"""
    CORRECT_DETECTION = "correct_detection"      # 正确检测
    INCORRECT_DETECTION = "incorrect_detection"  # 错误检测
    MISSING_DETECTION = "missing_detection"      # 漏检
    FALSE_POSITIVE = "false_positive"            # 误检
    SEVERITY_ADJUSTMENT = "severity_adjustment"   # 严重程度调整
    FEATURE_FEEDBACK = "feature_feedback"        # 特征反馈


class FeedbackSource(Enum):
    """反馈来源枚举"""
    EXPERT_REVIEW = "expert_review"              # 专家审查
    USER_INTERFACE = "user_interface"            # 用户界面
    AUTOMATED_CHECK = "automated_check"          # 自动检查
    BATCH_VALIDATION = "batch_validation"        # 批量验证


@dataclass
class UserFeedback:
    """用户反馈数据结构"""
    feedback_id: str
    report_id: str
    original_prediction: Dict[str, Any]
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    
    # 反馈内容
    is_correct: bool
    corrected_label: Optional[str] = None
    confidence_rating: Optional[float] = None  # 1-10
    explanation: Optional[str] = None
    
    # 详细反馈
    feature_feedback: Optional[Dict[str, Any]] = None
    severity_feedback: Optional[str] = None
    additional_notes: Optional[str] = None
    
    # 元数据
    user_id: Optional[str] = None
    user_expertise: Optional[str] = None  # 'expert', 'analyst', 'user'
    timestamp: datetime = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.feedback_id is None:
            self.feedback_id = self._generate_feedback_id()
    
    def _generate_feedback_id(self) -> str:
        """生成反馈ID"""
        content = f"{self.report_id}_{self.timestamp.isoformat()}_{self.feedback_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class FeedbackStatistics:
    """反馈统计信息"""
    total_feedback: int
    correct_detections: int
    incorrect_detections: int
    false_positives: int
    missing_detections: int
    accuracy_rate: float
    user_satisfaction: float
    expert_agreement: float
    recent_trends: Dict[str, Any]


class FeedbackCollector:
    """
    用户反馈收集器
    
    提供全面的反馈管理功能：
    1. 反馈收集 - 多渠道反馈收集和标准化
    2. 数据存储 - SQLite数据库持久化存储
    3. 统计分析 - 反馈趋势和模式分析
    4. 质量评估 - 反馈质量和可信度评估
    5. 数据导出 - 用于模型训练的数据格式化
    6. 自动验证 - 反馈一致性和有效性检查
    
    Args:
        config: 反馈收集配置
        
    Attributes:
        db_path: 数据库路径
        feedback_cache: 内存缓存
        statistics: 统计信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化反馈收集器"""
        self.config = config or {}
        self.file_manager = get_file_manager()
        
        # 数据库设置
        self.data_dir = Path("data/feedback")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "feedback.db"
        
        # 内存缓存
        self.feedback_cache: Dict[str, UserFeedback] = {}
        self.cache_lock = threading.RLock()
        
        # 统计信息
        self.statistics: Optional[FeedbackStatistics] = None
        
        # 初始化数据库
        self._initialize_database()
        
        # 加载最近的反馈到缓存
        self._load_recent_feedback()
        
        logger.info("用户反馈收集器初始化完成")
    
    def _initialize_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建反馈表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        report_id TEXT NOT NULL,
                        original_prediction TEXT NOT NULL,
                        feedback_type TEXT NOT NULL,
                        feedback_source TEXT NOT NULL,
                        is_correct BOOLEAN NOT NULL,
                        corrected_label TEXT,
                        confidence_rating REAL,
                        explanation TEXT,
                        feature_feedback TEXT,
                        severity_feedback TEXT,
                        additional_notes TEXT,
                        user_id TEXT,
                        user_expertise TEXT,
                        timestamp TEXT NOT NULL,
                        processing_time REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_report_id ON feedback(report_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON feedback(user_id)')
                
                conn.commit()
                
            logger.info("反馈数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _load_recent_feedback(self):
        """加载最近的反馈到缓存"""
        try:
            # 加载最近7天的反馈
            cutoff_date = datetime.now() - timedelta(days=7)
            recent_feedback = self.get_feedback_by_date_range(
                start_date=cutoff_date,
                end_date=datetime.now()
            )
            
            with self.cache_lock:
                for feedback in recent_feedback:
                    self.feedback_cache[feedback.feedback_id] = feedback
            
            logger.info(f"加载了 {len(recent_feedback)} 条最近反馈到缓存")
            
        except Exception as e:
            logger.error(f"加载最近反馈失败: {e}")
    
    def collect_feedback(self, report_id: str,
                        original_prediction: Dict[str, Any],
                        feedback_type: FeedbackType,
                        is_correct: bool,
                        feedback_source: FeedbackSource = FeedbackSource.USER_INTERFACE,
                        corrected_label: Optional[str] = None,
                        confidence_rating: Optional[float] = None,
                        explanation: Optional[str] = None,
                        user_id: Optional[str] = None,
                        user_expertise: Optional[str] = None,
                        **kwargs) -> str:
        """
        收集用户反馈
        
        Args:
            report_id: 报告ID
            original_prediction: 原始预测结果
            feedback_type: 反馈类型
            is_correct: 是否正确
            feedback_source: 反馈来源
            corrected_label: 修正标签
            confidence_rating: 置信度评分
            explanation: 解释说明
            user_id: 用户ID
            user_expertise: 用户专业程度
            **kwargs: 其他参数
            
        Returns:
            str: 反馈ID
        """
        try:
            # 创建反馈对象
            feedback = UserFeedback(
                feedback_id=None,  # 自动生成
                report_id=report_id,
                original_prediction=original_prediction,
                feedback_type=feedback_type,
                feedback_source=feedback_source,
                is_correct=is_correct,
                corrected_label=corrected_label,
                confidence_rating=confidence_rating,
                explanation=explanation,
                feature_feedback=kwargs.get('feature_feedback'),
                severity_feedback=kwargs.get('severity_feedback'),
                additional_notes=kwargs.get('additional_notes'),
                user_id=user_id,
                user_expertise=user_expertise,
                processing_time=kwargs.get('processing_time')
            )
            
            # 验证反馈
            if not self._validate_feedback(feedback):
                raise ValueError("反馈数据验证失败")
            
            # 保存到数据库
            self._save_feedback_to_db(feedback)
            
            # 添加到缓存
            with self.cache_lock:
                self.feedback_cache[feedback.feedback_id] = feedback
            
            # 更新统计信息
            self._update_statistics()
            
            logger.info(f"收集反馈成功: {feedback.feedback_id}")
            
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"收集反馈失败: {e}")
            raise
    
    def _validate_feedback(self, feedback: UserFeedback) -> bool:
        """验证反馈数据"""
        try:
            # 基本字段验证
            if not feedback.report_id:
                logger.error("报告ID不能为空")
                return False
            
            if not feedback.original_prediction:
                logger.error("原始预测不能为空")
                return False
            
            # 置信度范围验证
            if feedback.confidence_rating is not None:
                if not (1.0 <= feedback.confidence_rating <= 10.0):
                    logger.error("置信度评分必须在1-10之间")
                    return False
            
            # 反馈类型一致性验证
            if feedback.feedback_type == FeedbackType.INCORRECT_DETECTION and feedback.is_correct:
                logger.error("反馈类型与正确性标志不一致")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"反馈验证失败: {e}")
            return False
    
    def _save_feedback_to_db(self, feedback: UserFeedback):
        """保存反馈到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO feedback (
                        feedback_id, report_id, original_prediction, feedback_type,
                        feedback_source, is_correct, corrected_label, confidence_rating,
                        explanation, feature_feedback, severity_feedback, additional_notes,
                        user_id, user_expertise, timestamp, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback.feedback_id,
                    feedback.report_id,
                    json.dumps(feedback.original_prediction),
                    feedback.feedback_type.value,
                    feedback.feedback_source.value,
                    feedback.is_correct,
                    feedback.corrected_label,
                    feedback.confidence_rating,
                    feedback.explanation,
                    json.dumps(feedback.feature_feedback) if feedback.feature_feedback else None,
                    feedback.severity_feedback,
                    feedback.additional_notes,
                    feedback.user_id,
                    feedback.user_expertise,
                    feedback.timestamp.isoformat(),
                    feedback.processing_time
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存反馈到数据库失败: {e}")
            raise
    
    def get_feedback_by_id(self, feedback_id: str) -> Optional[UserFeedback]:
        """根据ID获取反馈"""
        try:
            # 先查缓存
            with self.cache_lock:
                if feedback_id in self.feedback_cache:
                    return self.feedback_cache[feedback_id]
            
            # 查数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM feedback WHERE feedback_id = ?', (feedback_id,))
                row = cursor.fetchone()
                
                if row:
                    feedback = self._row_to_feedback(row)
                    # 加入缓存
                    with self.cache_lock:
                        self.feedback_cache[feedback_id] = feedback
                    return feedback
            
            return None
            
        except Exception as e:
            logger.error(f"获取反馈失败: {e}")
            return None
    
    def get_feedback_by_report(self, report_id: str) -> List[UserFeedback]:
        """获取报告的所有反馈"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM feedback WHERE report_id = ? ORDER BY timestamp DESC', (report_id,))
                rows = cursor.fetchall()
                
                return [self._row_to_feedback(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取报告反馈失败: {e}")
            return []
    
    def get_feedback_by_date_range(self, start_date: datetime, end_date: datetime) -> List[UserFeedback]:
        """获取日期范围内的反馈"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM feedback 
                    WHERE timestamp BETWEEN ? AND ? 
                    ORDER BY timestamp DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                rows = cursor.fetchall()
                
                return [self._row_to_feedback(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取日期范围反馈失败: {e}")
            return []
    
    def _row_to_feedback(self, row: Tuple) -> UserFeedback:
        """将数据库行转换为反馈对象"""
        return UserFeedback(
            feedback_id=row[0],
            report_id=row[1],
            original_prediction=json.loads(row[2]),
            feedback_type=FeedbackType(row[3]),
            feedback_source=FeedbackSource(row[4]),
            is_correct=bool(row[5]),
            corrected_label=row[6],
            confidence_rating=row[7],
            explanation=row[8],
            feature_feedback=json.loads(row[9]) if row[9] else None,
            severity_feedback=row[10],
            additional_notes=row[11],
            user_id=row[12],
            user_expertise=row[13],
            timestamp=datetime.fromisoformat(row[14]),
            processing_time=row[15]
        )
    
    def calculate_statistics(self) -> FeedbackStatistics:
        """计算反馈统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总反馈数
                cursor.execute('SELECT COUNT(*) FROM feedback')
                total_feedback = cursor.fetchone()[0]
                
                # 各类型反馈统计
                cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type')
                type_counts = dict(cursor.fetchall())
                
                # 正确性统计
                cursor.execute('SELECT is_correct, COUNT(*) FROM feedback GROUP BY is_correct')
                correctness = dict(cursor.fetchall())
                
                # 计算准确率
                correct_count = correctness.get(True, 0)
                accuracy_rate = correct_count / total_feedback if total_feedback > 0 else 0.0
                
                # 用户满意度（基于置信度评分）
                cursor.execute('SELECT AVG(confidence_rating) FROM feedback WHERE confidence_rating IS NOT NULL')
                avg_confidence = cursor.fetchone()[0] or 0.0
                user_satisfaction = avg_confidence / 10.0  # 转换为0-1范围
                
                # 专家一致性
                cursor.execute('''
                    SELECT COUNT(*) FROM feedback 
                    WHERE user_expertise = "expert" AND is_correct = TRUE
                ''')
                expert_correct = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE user_expertise = "expert"')
                expert_total = cursor.fetchone()[0]
                
                expert_agreement = expert_correct / expert_total if expert_total > 0 else 0.0
                
                # 近期趋势（最近30天）
                cutoff_date = datetime.now() - timedelta(days=30)
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM feedback 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''', (cutoff_date.isoformat(),))
                
                recent_trends = {
                    'daily_counts': dict(cursor.fetchall()),
                    'total_recent': sum(dict(cursor.fetchall()).values())
                }
                
                statistics = FeedbackStatistics(
                    total_feedback=total_feedback,
                    correct_detections=type_counts.get(FeedbackType.CORRECT_DETECTION.value, 0),
                    incorrect_detections=type_counts.get(FeedbackType.INCORRECT_DETECTION.value, 0),
                    false_positives=type_counts.get(FeedbackType.FALSE_POSITIVE.value, 0),
                    missing_detections=type_counts.get(FeedbackType.MISSING_DETECTION.value, 0),
                    accuracy_rate=accuracy_rate,
                    user_satisfaction=user_satisfaction,
                    expert_agreement=expert_agreement,
                    recent_trends=recent_trends
                )
                
                self.statistics = statistics
                return statistics
                
        except Exception as e:
            logger.error(f"计算统计信息失败: {e}")
            return FeedbackStatistics(
                total_feedback=0,
                correct_detections=0,
                incorrect_detections=0,
                false_positives=0,
                missing_detections=0,
                accuracy_rate=0.0,
                user_satisfaction=0.0,
                expert_agreement=0.0,
                recent_trends={}
            )
    
    def _update_statistics(self):
        """更新统计信息"""
        try:
            self.statistics = self.calculate_statistics()
        except Exception as e:
            logger.error(f"更新统计信息失败: {e}")
    
    def export_training_data(self, output_path: Optional[str] = None,
                           min_confidence: float = 5.0,
                           expert_only: bool = False) -> str:
        """
        导出用于训练的反馈数据
        
        Args:
            output_path: 输出路径
            min_confidence: 最小置信度阈值
            expert_only: 是否只导出专家反馈
            
        Returns:
            str: 导出文件路径
        """
        try:
            # 构建查询条件
            conditions = ['1=1']
            params = []
            
            if min_confidence > 0:
                conditions.append('(confidence_rating IS NULL OR confidence_rating >= ?)')
                params.append(min_confidence)
            
            if expert_only:
                conditions.append('user_expertise = ?')
                params.append('expert')
            
            query = f'''
                SELECT * FROM feedback 
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
            '''
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            # 转换为训练数据格式
            training_data = []
            for row in rows:
                feedback = self._row_to_feedback(row)
                
                training_item = {
                    'report_id': feedback.report_id,
                    'features': feedback.original_prediction,
                    'label': feedback.corrected_label if feedback.corrected_label else feedback.is_correct,
                    'feedback_type': feedback.feedback_type.value,
                    'confidence': feedback.confidence_rating,
                    'explanation': feedback.explanation,
                    'user_expertise': feedback.user_expertise,
                    'timestamp': feedback.timestamp.isoformat()
                }
                
                training_data.append(training_item)
            
            # 保存到文件
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.data_dir / f"training_data_{timestamp}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"导出训练数据: {len(training_data)}条记录 -> {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"导出训练数据失败: {e}")
            raise
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        try:
            stats = self.statistics or self.calculate_statistics()
            
            return {
                'database_path': str(self.db_path),
                'cache_size': len(self.feedback_cache),
                'statistics': asdict(stats),
                'recent_feedback_count': len([
                    f for f in self.feedback_cache.values()
                    if f.timestamp > datetime.now() - timedelta(days=7)
                ]),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取反馈摘要失败: {e}")
            return {'error': str(e)}


# 全局反馈收集器实例
_global_feedback_collector = None


def get_feedback_collector() -> FeedbackCollector:
    """
    获取全局反馈收集器实例
    
    Returns:
        FeedbackCollector: 收集器实例
    """
    global _global_feedback_collector
    
    if _global_feedback_collector is None:
        _global_feedback_collector = FeedbackCollector()
    
    return _global_feedback_collector


if __name__ == "__main__":
    # 使用示例
    print("用户反馈收集器测试:")
    
    # 创建收集器
    collector = FeedbackCollector()
    
    # 模拟收集反馈
    feedback_id = collector.collect_feedback(
        report_id="report_123",
        original_prediction={
            'overall_anomaly_score': 0.8,
            'overall_anomaly_level': 'HIGH'
        },
        feedback_type=FeedbackType.CORRECT_DETECTION,
        is_correct=True,
        confidence_rating=8.0,
        explanation="检测结果准确，报告确实存在异常",
        user_id="analyst_001",
        user_expertise="expert"
    )
    
    print(f"收集反馈成功: {feedback_id}")
    
    # 获取反馈
    feedback = collector.get_feedback_by_id(feedback_id)
    if feedback:
        print(f"反馈详情: {feedback.explanation}")
    
    # 计算统计信息
    stats = collector.calculate_statistics()
    print(f"统计信息: 总反馈{stats.total_feedback}条, 准确率{stats.accuracy_rate:.2%}")
    
    # 获取摘要
    summary = collector.get_feedback_summary()
    print(f"摘要信息: {summary}") 
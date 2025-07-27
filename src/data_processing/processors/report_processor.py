"""
研报数据处理器
处理用户提供的研报数据，支持批量导入和分析
"""

import json
import pandas as pd
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import hashlib

from ...utils.logger import get_logger
from ...anomaly_detection import get_ensemble_detector
from ...continuous_learning import get_continuous_learning_system

logger = get_logger(__name__)


class ReportProcessor:
    """
    研报数据处理器
    
    功能：
    1. 批量导入研报数据
    2. 数据格式标准化
    3. 批量异常检测
    4. 结果统计分析
    5. 数据质量评估
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化处理器"""
        self.config = config or {}
        self.detector = get_ensemble_detector()
        self.continuous_learning = get_continuous_learning_system()
        
        # 处理统计
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'anomalous': 0,
            'normal': 0,
            'processing_time': 0.0,
            'start_time': None,
            'end_time': None,
            'success_rate': 0.0,
            'avg_processing_time': 0.0,
            'total_reports': 0
        }
        
        # 支持的数据格式
        self.supported_formats = {
            'json': self._process_json_data,
            'csv': self._process_csv_data,
            'excel': self._process_excel_data,
            'txt': self._process_txt_data
        }
        
        logger.info("研报数据处理器初始化完成")
    
    async def process_batch_data(self, data_source: Union[str, List[Dict], pd.DataFrame], 
                                data_format: str = 'auto') -> Dict[str, Any]:
        """
        批量处理研报数据
        
        Args:
            data_source: 数据源（文件路径、数据列表或DataFrame）
            data_format: 数据格式（json, csv, excel, txt, auto）
            
        Returns:
            处理结果统计
        """
        try:
            self._reset_stats()
            self.stats['start_time'] = datetime.now()
            
            logger.info(f"开始批量处理数据，格式: {data_format}")
            
            # 载入数据
            reports_data = await self._load_data(data_source, data_format)
            
            if not reports_data:
                raise ValueError("无法载入数据或数据为空")
            
            self.stats['total_processed'] = len(reports_data)
            logger.info(f"载入 {len(reports_data)} 条研报数据")
            
            # 批量处理
            results = await self._process_reports_batch(reports_data)
            
            # 统计分析
            self._calculate_final_stats(results)
            
            # 生成处理报告
            processing_report = self._generate_processing_report(results)
            
            logger.info(f"批量处理完成，成功: {self.stats['successful']}, 失败: {self.stats['failed']}")
            
            return processing_report
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            raise
    
    async def _load_data(self, data_source: Union[str, List[Dict], pd.DataFrame], 
                        data_format: str) -> List[Dict[str, Any]]:
        """载入数据"""
        try:
            if isinstance(data_source, list):
                return data_source
            
            if isinstance(data_source, pd.DataFrame):
                return data_source.to_dict('records')
            
            if isinstance(data_source, str):
                file_path = Path(data_source)
                
                if not file_path.exists():
                    raise FileNotFoundError(f"文件不存在: {data_source}")
                
                # 自动检测格式
                if data_format == 'auto':
                    data_format = file_path.suffix.lower().lstrip('.')
                
                if data_format not in self.supported_formats:
                    raise ValueError(f"不支持的数据格式: {data_format}")
                
                return await self.supported_formats[data_format](file_path)
            
            raise ValueError("不支持的数据源类型")
            
        except Exception as e:
            logger.error(f"数据载入失败: {e}")
            raise
    
    async def _process_json_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理JSON数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON数据格式错误")
    
    async def _process_csv_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理CSV数据"""
        df = pd.read_csv(file_path, encoding='utf-8')
        return df.to_dict('records')
    
    async def _process_excel_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理Excel数据"""
        df = pd.read_excel(file_path)
        return df.to_dict('records')
    
    async def _process_txt_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理文本数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 假设每个段落是一篇报告
        reports = []
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                reports.append({
                    'report_id': f"txt_report_{i+1}",
                    'title': f"文本报告 {i+1}",
                    'content': paragraph.strip(),
                    'source': str(file_path)
                })
        
        return reports
    
    async def _process_reports_batch(self, reports_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理研报"""
        results = []
        batch_size = self.config.get('batch_size', 10)
        
        # 分批处理
        for i in range(0, len(reports_data), batch_size):
            batch = reports_data[i:i + batch_size]
            
            logger.info(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 条")
            
            # 并发处理批次
            batch_tasks = [self._process_single_report(report) for report in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 收集结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"处理报告失败: {result}")
                    self.stats['failed'] += 1
                    results.append({
                        'report_id': batch[j].get('report_id', f'unknown_{i+j}'),
                        'status': 'failed',
                        'error': str(result),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    self.stats['successful'] += 1
                    if result.get('anomaly_result', {}).get('is_anomalous', False):
                        self.stats['anomalous'] += 1
                    else:
                        self.stats['normal'] += 1
                    results.append(result)
            
            # 进度提示
            progress = (i + len(batch)) / len(reports_data) * 100
            logger.info(f"处理进度: {progress:.1f}%")
            
            # 短暂延迟，避免系统过载
            await asyncio.sleep(0.1)
        
        return results
    
    async def _process_single_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个研报"""
        try:
            start_time = time.time()
            
            # 标准化数据格式
            standardized_report = self._standardize_report_data(report_data)
            
            # 执行异常检测
            detection_result = self.detector.detect_anomalies(
                standardized_report
            )
            
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                'report_id': standardized_report['report_id'],
                'title': standardized_report.get('title', ''),
                'status': 'success',
                'anomaly_result': detection_result,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'metadata': standardized_report.get('metadata', {})
            }
            
            # 记录性能指标
            self.continuous_learning.record_performance_metrics(
                prediction_time=processing_time,
                confidence=detection_result.get('confidence', 0.0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"处理单个报告失败: {e}")
            raise
    
    def _standardize_report_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化研报数据格式"""
        # 生成唯一ID
        if 'report_id' not in report_data:
            content_hash = hashlib.md5(
                str(report_data.get('content', '')).encode('utf-8')
            ).hexdigest()[:8]
            report_data['report_id'] = f"report_{content_hash}"
        
        # 必需字段检查
        required_fields = ['content']
        for field in required_fields:
            if field not in report_data or not report_data[field]:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 标准化格式
        standardized = {
            'report_id': report_data['report_id'],
            'title': report_data.get('title', report_data.get('name', '未命名报告')),
            'content': report_data['content'],
            'metadata': {
                'source': report_data.get('source', 'unknown'),
                'company': report_data.get('company', ''),
                'industry': report_data.get('industry', ''),
                'report_date': report_data.get('report_date', ''),
                'analyst': report_data.get('analyst', ''),
                'original_data': report_data  # 保留原始数据
            }
        }
        
        return standardized
    
    def _calculate_final_stats(self, results: List[Dict[str, Any]]):
        """计算最终统计信息"""
        self.stats['end_time'] = datetime.now()
        
        if self.stats['start_time']:
            total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            self.stats['processing_time'] = total_time
        
        # 计算处理速度
        if self.stats['processing_time'] > 0:
            self.stats['reports_per_second'] = self.stats['total_processed'] / self.stats['processing_time']
        
        # 异常检测统计
        if self.stats['successful'] > 0:
            self.stats['anomaly_rate'] = self.stats['anomalous'] / self.stats['successful']
            self.stats['success_rate'] = self.stats['successful'] / self.stats['total_processed']
    
    def _generate_processing_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成处理报告"""
        # 按异常级别统计
        anomaly_levels = {}
        processing_times = []
        
        for result in results:
            if result['status'] == 'success':
                anomaly_result = result.get('anomaly_result', {})
                level = anomaly_result.get('overall_anomaly_level', 'UNKNOWN')
                anomaly_levels[level] = anomaly_levels.get(level, 0) + 1
                
                if 'processing_time' in result:
                    processing_times.append(result['processing_time'])
        
        # 性能统计
        performance_stats = {}
        if processing_times:
            performance_stats = {
                'avg_processing_time': sum(processing_times) / len(processing_times),
                'min_processing_time': min(processing_times),
                'max_processing_time': max(processing_times),
                'total_processing_time': sum(processing_times)
            }
        
        report = {
            'summary': self.stats.copy(),
            'anomaly_distribution': anomaly_levels,
            'performance_stats': performance_stats,
            'detailed_results': results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成处理建议"""
        recommendations = []
        
        # 根据统计数据生成建议
        if self.stats.get('success_rate', 0) < 0.9:
            recommendations.append("建议检查输入数据格式和质量")
        
        if self.stats.get('avg_processing_time', 0) > 10:
            recommendations.append("建议优化处理性能，考虑增加并发数")
        
        if self.stats.get('total_reports', 0) > 100:
            recommendations.append("建议启用批量处理模式以提高效率")
        
        if not recommendations:
            recommendations.append("所有指标正常，系统运行良好")
        
        return recommendations
    
    def _reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'anomalous': 0,
            'normal': 0,
            'processing_time': 0.0,
            'start_time': None,
            'end_time': None,
            'success_rate': 0.0,
            'avg_processing_time': 0.0,
            'total_reports': 0
        }
    
    def export_results(self, results: Dict[str, Any], export_path: str, 
                      export_format: str = 'json') -> str:
        """导出处理结果"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            if export_format == 'json':
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            elif export_format == 'csv':
                # 导出摘要信息
                summary_data = []
                for result in results['detailed_results']:
                    if result['status'] == 'success':
                        anomaly_result = result.get('anomaly_result', {})
                        summary_data.append({
                            'report_id': result['report_id'],
                            'title': result['title'],
                            'is_anomalous': anomaly_result.get('is_anomalous', False),
                            'anomaly_score': anomaly_result.get('overall_anomaly_score', 0),
                            'anomaly_level': anomaly_result.get('overall_anomaly_level', ''),
                            'confidence': anomaly_result.get('confidence', 0),
                            'processing_time': result.get('processing_time', 0),
                            'timestamp': result['timestamp']
                        })
                
                df = pd.DataFrame(summary_data)
                df.to_csv(export_file, index=False, encoding='utf-8-sig')
            
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            logger.info(f"结果已导出到: {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"导出结果失败: {e}")
            raise


def get_report_processor(config: Optional[Dict[str, Any]] = None) -> ReportProcessor:
    """获取研报处理器实例"""
    return ReportProcessor(config) 
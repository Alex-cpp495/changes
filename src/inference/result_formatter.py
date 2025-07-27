"""
结果格式化器
将异常检测结果转换为多种格式的报告，支持不同的输出需求
"""

import json
import csv
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import html
import base64
from io import StringIO, BytesIO

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class OutputFormat(Enum):
    """输出格式枚举"""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    EXCEL = "excel"
    PDF = "pdf"
    TEXT = "text"


class ReportType(Enum):
    """报告类型枚举"""
    SUMMARY = "summary"           # 摘要报告
    DETAILED = "detailed"         # 详细报告
    DASHBOARD = "dashboard"       # 仪表板报告
    ALERT = "alert"              # 告警报告
    COMPARISON = "comparison"     # 对比报告
    TREND = "trend"              # 趋势报告


@dataclass
class FormattingOptions:
    """格式化选项"""
    include_metadata: bool = True
    include_raw_data: bool = False
    include_statistics: bool = True
    include_visualizations: bool = False
    compact_format: bool = False
    localize: bool = True
    timezone: str = "Asia/Shanghai"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    precision: int = 3


class ResultFormatter:
    """
    结果格式化器
    
    提供多样化的结果格式化功能：
    1. 多格式支持 - JSON、HTML、CSV、Excel、PDF等
    2. 报告类型 - 摘要、详细、仪表板、告警等
    3. 模板系统 - 可自定义的报告模板
    4. 本地化支持 - 多语言、时区、格式本地化
    5. 可视化集成 - 图表、统计图表、趋势分析
    6. 导出功能 - 文件保存、批量导出、压缩打包
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 格式化配置
        templates: 报告模板
        formatters: 格式化器映射
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化结果格式化器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.file_manager = get_file_manager()
        
        # 模板存储
        self.templates: Dict[str, str] = {}
        
        # 格式化器映射
        self.formatters: Dict[OutputFormat, callable] = {}
        
        # 统计信息
        self.stats = {
            'total_formats': 0,
            'successful_formats': 0,
            'failed_formats': 0,
            'format_times': {},
            'output_sizes': {}
        }
        
        # 注册默认格式化器
        self._register_default_formatters()
        
        # 加载模板
        self._load_templates()
        
        logger.info("结果格式化器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('result_formatting', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'output': {
                'default_format': 'json',
                'include_metadata': True,
                'include_statistics': True,
                'date_format': '%Y-%m-%d %H:%M:%S',
                'timezone': 'Asia/Shanghai',
                'precision': 3
            },
            'templates': {
                'template_directory': 'templates',
                'auto_load': True,
                'custom_templates': {}
            },
            'localization': {
                'default_language': 'zh_CN',
                'supported_languages': ['zh_CN', 'en_US'],
                'number_format': 'chinese'
            },
            'visualization': {
                'enable_charts': False,
                'chart_library': 'matplotlib',
                'chart_styles': ['default', 'business', 'scientific']
            },
            'export': {
                'output_directory': 'data/reports',
                'auto_timestamp': True,
                'compress_large_files': True,
                'size_threshold_mb': 10
            }
        }
    
    def _register_default_formatters(self):
        """注册默认格式化器"""
        self.formatters[OutputFormat.JSON] = self._format_json
        self.formatters[OutputFormat.HTML] = self._format_html
        self.formatters[OutputFormat.CSV] = self._format_csv
        self.formatters[OutputFormat.XML] = self._format_xml
        self.formatters[OutputFormat.MARKDOWN] = self._format_markdown
        self.formatters[OutputFormat.TEXT] = self._format_text
    
    def _load_templates(self):
        """加载报告模板"""
        try:
            template_dir = Path(self.config.get('templates', {}).get('template_directory', 'templates'))
            
            # 内置模板
            self.templates.update(self._get_builtin_templates())
            
            # 从文件加载自定义模板
            if template_dir.exists():
                for template_file in template_dir.glob('*.html'):
                    template_name = template_file.stem
                    template_content = template_file.read_text(encoding='utf-8')
                    self.templates[template_name] = template_content
                    logger.info(f"加载模板: {template_name}")
            
        except Exception as e:
            logger.warning(f"模板加载失败: {e}")
    
    def _get_builtin_templates(self) -> Dict[str, str]:
        """获取内置模板"""
        return {
            'summary_html': '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>研报异常检测摘要报告</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #e74c3c; padding-bottom: 20px; margin-bottom: 30px; }
        .title { color: #2c3e50; font-size: 28px; margin-bottom: 10px; }
        .subtitle { color: #7f8c8d; font-size: 16px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 32px; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 14px; opacity: 0.9; }
        .anomaly-level { padding: 10px; border-radius: 6px; margin: 10px 0; font-weight: bold; text-align: center; }
        .level-critical { background-color: #e74c3c; color: white; }
        .level-high { background-color: #f39c12; color: white; }
        .level-medium { background-color: #f1c40f; color: #2c3e50; }
        .level-low { background-color: #27ae60; color: white; }
        .level-normal { background-color: #95a5a6; color: white; }
        .details-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .details-table th, .details-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .details-table th { background-color: #34495e; color: white; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; text-align: center; color: #7f8c8d; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">研报异常检测摘要报告</h1>
            <p class="subtitle">生成时间: {timestamp}</p>
        </div>
        
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{total_reports}</div>
                <div class="metric-label">总报告数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{anomaly_count}</div>
                <div class="metric-label">异常报告</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{anomaly_rate}%</div>
                <div class="metric-label">异常率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_score}</div>
                <div class="metric-label">平均异常分数</div>
            </div>
        </div>
        
        <div class="anomaly-distribution">
            <h3>异常等级分布</h3>
            {anomaly_levels_html}
        </div>
        
        <div class="recent-anomalies">
            <h3>近期重要异常</h3>
            <table class="details-table">
                <thead>
                    <tr>
                        <th>报告标题</th>
                        <th>异常等级</th>
                        <th>异常分数</th>
                        <th>主要异常类型</th>
                        <th>检测时间</th>
                    </tr>
                </thead>
                <tbody>
                    {recent_anomalies_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>本报告由东吴证券研报异常检测系统自动生成</p>
        </div>
    </div>
</body>
</html>
            ''',
            'detailed_html': '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>详细异常检测报告</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; }
        .report-card { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .report-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .report-title { font-size: 24px; color: #2c3e50; margin: 0; }
        .report-meta { color: #7f8c8d; font-size: 14px; }
        .anomaly-badge { padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .detection-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .detection-section { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }
        .section-title { font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .issue-list { list-style: none; padding: 0; }
        .issue-item { background: white; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 3px solid #e74c3c; }
        .score-display { font-size: 24px; font-weight: bold; text-align: center; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        {report_cards_html}
    </div>
</body>
</html>
            '''
        }
    
    def format_result(self, result: Dict[str, Any], 
                     output_format: OutputFormat,
                     report_type: ReportType = ReportType.SUMMARY,
                     options: Optional[FormattingOptions] = None) -> str:
        """
        格式化单个结果
        
        Args:
            result: 检测结果
            output_format: 输出格式
            report_type: 报告类型
            options: 格式化选项
            
        Returns:
            str: 格式化后的结果
        """
        start_time = datetime.now()
        
        try:
            self.stats['total_formats'] += 1
            
            # 使用默认选项
            if options is None:
                options = FormattingOptions()
            
            # 预处理结果数据
            processed_result = self._preprocess_result(result, options)
            
            # 选择格式化器
            if output_format not in self.formatters:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            formatter_func = self.formatters[output_format]
            
            # 执行格式化
            formatted_result = formatter_func(processed_result, report_type, options)
            
            # 记录统计信息
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_formats'] += 1
            self.stats['format_times'][output_format.value] = processing_time
            self.stats['output_sizes'][output_format.value] = len(formatted_result)
            
            return formatted_result
            
        except Exception as e:
            self.stats['failed_formats'] += 1
            logger.error(f"结果格式化失败: {e}")
            raise Exception(f"格式化失败: {e}")
    
    def format_batch_results(self, results: List[Dict[str, Any]], 
                           output_format: OutputFormat,
                           report_type: ReportType = ReportType.SUMMARY,
                           options: Optional[FormattingOptions] = None) -> str:
        """
        格式化批量结果
        
        Args:
            results: 检测结果列表
            output_format: 输出格式
            report_type: 报告类型
            options: 格式化选项
            
        Returns:
            str: 格式化后的结果
        """
        try:
            # 汇总统计信息
            summary_stats = self._calculate_batch_statistics(results)
            
            # 根据报告类型处理
            if report_type == ReportType.SUMMARY:
                return self._format_summary_report(summary_stats, results, output_format, options)
            elif report_type == ReportType.DETAILED:
                return self._format_detailed_report(results, output_format, options)
            elif report_type == ReportType.DASHBOARD:
                return self._format_dashboard_report(summary_stats, results, output_format, options)
            else:
                # 默认处理：逐个格式化并合并
                formatted_results = []
                for result in results:
                    formatted_result = self.format_result(result, output_format, report_type, options)
                    formatted_results.append(formatted_result)
                
                return self._merge_formatted_results(formatted_results, output_format)
                
        except Exception as e:
            logger.error(f"批量结果格式化失败: {e}")
            raise Exception(f"批量格式化失败: {e}")
    
    def _preprocess_result(self, result: Dict[str, Any], options: FormattingOptions) -> Dict[str, Any]:
        """预处理结果数据"""
        processed = result.copy()
        
        # 时间本地化
        if options.localize:
            processed = self._localize_timestamps(processed, options.timezone, options.date_format)
        
        # 数值精度处理
        processed = self._round_numbers(processed, options.precision)
        
        # 添加格式化元数据
        if options.include_metadata:
            processed['formatting_metadata'] = {
                'formatted_at': datetime.now().strftime(options.date_format),
                'timezone': options.timezone,
                'precision': options.precision,
                'formatter_version': '1.0'
            }
        
        return processed
    
    def _localize_timestamps(self, data: Dict[str, Any], timezone: str, date_format: str) -> Dict[str, Any]:
        """本地化时间戳"""
        # 简化实现，实际需要处理时区转换
        def process_value(value):
            if isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            elif isinstance(value, str) and ('timestamp' in str(value).lower() or 'time' in str(value).lower()):
                try:
                    # 尝试解析ISO格式时间戳
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return dt.strftime(date_format)
                except:
                    return value
            else:
                return value
        
        return process_value(data)
    
    def _round_numbers(self, data: Dict[str, Any], precision: int) -> Dict[str, Any]:
        """数值四舍五入"""
        def process_value(value):
            if isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            elif isinstance(value, float):
                return round(value, precision)
            else:
                return value
        
        return process_value(data)
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算批量统计信息"""
        total_reports = len(results)
        anomaly_count = 0
        anomaly_scores = []
        anomaly_levels = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NORMAL': 0}
        
        for result in results:
            # 提取异常信息
            if 'overall_anomaly_level' in result:
                level = result['overall_anomaly_level']
                if level != 'NORMAL':
                    anomaly_count += 1
                anomaly_levels[level] = anomaly_levels.get(level, 0) + 1
            
            if 'overall_anomaly_score' in result:
                anomaly_scores.append(result['overall_anomaly_score'])
        
        return {
            'total_reports': total_reports,
            'anomaly_count': anomaly_count,
            'anomaly_rate': round((anomaly_count / total_reports * 100) if total_reports > 0 else 0, 2),
            'avg_score': round(sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0, 3),
            'anomaly_levels': anomaly_levels,
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # 格式化器实现
    def _format_json(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """JSON格式化"""
        if options.compact_format:
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _format_html(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """HTML格式化"""
        if report_type == ReportType.SUMMARY:
            return self._generate_html_summary(data, options)
        elif report_type == ReportType.DETAILED:
            return self._generate_html_detailed(data, options)
        else:
            return self._generate_html_basic(data, options)
    
    def _format_csv(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """CSV格式化"""
        output = StringIO()
        
        # 扁平化数据
        flattened_data = self._flatten_dict(data)
        
        if isinstance(data, list):
            # 多行数据
            if data:
                fieldnames = flattened_data[0].keys() if flattened_data else []
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
        else:
            # 单行数据
            fieldnames = flattened_data.keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(flattened_data)
        
        return output.getvalue()
    
    def _format_xml(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """XML格式化"""
        def dict_to_xml(d, root_tag='result'):
            xml_lines = [f'<?xml version="1.0" encoding="UTF-8"?>']
            xml_lines.append(f'<{root_tag}>')
            
            def process_item(key, value, indent=1):
                spaces = '  ' * indent
                if isinstance(value, dict):
                    xml_lines.append(f'{spaces}<{key}>')
                    for k, v in value.items():
                        process_item(k, v, indent + 1)
                    xml_lines.append(f'{spaces}</{key}>')
                elif isinstance(value, list):
                    xml_lines.append(f'{spaces}<{key}>')
                    for i, item in enumerate(value):
                        process_item(f'item_{i}', item, indent + 1)
                    xml_lines.append(f'{spaces}</{key}>')
                else:
                    escaped_value = html.escape(str(value))
                    xml_lines.append(f'{spaces}<{key}>{escaped_value}</{key}>')
            
            for key, value in d.items():
                process_item(key, value)
            
            xml_lines.append(f'</{root_tag}>')
            return '\n'.join(xml_lines)
        
        return dict_to_xml(data)
    
    def _format_markdown(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """Markdown格式化"""
        md_lines = []
        
        # 标题
        md_lines.append('# 异常检测结果报告\n')
        
        # 基本信息
        if 'title' in data:
            md_lines.append(f"## 报告: {data['title']}\n")
        
        # 异常概况
        if 'overall_anomaly_level' in data:
            level = data['overall_anomaly_level']
            score = data.get('overall_anomaly_score', 'N/A')
            md_lines.append(f"**异常等级**: {level}  ")
            md_lines.append(f"**异常分数**: {score}\n")
        
        # 详细检测结果
        md_lines.append('## 检测详情\n')
        
        if 'detection_results' in data:
            for detector_name, result in data['detection_results'].items():
                md_lines.append(f"### {detector_name}\n")
                
                if 'anomaly_score' in result:
                    md_lines.append(f"- **分数**: {result['anomaly_score']}")
                if 'anomaly_level' in result:
                    md_lines.append(f"- **等级**: {result['anomaly_level']}")
                if 'issues' in result and result['issues']:
                    md_lines.append("- **发现的问题**:")
                    for issue in result['issues']:
                        issue_desc = issue.get('message', str(issue))
                        md_lines.append(f"  - {issue_desc}")
                
                md_lines.append("")
        
        return '\n'.join(md_lines)
    
    def _format_text(self, data: Dict[str, Any], report_type: ReportType, options: FormattingOptions) -> str:
        """纯文本格式化"""
        lines = []
        
        lines.append("=" * 60)
        lines.append("异常检测结果报告")
        lines.append("=" * 60)
        
        # 基本信息
        if 'title' in data:
            lines.append(f"报告标题: {data['title']}")
        
        if 'overall_anomaly_level' in data:
            lines.append(f"异常等级: {data['overall_anomaly_level']}")
        
        if 'overall_anomaly_score' in data:
            lines.append(f"异常分数: {data['overall_anomaly_score']}")
        
        lines.append("")
        
        # 详细结果
        if 'detection_results' in data:
            lines.append("检测详情:")
            lines.append("-" * 40)
            
            for detector_name, result in data['detection_results'].items():
                lines.append(f"\n{detector_name}:")
                lines.append(f"  分数: {result.get('anomaly_score', 'N/A')}")
                lines.append(f"  等级: {result.get('anomaly_level', 'N/A')}")
                
                if 'issues' in result and result['issues']:
                    lines.append("  问题:")
                    for issue in result['issues']:
                        issue_desc = issue.get('message', str(issue))
                        lines.append(f"    - {issue_desc}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def _generate_html_summary(self, data: Dict[str, Any], options: FormattingOptions) -> str:
        """生成HTML摘要报告"""
        template = self.templates.get('summary_html', '')
        
        # 准备模板变量
        template_vars = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_reports': data.get('total_reports', 0),
            'anomaly_count': data.get('anomaly_count', 0),
            'anomaly_rate': data.get('anomaly_rate', 0),
            'avg_score': data.get('avg_score', 0),
            'anomaly_levels_html': self._generate_anomaly_levels_html(data.get('anomaly_levels', {})),
            'recent_anomalies_html': self._generate_recent_anomalies_html(data.get('recent_anomalies', []))
        }
        
        # 填充模板
        return template.format(**template_vars)
    
    def _generate_anomaly_levels_html(self, levels: Dict[str, int]) -> str:
        """生成异常等级分布HTML"""
        html_parts = []
        level_classes = {
            'CRITICAL': 'level-critical',
            'HIGH': 'level-high', 
            'MEDIUM': 'level-medium',
            'LOW': 'level-low',
            'NORMAL': 'level-normal'
        }
        
        for level, count in levels.items():
            css_class = level_classes.get(level, 'level-normal')
            html_parts.append(f'<div class="anomaly-level {css_class}">{level}: {count}</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_recent_anomalies_html(self, anomalies: List[Dict[str, Any]]) -> str:
        """生成近期异常HTML"""
        html_parts = []
        
        for anomaly in anomalies[:10]:  # 只显示前10个
            title = anomaly.get('title', 'N/A')
            level = anomaly.get('level', 'UNKNOWN')
            score = anomaly.get('score', 0)
            types = ', '.join(anomaly.get('types', []))
            time = anomaly.get('time', 'N/A')
            
            html_parts.append(f'''
                <tr>
                    <td>{html.escape(title)}</td>
                    <td><span class="anomaly-badge level-{level.lower()}">{level}</span></td>
                    <td>{score}</td>
                    <td>{html.escape(types)}</td>
                    <td>{time}</td>
                </tr>
            ''')
        
        return '\n'.join(html_parts)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """扁平化字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _format_summary_report(self, stats: Dict[str, Any], results: List[Dict[str, Any]], 
                             output_format: OutputFormat, options: FormattingOptions) -> str:
        """格式化摘要报告"""
        summary_data = {
            'report_type': 'summary',
            'statistics': stats,
            'generation_time': datetime.now().isoformat(),
            'total_results': len(results)
        }
        
        if output_format == OutputFormat.HTML:
            return self._generate_html_summary(summary_data, options)
        else:
            return self.formatters[output_format](summary_data, ReportType.SUMMARY, options)
    
    def _format_detailed_report(self, results: List[Dict[str, Any]], 
                              output_format: OutputFormat, options: FormattingOptions) -> str:
        """格式化详细报告"""
        detailed_data = {
            'report_type': 'detailed',
            'results': results,
            'generation_time': datetime.now().isoformat(),
            'total_results': len(results)
        }
        
        return self.formatters[output_format](detailed_data, ReportType.DETAILED, options)
    
    def _format_dashboard_report(self, stats: Dict[str, Any], results: List[Dict[str, Any]], 
                               output_format: OutputFormat, options: FormattingOptions) -> str:
        """格式化仪表板报告"""
        dashboard_data = {
            'report_type': 'dashboard',
            'statistics': stats,
            'recent_results': results[-20:] if len(results) > 20 else results,  # 最近20个结果
            'generation_time': datetime.now().isoformat()
        }
        
        return self.formatters[output_format](dashboard_data, ReportType.DASHBOARD, options)
    
    def _merge_formatted_results(self, formatted_results: List[str], output_format: OutputFormat) -> str:
        """合并格式化结果"""
        if output_format == OutputFormat.JSON:
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        elif output_format == OutputFormat.HTML:
            return '\n'.join(formatted_results)
        elif output_format == OutputFormat.CSV:
            return '\n'.join(formatted_results)
        else:
            return '\n\n'.join(formatted_results)
    
    def save_formatted_result(self, formatted_result: str, 
                            output_format: OutputFormat,
                            filename: Optional[str] = None) -> str:
        """
        保存格式化结果到文件
        
        Args:
            formatted_result: 格式化后的结果
            output_format: 输出格式
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 创建输出目录
            output_dir = Path(self.config.get('export', {}).get('output_directory', 'data/reports'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"anomaly_report_{timestamp}.{output_format.value}"
            
            file_path = output_dir / filename
            
            # 保存文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_result)
            
            logger.info(f"格式化结果已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存格式化结果失败: {e}")
            raise Exception(f"保存失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取格式化统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'formatting_statistics': self.stats.copy(),
            'supported_formats': [fmt.value for fmt in OutputFormat],
            'available_templates': list(self.templates.keys()),
            'config': self.config
        }


# 全局结果格式化器实例
_global_result_formatter = None


def get_result_formatter() -> ResultFormatter:
    """
    获取全局结果格式化器实例
    
    Returns:
        ResultFormatter: 格式化器实例
    """
    global _global_result_formatter
    
    if _global_result_formatter is None:
        _global_result_formatter = ResultFormatter()
    
    return _global_result_formatter


if __name__ == "__main__":
    # 使用示例
    formatter = ResultFormatter()
    
    print("结果格式化器测试:")
    
    # 测试数据
    test_result = {
        'title': '平安银行(000001.SZ)研报分析',
        'overall_anomaly_level': 'HIGH',
        'overall_anomaly_score': 0.75,
        'detection_results': {
            'statistical_detector': {
                'anomaly_score': 0.6,
                'anomaly_level': 'MEDIUM',
                'issues': [
                    {'type': 'length_anomaly', 'message': '文本长度异常'}
                ]
            },
            'semantic_detector': {
                'anomaly_score': 0.8,
                'anomaly_level': 'HIGH',
                'issues': [
                    {'type': 'contradiction', 'message': '逻辑矛盾'}
                ]
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 测试不同格式
    formats_to_test = [OutputFormat.JSON, OutputFormat.MARKDOWN, OutputFormat.TEXT]
    
    for fmt in formats_to_test:
        try:
            result = formatter.format_result(test_result, fmt)
            print(f"\n{fmt.value.upper()}格式化成功，长度: {len(result)}")
            
            if fmt == OutputFormat.TEXT:
                print("预览:")
                print(result[:200] + "..." if len(result) > 200 else result)
                
        except Exception as e:
            print(f"{fmt.value}格式化失败: {e}")
    
    # 获取统计信息
    stats = formatter.get_statistics()
    print(f"\n统计信息: {stats['formatting_statistics']}") 
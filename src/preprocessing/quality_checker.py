"""
数据质量检查器
对研报数据进行全面的质量评估、完整性验证、一致性检查等
"""

import re
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor

logger = get_logger(__name__)


class QualityChecker:
    """
    数据质量检查器
    
    提供全面的研报数据质量评估：
    1. 完整性检查 - 必填字段、数据完整性
    2. 一致性检查 - 内部逻辑一致性、格式一致性  
    3. 准确性检查 - 数据格式、数值合理性
    4. 时效性检查 - 数据新鲜度、时间逻辑
    5. 唯一性检查 - 重复数据识别
    6. 业务规则检查 - 行业特定规则验证
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 质量检查配置
        text_processor: 文本处理器
        quality_rules: 质量规则集合
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化数据质量检查器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        
        # 质量规则和模式
        self.quality_rules = self._compile_quality_rules()
        
        # 统计信息
        self.check_stats = {
            'total_checked': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warning_count': 0,
            'error_count': 0
        }
        
        logger.info("数据质量检查器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('quality_check', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'completeness': {
                'required_fields': ['content', 'author', 'title'],
                'min_content_length': 100,
                'max_content_length': 100000,
                'check_empty_fields': True
            },
            'consistency': {
                'check_date_logic': True,
                'check_numerical_consistency': True,
                'check_format_consistency': True,
                'author_name_variations': 3
            },
            'accuracy': {
                'check_stock_codes': True,
                'check_date_formats': True,
                'check_number_formats': True,
                'validate_email_phone': True,
                'reasonable_number_ranges': {
                    'growth_rate': (-100, 1000),  # 增长率范围 -100% 到 1000%
                    'pe_ratio': (0, 200),         # 市盈率范围
                    'price': (0, 10000),          # 股价范围（元）
                    'market_cap': (0, 100000)     # 市值范围（亿元）
                }
            },
            'timeliness': {
                'max_report_age_days': 365,
                'check_future_dates': True,
                'validate_market_hours': True
            },
            'uniqueness': {
                'check_duplicate_content': True,
                'similarity_threshold': 0.9,
                'check_duplicate_titles': True
            },
            'business_rules': {
                'valid_stock_exchanges': ['SZ', 'SH', 'BJ'],
                'valid_report_types': ['公司研究', '行业研究', '策略研究', '债券研究'],
                'valid_ratings': ['买入', '增持', '中性', '减持', '卖出'],
                'min_analyst_count': 1,
                'max_analyst_count': 10
            },
            'scoring': {
                'completeness_weight': 0.3,
                'consistency_weight': 0.25,
                'accuracy_weight': 0.25,
                'timeliness_weight': 0.1,
                'uniqueness_weight': 0.1
            }
        }
    
    def _compile_quality_rules(self) -> Dict[str, re.Pattern]:
        """编译质量检查的正则表达式规则"""
        rules = {}
        
        # 股票代码格式
        rules['stock_code'] = re.compile(r'^[0-9]{6}$')
        rules['stock_code_with_exchange'] = re.compile(r'^[0-9]{6}\.(SZ|SH|BJ)$')
        
        # 日期格式
        rules['date_iso'] = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        rules['date_chinese'] = re.compile(r'^\d{4}年\d{1,2}月\d{1,2}日$')
        rules['datetime_iso'] = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
        
        # 数字格式
        rules['number'] = re.compile(r'^-?\d+(\.\d+)?$')
        rules['percentage'] = re.compile(r'^-?\d+(\.\d+)?%$')
        rules['currency'] = re.compile(r'^-?\d+(\.\d+)?[万亿千百十元美金]?$')
        
        # 联系方式
        rules['email'] = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        rules['phone'] = re.compile(r'^(\+?86[-\s]?)?1[3-9]\d{9}$|^\d{3,4}-?\d{7,8}$')
        
        # 文本质量
        rules['repeated_sentences'] = re.compile(r'(.{20,}?)\1{2,}')
        rules['excessive_punctuation'] = re.compile(r'[!！?？.。]{3,}')
        rules['mixed_language'] = re.compile(r'[a-zA-Z]{10,}.*[\u4e00-\u9fff]{10,}|[\u4e00-\u9fff]{10,}.*[a-zA-Z]{10,}')
        
        return rules
    
    def check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        全面检查数据质量
        
        Args:
            data: 研报数据字典
            
        Returns:
            Dict[str, Any]: 质量检查结果
        """
        self.check_stats['total_checked'] += 1
        
        # 初始化检查结果
        check_result = {
            'overall_score': 0.0,
            'quality_level': 'UNKNOWN',
            'passed': False,
            'issues': [],
            'warnings': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. 完整性检查
            completeness_result = self._check_completeness(data)
            check_result['details']['completeness'] = completeness_result
            
            # 2. 一致性检查
            consistency_result = self._check_consistency(data)
            check_result['details']['consistency'] = consistency_result
            
            # 3. 准确性检查
            accuracy_result = self._check_accuracy(data)
            check_result['details']['accuracy'] = accuracy_result
            
            # 4. 时效性检查
            timeliness_result = self._check_timeliness(data)
            check_result['details']['timeliness'] = timeliness_result
            
            # 5. 唯一性检查（需要外部数据支持）
            uniqueness_result = self._check_uniqueness(data)
            check_result['details']['uniqueness'] = uniqueness_result
            
            # 6. 业务规则检查
            business_rules_result = self._check_business_rules(data)
            check_result['details']['business_rules'] = business_rules_result
            
            # 汇总结果
            check_result = self._aggregate_results(check_result)
            
            # 更新统计信息
            if check_result['passed']:
                self.check_stats['passed_checks'] += 1
            else:
                self.check_stats['failed_checks'] += 1
            
            self.check_stats['warning_count'] += len(check_result['warnings'])
            self.check_stats['error_count'] += len([issue for issue in check_result['issues'] if issue.get('severity') == 'error'])
            
            return check_result
            
        except Exception as e:
            logger.error(f"数据质量检查失败: {e}")
            check_result.update({
                'overall_score': 0.0,
                'quality_level': 'ERROR',
                'passed': False,
                'issues': [{'type': 'system_error', 'message': str(e), 'severity': 'error'}],
                'error': str(e)
            })
            return check_result
    
    def _check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据完整性"""
        result = {
            'score': 1.0,
            'issues': [],
            'missing_fields': [],
            'empty_fields': [],
            'length_issues': []
        }
        
        # 检查必填字段
        required_fields = self.config['completeness']['required_fields']
        for field in required_fields:
            if field not in data:
                result['missing_fields'].append(field)
                result['issues'].append({
                    'type': 'missing_field',
                    'field': field,
                    'message': f'缺少必填字段: {field}',
                    'severity': 'error'
                })
            elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
                result['empty_fields'].append(field)
                result['issues'].append({
                    'type': 'empty_field',
                    'field': field,
                    'message': f'字段为空: {field}',
                    'severity': 'error'
                })
        
        # 检查内容长度
        if 'content' in data and isinstance(data['content'], str):
            content_length = len(data['content'])
            min_length = self.config['completeness']['min_content_length']
            max_length = self.config['completeness']['max_content_length']
            
            if content_length < min_length:
                result['length_issues'].append('content_too_short')
                result['issues'].append({
                    'type': 'content_too_short',
                    'actual_length': content_length,
                    'min_length': min_length,
                    'message': f'内容过短: {content_length} < {min_length}',
                    'severity': 'warning'
                })
            elif content_length > max_length:
                result['length_issues'].append('content_too_long')
                result['issues'].append({
                    'type': 'content_too_long',
                    'actual_length': content_length,
                    'max_length': max_length,
                    'message': f'内容过长: {content_length} > {max_length}',
                    'severity': 'warning'
                })
        
        # 计算完整性分数
        total_fields = len(required_fields)
        missing_count = len(result['missing_fields']) + len(result['empty_fields'])
        if total_fields > 0:
            result['score'] = 1.0 - (missing_count / total_fields)
        
        # 长度问题的轻微扣分
        if result['length_issues']:
            result['score'] *= 0.9
        
        return result
    
    def _check_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据一致性"""
        result = {
            'score': 1.0,
            'issues': [],
            'date_inconsistencies': [],
            'numerical_inconsistencies': [],
            'format_inconsistencies': []
        }
        
        # 检查日期逻辑一致性
        if self.config['consistency']['check_date_logic']:
            date_issues = self._check_date_consistency(data)
            result['date_inconsistencies'].extend(date_issues)
            result['issues'].extend(date_issues)
        
        # 检查数值一致性
        if self.config['consistency']['check_numerical_consistency']:
            numerical_issues = self._check_numerical_consistency(data)
            result['numerical_inconsistencies'].extend(numerical_issues)
            result['issues'].extend(numerical_issues)
        
        # 检查格式一致性
        if self.config['consistency']['check_format_consistency']:
            format_issues = self._check_format_consistency(data)
            result['format_inconsistencies'].extend(format_issues)
            result['issues'].extend(format_issues)
        
        # 计算一致性分数
        total_issues = len(result['issues'])
        if total_issues > 0:
            result['score'] = max(0.0, 1.0 - total_issues * 0.1)
        
        return result
    
    def _check_date_consistency(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查日期一致性"""
        issues = []
        
        # 收集所有日期字段
        date_fields = ['publish_date', 'analysis_date', 'report_date', 'update_date']
        dates = {}
        
        for field in date_fields:
            if field in data and data[field]:
                try:
                    if isinstance(data[field], str):
                        # 尝试解析日期字符串
                        date_obj = self._parse_date_string(data[field])
                        if date_obj:
                            dates[field] = date_obj
                    elif isinstance(data[field], datetime):
                        dates[field] = data[field]
                except Exception as e:
                    issues.append({
                        'type': 'date_parse_error',
                        'field': field,
                        'value': data[field],
                        'message': f'日期解析失败: {field} = {data[field]}',
                        'severity': 'error'
                    })
        
        # 检查日期逻辑
        if 'publish_date' in dates and 'analysis_date' in dates:
            if dates['publish_date'] < dates['analysis_date']:
                issues.append({
                    'type': 'date_logic_error',
                    'message': '发布日期早于分析日期',
                    'severity': 'error'
                })
        
        # 检查未来日期
        now = datetime.now()
        for field, date_obj in dates.items():
            if date_obj > now + timedelta(days=1):  # 允许1天的时差
                issues.append({
                    'type': 'future_date',
                    'field': field,
                    'value': date_obj.isoformat(),
                    'message': f'{field}是未来日期',
                    'severity': 'warning'
                })
        
        return issues
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y年%m月%d日',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _check_numerical_consistency(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查数值一致性"""
        issues = []
        
        # 提取数值字段
        numerical_fields = ['target_price', 'current_price', 'growth_rate', 'pe_ratio', 'market_cap']
        numbers = {}
        
        for field in numerical_fields:
            if field in data and data[field] is not None:
                try:
                    if isinstance(data[field], (int, float)):
                        numbers[field] = float(data[field])
                    elif isinstance(data[field], str):
                        # 从字符串中提取数字
                        number = self._extract_number_from_string(data[field])
                        if number is not None:
                            numbers[field] = number
                except:
                    issues.append({
                        'type': 'number_parse_error',
                        'field': field,
                        'value': data[field],
                        'message': f'数值解析失败: {field} = {data[field]}',
                        'severity': 'error'
                    })
        
        # 检查数值逻辑
        if 'target_price' in numbers and 'current_price' in numbers:
            if numbers['target_price'] <= 0 or numbers['current_price'] <= 0:
                issues.append({
                    'type': 'invalid_price',
                    'message': '价格不能为负数或零',
                    'severity': 'error'
                })
        
        # 检查合理性范围
        ranges = self.config['accuracy']['reasonable_number_ranges']
        for field, number in numbers.items():
            if field in ranges:
                min_val, max_val = ranges[field]
                if not (min_val <= number <= max_val):
                    issues.append({
                        'type': 'unreasonable_value',
                        'field': field,
                        'value': number,
                        'expected_range': f'{min_val} - {max_val}',
                        'message': f'{field}数值超出合理范围: {number}',
                        'severity': 'warning'
                    })
        
        return issues
    
    def _extract_number_from_string(self, text: str) -> Optional[float]:
        """从字符串中提取数字"""
        # 移除百分号并处理
        text = text.replace('%', '')
        
        # 处理中文数字单位
        multipliers = {'万': 10000, '亿': 100000000, '千': 1000, '百': 100}
        
        for unit, multiplier in multipliers.items():
            if unit in text:
                number_part = text.replace(unit, '').strip()
                try:
                    return float(number_part) * multiplier
                except ValueError:
                    continue
        
        # 直接尝试转换为浮点数
        try:
            return float(text)
        except ValueError:
            return None
    
    def _check_format_consistency(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查格式一致性"""
        issues = []
        
        # 检查股票代码格式
        if 'stocks' in data and isinstance(data['stocks'], list):
            for stock_code in data['stocks']:
                if not self.quality_rules['stock_code'].match(stock_code) and \
                   not self.quality_rules['stock_code_with_exchange'].match(stock_code):
                    issues.append({
                        'type': 'invalid_stock_code_format',
                        'value': stock_code,
                        'message': f'股票代码格式错误: {stock_code}',
                        'severity': 'error'
                    })
        
        # 检查邮箱格式
        if 'email' in data and data['email']:
            if not self.quality_rules['email'].match(data['email']):
                issues.append({
                    'type': 'invalid_email_format',
                    'value': data['email'],
                    'message': f'邮箱格式错误: {data["email"]}',
                    'severity': 'warning'
                })
        
        # 检查电话格式
        if 'phone' in data and data['phone']:
            if not self.quality_rules['phone'].match(data['phone']):
                issues.append({
                    'type': 'invalid_phone_format',
                    'value': data['phone'],
                    'message': f'电话格式错误: {data["phone"]}',
                    'severity': 'warning'
                })
        
        return issues
    
    def _check_accuracy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据准确性"""
        result = {
            'score': 1.0,
            'issues': [],
            'format_errors': [],
            'validation_errors': []
        }
        
        # 检查各种格式和数据有效性
        format_issues = self._check_format_consistency(data)
        result['format_errors'].extend(format_issues)
        result['issues'].extend(format_issues)
        
        # 检查内容中的文本质量
        if 'content' in data and isinstance(data['content'], str):
            text_quality_issues = self._check_text_quality(data['content'])
            result['validation_errors'].extend(text_quality_issues)
            result['issues'].extend(text_quality_issues)
        
        # 计算准确性分数
        error_count = len([issue for issue in result['issues'] if issue.get('severity') == 'error'])
        warning_count = len([issue for issue in result['issues'] if issue.get('severity') == 'warning'])
        
        result['score'] = max(0.0, 1.0 - error_count * 0.2 - warning_count * 0.1)
        
        return result
    
    def _check_text_quality(self, text: str) -> List[Dict[str, Any]]:
        """检查文本质量"""
        issues = []
        
        # 检查重复句子
        if self.quality_rules['repeated_sentences'].search(text):
            issues.append({
                'type': 'repeated_content',
                'message': '发现重复的句子或段落',
                'severity': 'warning'
            })
        
        # 检查过度标点符号
        if self.quality_rules['excessive_punctuation'].search(text):
            issues.append({
                'type': 'excessive_punctuation',
                'message': '存在过多的标点符号',
                'severity': 'warning'
            })
        
        # 检查中英文混杂
        if self.quality_rules['mixed_language'].search(text):
            issues.append({
                'type': 'mixed_language',
                'message': '中英文混杂可能影响可读性',
                'severity': 'info'
            })
        
        return issues
    
    def _check_timeliness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据时效性"""
        result = {
            'score': 1.0,
            'issues': [],
            'age_days': 0,
            'is_stale': False
        }
        
        # 检查报告年龄
        publish_date = None
        if 'publish_date' in data:
            if isinstance(data['publish_date'], datetime):
                publish_date = data['publish_date']
            elif isinstance(data['publish_date'], str):
                publish_date = self._parse_date_string(data['publish_date'])
        
        if publish_date:
            age = datetime.now() - publish_date
            result['age_days'] = age.days
            
            max_age = self.config['timeliness']['max_report_age_days']
            if age.days > max_age:
                result['is_stale'] = True
                result['issues'].append({
                    'type': 'stale_data',
                    'age_days': age.days,
                    'max_age_days': max_age,
                    'message': f'数据过期: {age.days}天前的报告',
                    'severity': 'warning'
                })
                result['score'] = max(0.0, 1.0 - (age.days - max_age) / max_age)
        
        return result
    
    def _check_uniqueness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据唯一性（简化版本，实际需要外部数据库支持）"""
        result = {
            'score': 1.0,
            'issues': [],
            'potential_duplicates': []
        }
        
        # 简单的内容重复检查（在实际应用中需要与数据库比较）
        if 'content' in data and isinstance(data['content'], str):
            content = data['content']
            
            # 检查内容是否过于简短（可能是模板）
            if len(content) < 200:
                result['issues'].append({
                    'type': 'template_like_content',
                    'message': '内容过短，可能是模板内容',
                    'severity': 'info'
                })
        
        return result
    
    def _check_business_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查业务规则"""
        result = {
            'score': 1.0,
            'issues': [],
            'rule_violations': []
        }
        
        # 检查评级有效性
        if 'rating' in data:
            valid_ratings = self.config['business_rules']['valid_ratings']
            if data['rating'] and data['rating'] not in valid_ratings:
                result['issues'].append({
                    'type': 'invalid_rating',
                    'value': data['rating'],
                    'valid_values': valid_ratings,
                    'message': f'无效的评级: {data["rating"]}',
                    'severity': 'error'
                })
        
        # 检查分析师数量
        if 'analysts' in data and isinstance(data['analysts'], list):
            analyst_count = len(data['analysts'])
            min_count = self.config['business_rules']['min_analyst_count']
            max_count = self.config['business_rules']['max_analyst_count']
            
            if analyst_count < min_count or analyst_count > max_count:
                result['issues'].append({
                    'type': 'invalid_analyst_count',
                    'count': analyst_count,
                    'valid_range': f'{min_count}-{max_count}',
                    'message': f'分析师数量异常: {analyst_count}',
                    'severity': 'warning'
                })
        
        # 检查股票交易所
        if 'stocks' in data and isinstance(data['stocks'], list):
            valid_exchanges = self.config['business_rules']['valid_stock_exchanges']
            for stock in data['stocks']:
                if '.' in stock:
                    exchange = stock.split('.')[1]
                    if exchange not in valid_exchanges:
                        result['issues'].append({
                            'type': 'invalid_exchange',
                            'value': stock,
                            'valid_exchanges': valid_exchanges,
                            'message': f'无效的交易所: {exchange}',
                            'severity': 'error'
                        })
        
        # 计算业务规则分数
        error_count = len([issue for issue in result['issues'] if issue.get('severity') == 'error'])
        result['score'] = max(0.0, 1.0 - error_count * 0.3)
        
        return result
    
    def _aggregate_results(self, check_result: Dict[str, Any]) -> Dict[str, Any]:
        """汇总检查结果"""
        details = check_result['details']
        weights = self.config['scoring']
        
        # 计算加权总分
        total_score = (
            details['completeness']['score'] * weights['completeness_weight'] +
            details['consistency']['score'] * weights['consistency_weight'] +
            details['accuracy']['score'] * weights['accuracy_weight'] +
            details['timeliness']['score'] * weights['timeliness_weight'] +
            details['uniqueness']['score'] * weights['uniqueness_weight']
        )
        
        check_result['overall_score'] = total_score
        
        # 收集所有问题
        all_issues = []
        warnings = []
        
        for category_result in details.values():
            if 'issues' in category_result:
                for issue in category_result['issues']:
                    if issue.get('severity') == 'warning':
                        warnings.append(issue)
                    else:
                        all_issues.append(issue)
        
        check_result['issues'] = all_issues
        check_result['warnings'] = warnings
        
        # 确定质量等级
        if total_score >= 0.9:
            check_result['quality_level'] = 'EXCELLENT'
        elif total_score >= 0.8:
            check_result['quality_level'] = 'GOOD'
        elif total_score >= 0.6:
            check_result['quality_level'] = 'FAIR'
        elif total_score >= 0.4:
            check_result['quality_level'] = 'POOR'
        else:
            check_result['quality_level'] = 'VERY_POOR'
        
        # 判断是否通过
        check_result['passed'] = total_score >= 0.6 and len(all_issues) == 0
        
        return check_result
    
    def batch_check_quality(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量检查数据质量
        
        Args:
            data_list: 数据列表
            
        Returns:
            List[Dict[str, Any]]: 检查结果列表
        """
        results = []
        
        for i, data in enumerate(data_list):
            try:
                result = self.check_data_quality(data)
                result['index'] = i
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"已检查 {i + 1}/{len(data_list)} 份数据")
                    
            except Exception as e:
                logger.error(f"批量质量检查第 {i} 份数据失败: {e}")
                results.append({
                    'index': i,
                    'overall_score': 0.0,
                    'quality_level': 'ERROR',
                    'passed': False,
                    'issues': [{'type': 'processing_error', 'message': str(e), 'severity': 'error'}],
                    'error': str(e)
                })
        
        return results
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        获取质量检查统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total = self.check_stats['total_checked']
        
        return {
            'total_checked': total,
            'passed_rate': self.check_stats['passed_checks'] / max(total, 1),
            'failed_rate': self.check_stats['failed_checks'] / max(total, 1),
            'average_warnings_per_check': self.check_stats['warning_count'] / max(total, 1),
            'average_errors_per_check': self.check_stats['error_count'] / max(total, 1),
            'config': self.config
        }


# 全局质量检查器实例
_global_quality_checker = None


def get_quality_checker() -> QualityChecker:
    """
    获取全局质量检查器实例
    
    Returns:
        QualityChecker: 检查器实例
    """
    global _global_quality_checker
    
    if _global_quality_checker is None:
        _global_quality_checker = QualityChecker()
    
    return _global_quality_checker


if __name__ == "__main__":
    # 使用示例
    checker = QualityChecker()
    
    # 测试数据
    test_data = {
        'title': '平安银行(000001.SZ)投资价值分析',
        'content': '平安银行作为国内领先的股份制银行，在零售业务方面表现突出。' * 50,  # 足够长的内容
        'author': '张三',
        'analysts': ['张三', '李四'],
        'stocks': ['000001.SZ'],
        'rating': '买入',
        'target_price': 15.50,
        'current_price': 12.30,
        'growth_rate': 15.5,
        'publish_date': '2024-01-15',
        'analysis_date': '2024-01-14',
        'email': 'analyst@example.com',
        'phone': '021-12345678'
    }
    
    result = checker.check_data_quality(test_data)
    
    print("数据质量检查结果:")
    print(f"总体分数: {result['overall_score']:.2f}")
    print(f"质量等级: {result['quality_level']}")
    print(f"是否通过: {result['passed']}")
    print(f"问题数量: {len(result['issues'])}")
    print(f"警告数量: {len(result['warnings'])}")
    
    if result['issues']:
        print("\n发现的问题:")
        for issue in result['issues']:
            print(f"  - {issue['type']}: {issue['message']} ({issue['severity']})")
    
    # 获取统计信息
    stats = checker.get_quality_statistics()
    print(f"\n质量检查统计: {stats}") 
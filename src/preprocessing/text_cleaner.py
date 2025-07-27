"""
文本清洗器
处理研报文本的清洗、标准化、去噪、格式化等功能
"""

import re
import html
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor

logger = get_logger(__name__)


class TextCleaner:
    """
    文本清洗器
    
    提供全面的中文研报文本清洗功能：
    1. 基础清洗 - 去除HTML标签、特殊字符、编码问题
    2. 格式标准化 - 统一标点符号、空格、换行符
    3. 内容过滤 - 移除无关内容、广告、免责声明
    4. 质量检查 - 检测乱码、重复内容、异常格式
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 清洗配置参数
        text_processor: 文本处理器
        cleaning_rules: 清洗规则集合
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化文本清洗器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        
        # 预编译正则表达式提高性能
        self.cleaning_rules = self._compile_cleaning_rules()
        
        # 统计信息
        self.cleaning_stats = {
            'total_processed': 0,
            'total_cleaned': 0,
            'html_removed': 0,
            'special_chars_removed': 0,
            'encoding_fixed': 0,
            'duplicates_removed': 0
        }
        
        logger.info("文本清洗器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('text_cleaning', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'basic_cleaning': {
                'remove_html': True,
                'fix_encoding': True,
                'normalize_unicode': True,
                'remove_control_chars': True
            },
            'format_standardization': {
                'normalize_punctuation': True,
                'normalize_whitespace': True,
                'normalize_quotes': True,
                'standardize_numbers': True
            },
            'content_filtering': {
                'remove_disclaimers': True,
                'remove_advertisements': True,
                'remove_headers_footers': True,
                'remove_contact_info': True,
                'min_sentence_length': 10,
                'max_sentence_length': 1000
            },
            'quality_control': {
                'detect_gibberish': True,
                'remove_duplicates': True,
                'check_language': True,
                'min_chinese_ratio': 0.7,
                'max_repeat_ratio': 0.3
            },
            'preservation': {
                'preserve_financial_terms': True,
                'preserve_stock_codes': True,
                'preserve_dates': True,
                'preserve_numbers_with_units': True
            }
        }
    
    def _compile_cleaning_rules(self) -> Dict[str, re.Pattern]:
        """预编译清洗规则的正则表达式"""
        rules = {}
        
        # HTML标签和实体
        rules['html_tags'] = re.compile(r'<[^>]+>')
        rules['html_entities'] = re.compile(r'&[a-zA-Z0-9#]+;')
        
        # 控制字符和特殊字符
        rules['control_chars'] = re.compile(r'[\x00-\x1f\x7f-\x9f]')
        rules['invisible_chars'] = re.compile(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]')
        
        # 空白字符规范化
        rules['multiple_spaces'] = re.compile(r'\s+')
        rules['multiple_newlines'] = re.compile(r'\n\s*\n\s*\n+')
        
        # 标点符号规范化
        rules['chinese_punctuation'] = re.compile(r'[，。！？；：""''（）【】《》〈〉『』「」〔〕]')
        rules['english_punctuation'] = re.compile(r'[,.!?;:"\'()\[\]<>{}]')
        
        # 数字和单位
        rules['number_with_unit'] = re.compile(r'\d+(?:\.\d+)?[万亿千百十元美金%‰]+')
        rules['stock_code'] = re.compile(r'[0-9]{6}(?:\.[A-Z]{2})?')
        rules['date_pattern'] = re.compile(r'\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?')
        
        # 垃圾内容模式
        rules['disclaimer'] = re.compile(r'免责声明|风险提示|投资建议|市场有风险|投资需谨慎|本报告|仅供参考', re.IGNORECASE)
        rules['advertisement'] = re.compile(r'广告|推广|扫码|微信|QQ|电话|邮箱|网址|链接', re.IGNORECASE)
        rules['contact_info'] = re.compile(r'(\d{3,4}[-\s]?\d{7,8})|(\w+@\w+\.\w+)|(www\.\w+\.\w+)', re.IGNORECASE)
        
        # 重复内容
        rules['repeated_chars'] = re.compile(r'(.)\1{3,}')  # 4个以上相同字符
        rules['repeated_words'] = re.compile(r'(\S+)(\s+\1){2,}')  # 重复单词
        
        return rules
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> Dict[str, Any]:
        """
        全面清洗文本
        
        Args:
            text: 原始文本
            preserve_structure: 是否保留文本结构（段落、换行）
            
        Returns:
            Dict[str, Any]: 清洗结果
        """
        if not text or not isinstance(text, str):
            return {
                'original_text': text,
                'cleaned_text': '',
                'cleaning_applied': [],
                'quality_score': 0.0,
                'issues_found': ['empty_or_invalid_input']
            }
        
        self.cleaning_stats['total_processed'] += 1
        
        # 保存原始文本
        original_text = text
        cleaned_text = text
        cleaning_applied = []
        issues_found = []
        
        try:
            # 1. 基础清洗
            if self.config['basic_cleaning']['remove_html']:
                cleaned_text, html_removed = self._remove_html(cleaned_text)
                if html_removed:
                    cleaning_applied.append('html_removed')
                    self.cleaning_stats['html_removed'] += 1
            
            if self.config['basic_cleaning']['fix_encoding']:
                cleaned_text, encoding_fixed = self._fix_encoding_issues(cleaned_text)
                if encoding_fixed:
                    cleaning_applied.append('encoding_fixed')
                    self.cleaning_stats['encoding_fixed'] += 1
            
            if self.config['basic_cleaning']['normalize_unicode']:
                cleaned_text = self._normalize_unicode(cleaned_text)
                cleaning_applied.append('unicode_normalized')
            
            if self.config['basic_cleaning']['remove_control_chars']:
                cleaned_text = self._remove_control_characters(cleaned_text)
                cleaning_applied.append('control_chars_removed')
            
            # 2. 格式标准化
            if self.config['format_standardization']['normalize_whitespace']:
                cleaned_text = self._normalize_whitespace(cleaned_text, preserve_structure)
                cleaning_applied.append('whitespace_normalized')
            
            if self.config['format_standardization']['normalize_punctuation']:
                cleaned_text = self._normalize_punctuation(cleaned_text)
                cleaning_applied.append('punctuation_normalized')
            
            if self.config['format_standardization']['normalize_quotes']:
                cleaned_text = self._normalize_quotes(cleaned_text)
                cleaning_applied.append('quotes_normalized')
            
            # 3. 内容过滤
            if self.config['content_filtering']['remove_disclaimers']:
                cleaned_text, disclaimers_removed = self._remove_disclaimers(cleaned_text)
                if disclaimers_removed:
                    cleaning_applied.append('disclaimers_removed')
            
            if self.config['content_filtering']['remove_advertisements']:
                cleaned_text, ads_removed = self._remove_advertisements(cleaned_text)
                if ads_removed:
                    cleaning_applied.append('advertisements_removed')
            
            if self.config['content_filtering']['remove_contact_info']:
                cleaned_text, contact_removed = self._remove_contact_info(cleaned_text)
                if contact_removed:
                    cleaning_applied.append('contact_info_removed')
            
            # 4. 质量控制
            if self.config['quality_control']['remove_duplicates']:
                cleaned_text, duplicates_removed = self._remove_duplicates(cleaned_text)
                if duplicates_removed:
                    cleaning_applied.append('duplicates_removed')
                    self.cleaning_stats['duplicates_removed'] += 1
            
            # 5. 质量检查
            quality_issues = self._check_text_quality(cleaned_text)
            issues_found.extend(quality_issues)
            
            # 6. 最终清理
            cleaned_text = self._final_cleanup(cleaned_text)
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(original_text, cleaned_text, issues_found)
            
            if len(cleaning_applied) > 0:
                self.cleaning_stats['total_cleaned'] += 1
            
            return {
                'original_text': original_text,
                'cleaned_text': cleaned_text,
                'cleaning_applied': cleaning_applied,
                'quality_score': quality_score,
                'issues_found': issues_found,
                'length_reduction': len(original_text) - len(cleaned_text),
                'reduction_ratio': 1 - len(cleaned_text) / max(len(original_text), 1)
            }
            
        except Exception as e:
            logger.error(f"文本清洗失败: {e}")
            return {
                'original_text': original_text,
                'cleaned_text': original_text,  # 返回原文
                'cleaning_applied': [],
                'quality_score': 0.5,
                'issues_found': [f'cleaning_error: {str(e)}'],
                'error': str(e)
            }
    
    def _remove_html(self, text: str) -> Tuple[str, bool]:
        """移除HTML标签和实体"""
        original_length = len(text)
        
        # 移除HTML标签
        text = self.cleaning_rules['html_tags'].sub('', text)
        
        # 解码HTML实体
        text = html.unescape(text)
        
        # 移除剩余的HTML实体
        text = self.cleaning_rules['html_entities'].sub('', text)
        
        return text, len(text) < original_length
    
    def _fix_encoding_issues(self, text: str) -> Tuple[str, bool]:
        """修复编码问题"""
        original_text = text
        fixed = False
        
        try:
            # 检测和修复常见的编码问题
            if '锟' in text or '�' in text:
                # UTF-8解码错误的修复尝试
                try:
                    # 尝试重新编码
                    text = text.encode('latin1').decode('utf-8', errors='ignore')
                    fixed = True
                except:
                    pass
            
            # 移除无法显示的字符
            text = ''.join(char for char in text if ord(char) < 65536)
            
        except Exception as e:
            logger.debug(f"编码修复失败: {e}")
        
        return text, fixed or len(text) != len(original_text)
    
    def _normalize_unicode(self, text: str) -> str:
        """Unicode规范化"""
        # 使用NFKC规范化，将兼容字符转换为标准形式
        return unicodedata.normalize('NFKC', text)
    
    def _remove_control_characters(self, text: str) -> str:
        """移除控制字符"""
        # 移除控制字符，但保留换行符和制表符
        text = self.cleaning_rules['control_chars'].sub('', text)
        
        # 移除不可见字符
        text = self.cleaning_rules['invisible_chars'].sub('', text)
        
        return text
    
    def _normalize_whitespace(self, text: str, preserve_structure: bool) -> str:
        """规范化空白字符"""
        if preserve_structure:
            # 保留段落结构，但规范化空格
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # 规范化行内空格
                line = self.cleaning_rules['multiple_spaces'].sub(' ', line.strip())
                cleaned_lines.append(line)
            
            # 移除多余的空行
            text = '\n'.join(cleaned_lines)
            text = self.cleaning_rules['multiple_newlines'].sub('\n\n', text)
        else:
            # 将所有空白字符转换为单个空格
            text = self.cleaning_rules['multiple_spaces'].sub(' ', text)
        
        return text.strip()
    
    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号"""
        # 统一中英文标点符号 - 使用列表避免字典键重复问题
        punctuation_replacements = [
            ('，', '，'), ('。', '。'), ('！', '！'), ('？', '？'),
            ('；', '；'), ('：', '：'), ('"', '"'), ('"', '"'),
            (''', '''), (''', '''), ('（', '（'), ('）', '）'),
            ('【', '【'), ('】', '】'), ('《', '《'), ('》', '》')
        ]
        
        for old, new in punctuation_replacements:
            text = text.replace(old, new)
        
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """规范化引号"""
        # 统一引号格式
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        return text
    
    def _remove_disclaimers(self, text: str) -> Tuple[str, bool]:
        """移除免责声明"""
        original_length = len(text)
        
        # 寻找免责声明段落并移除
        sentences = text.split('。')
        filtered_sentences = []
        
        for sentence in sentences:
            if not self.cleaning_rules['disclaimer'].search(sentence):
                filtered_sentences.append(sentence)
        
        text = '。'.join(filtered_sentences)
        
        return text, len(text) < original_length
    
    def _remove_advertisements(self, text: str) -> Tuple[str, bool]:
        """移除广告内容"""
        original_length = len(text)
        
        # 移除包含广告关键词的句子
        sentences = text.split('。')
        filtered_sentences = []
        
        for sentence in sentences:
            if not self.cleaning_rules['advertisement'].search(sentence):
                filtered_sentences.append(sentence)
        
        text = '。'.join(filtered_sentences)
        
        return text, len(text) < original_length
    
    def _remove_contact_info(self, text: str) -> Tuple[str, bool]:
        """移除联系信息"""
        original_length = len(text)
        
        # 移除联系方式
        text = self.cleaning_rules['contact_info'].sub('', text)
        
        return text, len(text) < original_length
    
    def _remove_duplicates(self, text: str) -> Tuple[str, bool]:
        """移除重复内容"""
        original_length = len(text)
        
        # 移除重复字符
        text = self.cleaning_rules['repeated_chars'].sub(r'\1\1\1', text)
        
        # 移除重复单词
        text = self.cleaning_rules['repeated_words'].sub(r'\1', text)
        
        # 移除重复句子
        sentences = text.split('。')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence)
        
        text = '。'.join(unique_sentences)
        
        return text, len(text) < original_length
    
    def _check_text_quality(self, text: str) -> List[str]:
        """检查文本质量"""
        issues = []
        
        # 检查长度
        if len(text) < 50:
            issues.append('text_too_short')
        elif len(text) > 50000:
            issues.append('text_too_long')
        
        # 检查中文比例
        if self.config['quality_control']['check_language']:
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            chinese_ratio = chinese_chars / max(len(text), 1)
            
            min_chinese_ratio = self.config['quality_control']['min_chinese_ratio']
            if chinese_ratio < min_chinese_ratio:
                issues.append(f'low_chinese_ratio_{chinese_ratio:.2f}')
        
        # 检查重复内容比例
        if self.config['quality_control']['detect_gibberish']:
            # 简单的乱码检测
            weird_chars = len([c for c in text if ord(c) > 65535])
            if weird_chars > len(text) * 0.05:
                issues.append('possible_gibberish')
        
        # 检查重复比例
        max_repeat_ratio = self.config['quality_control']['max_repeat_ratio']
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            repeat_ratio = max_count / len(words)
            
            if repeat_ratio > max_repeat_ratio:
                issues.append(f'high_repetition_{repeat_ratio:.2f}')
        
        return issues
    
    def _final_cleanup(self, text: str) -> str:
        """最终清理"""
        # 移除首尾空白
        text = text.strip()
        
        # 确保句子以正确的标点结尾
        if text and text[-1] not in '。！？…':
            text += '。'
        
        return text
    
    def _calculate_quality_score(self, original_text: str, cleaned_text: str, 
                                issues_found: List[str]) -> float:
        """计算文本质量分数"""
        base_score = 1.0
        
        # 根据问题扣分
        issue_penalties = {
            'text_too_short': 0.3,
            'text_too_long': 0.1,
            'possible_gibberish': 0.4,
            'high_repetition': 0.2,
            'low_chinese_ratio': 0.3
        }
        
        for issue in issues_found:
            for issue_type, penalty in issue_penalties.items():
                if issue.startswith(issue_type):
                    base_score -= penalty
                    break
        
        # 根据清洗效果调整
        if original_text and cleaned_text:
            length_ratio = len(cleaned_text) / len(original_text)
            
            # 适当的长度减少是好的
            if 0.7 <= length_ratio <= 0.95:
                base_score += 0.1
            elif length_ratio < 0.5:  # 过度删减
                base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def batch_clean(self, texts: List[str], preserve_structure: bool = True) -> List[Dict[str, Any]]:
        """
        批量清洗文本
        
        Args:
            texts: 文本列表
            preserve_structure: 是否保留文本结构
            
        Returns:
            List[Dict[str, Any]]: 清洗结果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.clean_text(text, preserve_structure)
                result['index'] = i
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(texts)} 篇文本")
                    
            except Exception as e:
                logger.error(f"批量清洗第 {i} 篇文本失败: {e}")
                results.append({
                    'index': i,
                    'original_text': text,
                    'cleaned_text': text,
                    'cleaning_applied': [],
                    'quality_score': 0.0,
                    'issues_found': [f'processing_error: {str(e)}'],
                    'error': str(e)
                })
        
        return results
    
    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """
        获取清洗统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total = self.cleaning_stats['total_processed']
        
        return {
            'total_processed': total,
            'total_cleaned': self.cleaning_stats['total_cleaned'],
            'cleaning_rate': self.cleaning_stats['total_cleaned'] / max(total, 1),
            'html_removal_rate': self.cleaning_stats['html_removed'] / max(total, 1),
            'encoding_fix_rate': self.cleaning_stats['encoding_fixed'] / max(total, 1),
            'duplicate_removal_rate': self.cleaning_stats['duplicates_removed'] / max(total, 1),
            'config': self.config
        }


# 全局清洗器实例
_global_text_cleaner = None


def get_text_cleaner() -> TextCleaner:
    """
    获取全局文本清洗器实例
    
    Returns:
        TextCleaner: 清洗器实例
    """
    global _global_text_cleaner
    
    if _global_text_cleaner is None:
        _global_text_cleaner = TextCleaner()
    
    return _global_text_cleaner


if __name__ == "__main__":
    # 使用示例
    cleaner = TextCleaner()
    
    # 测试文本清洗
    test_text = """
    <p>这是一份关于<strong>平安银行</strong>(000001.SZ)的研报。</p>
    
    &nbsp;&nbsp;&nbsp;免责声明：本报告仅供参考，投资有风险。
    
    该公司2023年营收同比增长15.5%，净利润增长12.3%。。。
    
    更多详情请联系我们：电话 400-123-4567，邮箱 service@test.com
    
    重要重要重要的信息！！！！
    """
    
    result = cleaner.clean_text(test_text)
    
    print("文本清洗结果:")
    print(f"原文长度: {len(result['original_text'])}")
    print(f"清洗后长度: {len(result['cleaned_text'])}")
    print(f"质量分数: {result['quality_score']:.2f}")
    print(f"应用的清洗: {result['cleaning_applied']}")
    print(f"发现的问题: {result['issues_found']}")
    print(f"\n清洗后文本:\n{result['cleaned_text']}")
    
    # 获取统计信息
    stats = cleaner.get_cleaning_statistics()
    print(f"\n清洗统计: {stats}") 
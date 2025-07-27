"""
文本处理工具模块

提供文本清洗、分词、特征提取、统计分析等功能
"""

import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
from collections import Counter, defaultdict
import string
from .logger import get_logger

logger = get_logger(__name__)

# 初始化jieba分词器
jieba.initialize()

# 常用的停用词集合
STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '就是', '我们', '他们', '她们', '可以', '能够', '因为',
    '所以', '但是', '然后', '如果', '虽然', '却', '还是', '已经', '现在', '之后',
    '之前', '比较', '非常', '特别', '尤其', '包括', '以及', '或者', '并且', '而且',
    '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '怎样', '哪里',
    '什么时候', '多少', '几个', '一些', '许多', '很多', '不少', '大量', '少量'
}

# 财经领域专用词典
FINANCE_KEYWORDS = {
    '股票', '股价', '涨幅', '跌幅', '市值', '成交量', '换手率', '市盈率', '市净率',
    '营收', '净利润', '毛利率', 'ROE', 'ROA', 'EPS', 'PE', 'PB', 'PEG',
    '券商', '研报', '评级', '目标价', '买入', '卖出', '持有', '增持', '减持',
    '上涨', '下跌', '横盘', '震荡', '突破', '支撑', '阻力', '趋势', '反弹',
    '业绩', '财报', '年报', '季报', '分红', '派息', '配股', '增发', '重组',
    '并购', 'IPO', '上市', '退市', '停牌', '复牌', '限售', '解禁'
}


class TextProcessor:
    """
    文本处理器类
    
    提供文本清洗、分词、特征提取等功能
    
    Attributes:
        stopwords: 停用词集合
        custom_dict: 自定义词典
    """
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None,
                 custom_dict_path: Optional[str] = None):
        """
        初始化文本处理器
        
        Args:
            custom_stopwords: 自定义停用词
            custom_dict_path: 自定义词典路径
        """
        self.stopwords = STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # 加载自定义词典
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
            logger.info(f"加载自定义词典: {custom_dict_path}")
        
        # 添加财经词汇
        for word in FINANCE_KEYWORDS:
            jieba.add_word(word)
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 删除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 删除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 删除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 删除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5\w\s.,!?;:()（）【】""''《》、。，！？；：]', '', text)
        
        # 删除过长的数字串（可能是无意义的ID）
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        return text.strip()
    
    def segment_text(self, text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
        """
        分词
        
        Args:
            text: 文本
            pos_filter: 词性过滤列表，如['n', 'v', 'a']表示只保留名词、动词、形容词
            
        Returns:
            分词结果
        """
        if not text:
            return []
        
        if pos_filter:
            # 带词性标注的分词
            words_with_pos = pseg.cut(text)
            words = [word for word, pos in words_with_pos if pos in pos_filter]
        else:
            # 普通分词
            words = list(jieba.cut(text))
        
        # 过滤停用词和无意义词汇
        words = [
            word.strip() for word in words
            if word.strip() and 
            len(word.strip()) > 1 and
            word.strip() not in self.stopwords and
            not word.isdigit() and
            not all(c in string.punctuation for c in word)
        ]
        
        return words
    
    def extract_keywords(self, text: str, topk: int = 20) -> List[Tuple[str, float]]:
        """
        提取关键词
        
        Args:
            text: 文本
            topk: 返回前K个关键词
            
        Returns:
            关键词列表 [(词, 权重), ...]
        """
        import jieba.analyse
        
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(
            text, 
            topK=topk, 
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an')
        )
        
        return keywords
    
    def calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """
        计算文本统计信息
        
        Args:
            text: 文本
            
        Returns:
            统计信息字典
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'unique_word_ratio': 0
            }
        
        # 字符数
        char_count = len(text)
        
        # 分词
        words = self.segment_text(text)
        word_count = len(words)
        
        # 句子数
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # 平均词长
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # 平均句长
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # 词汇多样性
        unique_words = set(words)
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_word_ratio': unique_word_ratio,
            'unique_word_count': len(unique_words)
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取命名实体
        
        Args:
            text: 文本
            
        Returns:
            实体字典 {'人名': [...], '地名': [...], '机构名': [...]}
        """
        entities = {
            'person': [],      # 人名
            'location': [],    # 地名
            'organization': [], # 机构名
            'time': [],        # 时间
            'number': []       # 数字
        }
        
        # 使用词性标注提取实体
        words_with_pos = pseg.cut(text)
        
        for word, pos in words_with_pos:
            word = word.strip()
            if len(word) < 2:
                continue
                
            if pos == 'nr':  # 人名
                entities['person'].append(word)
            elif pos in ['ns', 'nt']:  # 地名、机构名
                entities['location'].append(word)
            elif pos == 'nz':  # 其他专名（通常是机构）
                entities['organization'].append(word)
            elif pos == 't':  # 时间
                entities['time'].append(word)
            elif pos == 'm':  # 数词
                entities['number'].append(word)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def calculate_similarity(self, text1: str, text2: str, method: str = 'cosine') -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            method: 相似度计算方法 ('cosine', 'jaccard')
            
        Returns:
            相似度分数 (0-1)
        """
        words1 = set(self.segment_text(text1))
        words2 = set(self.segment_text(text2))
        
        if not words1 or not words2:
            return 0.0
        
        if method == 'jaccard':
            # Jaccard相似度
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        
        elif method == 'cosine':
            # 余弦相似度（基于词汇重叠）
            intersection = len(words1 & words2)
            return intersection / (np.sqrt(len(words1)) * np.sqrt(len(words2)))
        
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
    
    def detect_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """
        检测情感关键词
        
        Args:
            text: 文本
            
        Returns:
            情感关键词统计
        """
        positive_words = {
            '增长', '上涨', '上升', '提升', '改善', '优化', '超预期', '强劲', '稳健',
            '乐观', '看好', '推荐', '买入', '增持', '突破', '新高', '利好', '积极'
        }
        
        negative_words = {
            '下跌', '下降', '下滑', '恶化', '放缓', '疲软', '低迷', '悲观', '谨慎',
            '卖出', '减持', '风险', '压力', '挑战', '困难', '亏损', '利空', '消极'
        }
        
        words = self.segment_text(text)
        
        sentiment_stats = {
            'positive_count': sum(1 for word in words if word in positive_words),
            'negative_count': sum(1 for word in words if word in negative_words),
            'total_words': len(words)
        }
        
        # 计算情感倾向
        if sentiment_stats['total_words'] > 0:
            sentiment_stats['positive_ratio'] = sentiment_stats['positive_count'] / sentiment_stats['total_words']
            sentiment_stats['negative_ratio'] = sentiment_stats['negative_count'] / sentiment_stats['total_words']
            sentiment_stats['sentiment_score'] = sentiment_stats['positive_ratio'] - sentiment_stats['negative_ratio']
        else:
            sentiment_stats['positive_ratio'] = 0
            sentiment_stats['negative_ratio'] = 0
            sentiment_stats['sentiment_score'] = 0
        
        return sentiment_stats
    
    def extract_financial_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        提取财经数字信息
        
        Args:
            text: 文本
            
        Returns:
            数字信息列表
        """
        # 匹配各种数字模式
        patterns = [
            (r'(\d+\.?\d*)\s*万元', 'money_wan'),
            (r'(\d+\.?\d*)\s*亿元', 'money_yi'),
            (r'(\d+\.?\d*)\s*元', 'money_yuan'),
            (r'(\d+\.?\d*)%', 'percentage'),
            (r'市盈率\s*(\d+\.?\d*)', 'pe_ratio'),
            (r'市净率\s*(\d+\.?\d*)', 'pb_ratio'),
            (r'(\d+\.?\d*)\s*倍', 'multiple'),
            (r'目标价\s*(\d+\.?\d*)', 'target_price'),
            (r'(\d{4})年', 'year'),
            (r'(\d+\.?\d*)\s*万股', 'shares_wan')
        ]
        
        financial_numbers = []
        
        for pattern, number_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    value = float(match.group(1))
                    financial_numbers.append({
                        'value': value,
                        'type': number_type,
                        'text': match.group(0),
                        'position': match.span()
                    })
                except (ValueError, IndexError):
                    continue
        
        return financial_numbers
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        计算文本可读性指标
        
        Args:
            text: 文本
            
        Returns:
            可读性指标
        """
        stats = self.calculate_text_stats(text)
        
        if stats['sentence_count'] == 0 or stats['word_count'] == 0:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'complexity_score': 0
            }
        
        # 简单的可读性指标
        avg_sentence_length = stats['avg_sentence_length']
        avg_word_length = stats['avg_word_length']
        
        # 复杂度分数 (值越高越复杂)
        complexity_score = (avg_sentence_length * 0.6 + avg_word_length * 0.4) / 10
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'complexity_score': min(complexity_score, 1.0)  # 限制在0-1之间
        }


def batch_process_texts(texts: List[str], processor: TextProcessor, 
                       operations: List[str]) -> List[Dict[str, Any]]:
    """
    批量处理文本
    
    Args:
        texts: 文本列表
        processor: 文本处理器
        operations: 操作列表，如['clean', 'segment', 'keywords', 'stats']
        
    Returns:
        处理结果列表
    """
    logger.info(f"开始批量处理 {len(texts)} 个文本")
    
    results = []
    
    for i, text in enumerate(texts):
        try:
            result = {'index': i, 'original_text': text}
            
            if 'clean' in operations:
                result['cleaned_text'] = processor.clean_text(text)
                working_text = result['cleaned_text']
            else:
                working_text = text
            
            if 'segment' in operations:
                result['words'] = processor.segment_text(working_text)
            
            if 'keywords' in operations:
                result['keywords'] = processor.extract_keywords(working_text)
            
            if 'stats' in operations:
                result['stats'] = processor.calculate_text_stats(working_text)
            
            if 'entities' in operations:
                result['entities'] = processor.extract_entities(working_text)
            
            if 'sentiment' in operations:
                result['sentiment'] = processor.detect_sentiment_keywords(working_text)
            
            if 'financial_numbers' in operations:
                result['financial_numbers'] = processor.extract_financial_numbers(working_text)
            
            if 'readability' in operations:
                result['readability'] = processor.calculate_readability(working_text)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"文本处理失败 (索引: {i}): {str(e)}")
            results.append({'index': i, 'error': str(e)})
    
    logger.info(f"批量处理完成: {len(results)} 个结果")
    return results


# 全局文本处理器实例
_global_text_processor: Optional[TextProcessor] = None


def get_text_processor() -> TextProcessor:
    """
    获取全局文本处理器实例
    
    Returns:
        文本处理器实例
    """
    global _global_text_processor
    if _global_text_processor is None:
        _global_text_processor = TextProcessor()
    return _global_text_processor 
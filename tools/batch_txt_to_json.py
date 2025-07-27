#!/usr/bin/env python3
"""
批量TXT转JSON工具
专为东吴证券研报格式转换设计，支持批量处理
"""

import os
import re
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


class TxtToJsonConverter:
    """
    将东吴证券研报TXT文件转换为标准化JSON格式的工具
    同时提取东吴证券研报的风格特征，为后续风格分析和异常检测提供基础
    """
    
    def __init__(self, output_dir: str = "user_data/json_files"):
        """
        初始化转换器
        
        Args:
            output_dir: JSON输出目录
        """
        self.output_dir = output_dir
        
        # 投资评级关键词
        self.investment_ratings = {
            "买入": ["买入", "强烈推荐", "强推"],
            "增持": ["增持", "推荐", "跑赢大盘"],
            "中性": ["中性", "持有", "观望", "中立"],
            "减持": ["减持", "卖出", "回避", "弱于大市"]
        }
        
        # 主要驱动因素关键词
        self.driving_factors = [
            "关键假设", "驱动因素", "核心驱动", "主要驱动", "业绩驱动", "增长驱动",
            "催化剂", "投资逻辑", "投资建议", "投资要点"
        ]
        
        # 风险提示关键词
        self.risk_keywords = [
            "风险提示", "风险因素", "风险警示", "风险点", "主要风险"
        ]
        
        # 日期模式
        self.date_patterns = [
            r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?',  # 2025-01-23, 2025年01月23日
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',              # 2025-1-23, 2025/1/23
        ]
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        从文件名中提取元数据
        
        Args:
            filename: 文件名
            
        Returns:
            包含元数据的字典
        """
        metadata = {
            "report_id": f"dongwu_{Path(filename).stem}",
            "broker": "东吴证券",
            "input_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # 尝试从文件名中提取日期
        date_match = re.search(r'(\d{8})|(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            date_str = date_match.group(0)
            if '-' not in date_str and len(date_str) == 8:
                metadata["report_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return metadata
    
    def clean_content(self, content: str) -> str:
        """
        清理文本内容，移除多余空白和特殊字符
        
        Args:
            content: 原始文本内容
            
        Returns:
            清理后的文本
        """
        # 替换多个空格为单个空格
        content = re.sub(r'\s+', ' ', content)
        # 替换多个换行为双换行
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def extract_title(self, content: str) -> str:
        """
        提取研报标题
        
        Args:
            content: 研报内容
            
        Returns:
            研报标题
        """
        # 首先尝试提取第一行作为可能的标题
        lines = content.split('\n')
        if lines:
            first_line = lines[0].strip()
            # 如果第一行长度合理且不是目录或其他非标题内容
            if 5 < len(first_line) < 50 and not re.match(r'^(目录|内容|第.*章|[0-9]+\.)', first_line):
                return first_line
        
        # 如果第一行不合适，尝试在前10行中找到合适的标题
        potential_titles = []
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # 跳过空行、太长的行、以数字或常见非标题开头的行
            if not line or len(line) > 50 or re.match(r'^(目录|内容|第.*章|[0-9]+\.|表格|图表)', line):
                continue
            
            # 如果行包含冒号且冒号前的内容较短，可能是标题
            if ':' in line or '：' in line:
                parts = re.split(r'[:：]', line, 1)
                if len(parts[0]) < 30:  # 标题通常不会太长
                    potential_titles.append(parts[0].strip())
            else:
                potential_titles.append(line)
        
        # 选择最合适的标题（优先选择较短且包含关键词的行）
        if potential_titles:
            # 按长度排序，优先选择较短的
            potential_titles.sort(key=len)
            
            # 检查是否有包含关键词的标题
            for title in potential_titles:
                if any(keyword in title for keyword in ["报告", "研究", "分析", "策略", "行业", "公司"]):
                    return title
            
            # 如果没有包含关键词的，返回最短的
            return potential_titles[0]
        
        # 如果前面的方法都失败了，尝试使用正则表达式匹配标题模式
        title_match = re.search(r'([^：\n]{2,30})[：:](.*?)(?:\n|$)', content[:500])
        if title_match:
            return title_match.group(0).strip()
        
        # 如果所有方法都失败，返回一个默认值
        return "东吴证券研究报告"
    
    def extract_publish_date(self, content: str) -> str:
        """
        提取发布日期
        
        Args:
            content: 研报内容
            
        Returns:
            发布日期，格式为YYYY-MM-DD
        """
        # 在文本前2000个字符中查找日期
        text_to_search = content[:2000]
        
        # 首先查找明确标记的日期
        date_markers = ["发布日期", "报告日期", "研究日期", "日期", "Date"]
        for marker in date_markers:
            marker_pattern = r'{}[：:]\s*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?'.format(marker)
            match = re.search(marker_pattern, text_to_search)
            if match:
                year, month, day = match.groups()
                # 确保月和日是两位数
                month = month.zfill(2) if len(month) == 1 else month
                day = day.zfill(2) if len(day) == 1 else day
                return f"{year}-{month}-{day}"
        
        # 如果没有明确标记，查找文本中的日期格式
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                year, month, day = matches[0]
                # 确保月和日是两位数
                month = month.zfill(2) if len(month) == 1 else month
                day = day.zfill(2) if len(day) == 1 else day
                return f"{year}-{month}-{day}"
        
        # 查找特定格式的日期，如"截止2025/1/24"
        special_date_match = re.search(r'截[至止].*?(\d{4})[/-年](\d{1,2})[/-月]?(\d{1,2})?[日号]?', text_to_search)
        if special_date_match:
            year, month, day = special_date_match.groups()
            # 如果日期不完整（可能只有年月），则设置为月底或当前日
            if day is None:
                # 如果只有年月，设为当月1日
                day = "01"
            month = month.zfill(2) if len(month) == 1 else month
            day = day.zfill(2) if len(day) == 1 else day
            return f"{year}-{month}-{day}"
            
        # 查找表格中的日期，如"收盘价截止2025/1/24"
        table_date_match = re.search(r'截[至止].*?(\d{4})[/-](\d{1,2})[/-](\d{1,2})', text_to_search)
        if table_date_match:
            year, month, day = table_date_match.groups()
            month = month.zfill(2) if len(month) == 1 else month
            day = day.zfill(2) if len(day) == 1 else day
            return f"{year}-{month}-{day}"
        
        # 如果没找到，返回今天的日期
        return datetime.datetime.now().strftime("%Y-%m-%d")
    
    def extract_investment_rating(self, content: str) -> str:
        """
        提取投资评级
        
        Args:
            content: 研报内容
            
        Returns:
            投资评级
        """
        # 查找评级关键词
        for rating, keywords in self.investment_ratings.items():
            for keyword in keywords:
                # 查找评级关键词周围的上下文
                pattern = r'([^。；，\n]*{}[^。；，\n]*)'.format(keyword)
                matches = re.findall(pattern, content)
                if matches:
                    return rating
        
        # 如果没有找到明确的评级，尝试查找"评级"周围的内容
        rating_context = re.findall(r'([^。；，\n]*评级[^。；，\n]*)', content)
        if rating_context:
            for context in rating_context:
                for rating, keywords in self.investment_ratings.items():
                    for keyword in keywords:
                        if keyword in context:
                            return rating
        
        return "未明确"
    
    def extract_driving_factors(self, content: str) -> List[str]:
        """
        提取主要驱动因素
        
        Args:
            content: 研报内容
            
        Returns:
            主要驱动因素列表
        """
        factors = []
        
        # 查找关键假设和驱动因素部分
        for keyword in self.driving_factors:
            # 查找关键词后的段落
            pattern = r'{}[：:](.*?)(?:(?:\n\n)|(?:风险提示)|(?:股价催化剂)|(?:有别于大众的认识))'.format(keyword)
            matches = re.findall(pattern, content, re.DOTALL)
            
            if matches:
                # 处理找到的内容
                for match in matches:
                    # 按数字或分隔符拆分为多个因素
                    items = re.split(r'(\d+[\.、]|；|;)', match.strip())
                    # 过滤空项并清理
                    items = [item.strip() for item in items if item.strip() and not re.match(r'^\d+[\.、]$', item)]
                    factors.extend(items)
        
        # 如果没有找到结构化的驱动因素，尝试从投资建议部分提取
        if not factors:
            investment_section = re.findall(r'投资建议[：:](.*?)(?:(?:\n\n)|(?:关键假设)|(?:风险提示))', content, re.DOTALL)
            if investment_section:
                # 取投资建议的前3句话作为驱动因素
                sentences = re.split(r'[。；;]', investment_section[0])
                factors = [s.strip() for s in sentences[:3] if s.strip()]
        
        # 去重
        return list(dict.fromkeys(factors))
    
    def extract_risk_factors(self, content: str) -> List[str]:
        """
        提取风险因素
        
        Args:
            content: 研报内容
            
        Returns:
            风险因素列表
        """
        risks = []
        
        # 查找风险提示部分
        for keyword in self.risk_keywords:
            # 查找关键词后的段落
            pattern = r'{}[：:](.*?)(?:\n\n|$)'.format(keyword)
            matches = re.findall(pattern, content, re.DOTALL)
            
            if matches:
                # 处理找到的内容
                for match in matches:
                    # 按数字或分隔符拆分为多个风险
                    items = re.split(r'(\d+[\.、]|；|;)', match.strip())
                    # 过滤空项并清理
                    items = [item.strip() for item in items if item.strip() and not re.match(r'^\d+[\.、]$', item)]
                    risks.extend(items)
        
        # 去重
        return list(dict.fromkeys(risks))
    
    def extract_target_companies(self, content: str) -> List[Dict[str, str]]:
        """
        提取目标公司
        
        Args:
            content: 研报内容
            
        Returns:
            目标公司列表，包含代码和名称
        """
        companies = []
        
        # 查找股票代码和名称
        # 格式通常为: 600066.SH 宇通客车
        pattern = r'(\d{6})[\.。]([A-Z]{2})\s*([^\s\d\(（]+)'
        matches = re.findall(pattern, content)
        
        for code, market, name in matches:
            companies.append({
                "code": f"{code}.{market}",
                "name": name.strip()
            })
        
        # 如果没有找到，尝试在标题中查找公司名称
        if not companies:
            title = self.extract_title(content)
            # 查找常见的公司名称模式（中文名+科技/股份/集团等）
            company_names = re.findall(r'([^\s，。:：]{2,6}(?:科技|股份|集团|证券|公司|电子|能源|医药|软件|网络))', title)
            for name in company_names:
                companies.append({
                    "code": "未知",
                    "name": name
                })
        
        return companies
    
    def extract_dongwu_features(self, content: str) -> Dict[str, Any]:
        """
        提取东吴证券研报的风格特征
        
        Args:
            content: 研报内容
            
        Returns:
            风格特征字典
        """
        # 计算研报总长度
        total_length = len(content)
        
        # 计算段落数量和平均长度
        paragraphs = [p for p in re.split(r'\n{2,}', content) if p.strip()]
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # 计算数字密度（数字字符占比）
        digit_count = len(re.findall(r'\d', content))
        digit_density = digit_count / total_length if total_length > 0 else 0
        
        # 计算标点符号密度
        punctuation_count = len(re.findall(r'[，。；：""（）、？！]', content))
        punctuation_density = punctuation_count / total_length if total_length > 0 else 0
        
        # 检测是否包含特定章节结构
        has_investment_advice = bool(re.search(r'投资建议[：:]', content))
        has_key_assumptions = bool(re.search(r'关键假设[：:]', content))
        has_risk_tips = bool(re.search(r'风险提示[：:]', content))
        has_price_catalyst = bool(re.search(r'股价催化剂[：:]', content))
        has_different_views = bool(re.search(r'有别于大众的认识[：:]', content))
        
        # 检测是否包含表格
        has_tables = bool(re.search(r'表\s*\d+[：:]', content))
        
        # 检测是否包含图表
        has_charts = bool(re.search(r'图\s*\d+[：:]', content))
        
        # 检测是否包含目录
        has_toc = bool(re.search(r'(内容目录|目录)', content[:1000]))
        
        # 提取投资评级
        investment_rating = self.extract_investment_rating(content)
        
        # 提取主要驱动因素
        driving_factors = self.extract_driving_factors(content)
        
        # 提取风险因素
        risk_factors = self.extract_risk_factors(content)
        
        # 提取目标公司
        target_companies = self.extract_target_companies(content)
        
        # 返回东吴证券研报的风格特征
        return {
            "text_stats": {
                "total_length": total_length,
                "paragraph_count": len(paragraphs),
                "avg_paragraph_length": avg_paragraph_length,
                "digit_density": digit_density,
                "punctuation_density": punctuation_density
            },
            "structure_features": {
                "has_investment_advice": has_investment_advice,
                "has_key_assumptions": has_key_assumptions,
                "has_risk_tips": has_risk_tips,
                "has_price_catalyst": has_price_catalyst,
                "has_different_views": has_different_views,
                "has_tables": has_tables,
                "has_charts": has_charts,
                "has_toc": has_toc
            },
            "content_features": {
                "investment_rating": investment_rating,
                "driving_factors": driving_factors,
                "risk_factors": risk_factors,
                "target_companies": target_companies
            }
        }
    
    def convert_single_file(self, txt_path: Path) -> Dict[str, Any]:
        """
        转换单个TXT文件为JSON格式
        
        Args:
            txt_path: TXT文件路径
            
        Returns:
            JSON数据字典
        """
        try:
            # 读取TXT文件内容
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 清理内容
            cleaned_content = self.clean_content(content)
            
            # 提取元数据
            metadata = self.extract_metadata_from_filename(txt_path.name)
            
            # 提取标题
            title = self.extract_title(cleaned_content)
            
            # 提取发布日期
            publish_date = self.extract_publish_date(cleaned_content)
            
            # 提取东吴证券研报的风格特征
            dongwu_features = self.extract_dongwu_features(cleaned_content)
            
            # 构建JSON数据
            json_data = {
                "report_id": metadata["report_id"],
                "metadata": {
                    "broker": metadata["broker"],
                    "title": title,
                    "publish_date": publish_date,
                    "input_date": metadata["input_date"],
                    "source_file": txt_path.name
                },
                "content": {
                    "full_text": cleaned_content
                },
                "dongwu_style_features": dongwu_features
            }
            
            # 保存JSON文件
            output_path = Path(self.output_dir) / f"{txt_path.stem}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功转换: {txt_path.name} -> {output_path.name}")
            return json_data
            
        except Exception as e:
            print(f"转换失败 {txt_path.name}: {str(e)}")
            return {}
    
    def batch_convert(self, txt_dir: str, pattern: str = "*.txt") -> Tuple[List[str], List[str]]:
        """
        批量转换TXT文件为JSON格式
        
        Args:
            txt_dir: TXT文件目录
            pattern: 文件匹配模式
            
        Returns:
            成功和失败的文件列表
        """
        txt_dir_path = Path(txt_dir)
        txt_files = list(txt_dir_path.glob(pattern))
        
        if not txt_files:
            print(f"在 {txt_dir} 中没有找到 {pattern} 文件")
            return [], []
        
        converted = []
        failed = []
        
        for txt_file in txt_files:
            try:
                self.convert_single_file(txt_file)
                converted.append(str(txt_file))
            except Exception as e:
                print(f"转换失败 {txt_file.name}: {str(e)}")
                failed.append(str(txt_file))
        
        # 生成转换报告
        self.generate_conversion_report(converted, failed)
        
        return converted, failed
    
    def generate_conversion_report(self, converted: List[str], failed: List[str]) -> None:
        """
        生成转换报告
        
        Args:
            converted: 成功转换的文件列表
            failed: 转换失败的文件列表
        """
        report = {
            "conversion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(converted) + len(failed),
            "success_count": len(converted),
            "failure_count": len(failed),
            "success_rate": len(converted) / (len(converted) + len(failed)) if (len(converted) + len(failed)) > 0 else 0,
            "converted_files": [Path(f).name for f in converted],
            "failed_files": [Path(f).name for f in failed]
        }
        
        # 保存报告
        report_path = Path(self.output_dir) / "conversion_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n转换报告:")
        print(f"总文件数: {report['total_files']}")
        print(f"成功转换: {report['success_count']}")
        print(f"转换失败: {report['failure_count']}")
        print(f"成功率: {report['success_rate']*100:.2f}%")
        print(f"报告已保存至: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将东吴证券研报TXT文件转换为JSON格式")
    parser.add_argument("--txt_dir", type=str, default="user_data/txt_files", help="TXT文件目录")
    parser.add_argument("--output_dir", type=str, default="user_data/json_files", help="JSON输出目录")
    parser.add_argument("--pattern", type=str, default="*.txt", help="文件匹配模式")
    parser.add_argument("--single", type=str, help="单个TXT文件路径")
    
    args = parser.parse_args()
    
    converter = TxtToJsonConverter(output_dir=args.output_dir)
    
    if args.single:
        converter.convert_single_file(Path(args.single))
    else:
        converter.batch_convert(args.txt_dir, args.pattern) 
#!/usr/bin/env python3
"""
东吴证券研报数据解析完整流程演示
标准化的数据处理pipeline，适用于批量处理
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools"))

def print_step_header(step_num: int, step_name: str):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"📊 步骤 {step_num}: {step_name}")
    print(f"{'='*60}")

def print_result_summary(data: Dict[str, Any]):
    """打印处理结果摘要"""
    basic_info = data.get('basic_info', {})
    core_content = data.get('core_content', {})
    stocks = data.get('recommended_stocks', [])
    
    print(f"\n✅ **解析结果摘要**:")
    print(f"   📋 主题: {basic_info.get('theme', '未识别')[:50]}...")
    print(f"   📊 类型: {basic_info.get('type', '未识别')}")
    print(f"   📅 日期: {basic_info.get('publish_date', '未识别')}")
    print(f"   💡 建议: {basic_info.get('investment_advice', '未识别')}")
    print(f"   🔄 变化: {basic_info.get('has_changes', '未识别')}")
    print(f"   📈 推荐股票: {len(stocks)} 只")

def extract_key_stocks(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """从解析结果中提取关键股票信息"""
    content = data.get('core_content', {}).get('investment_highlights', '')
    
    # 简单的股票名称提取（后续可以优化）
    stock_keywords = [
        "三花智控", "宇通客车", "杰瑞股份", "道通科技", "蓝思科技",
        "华秦科技", "上海港湾", "瀚蓝环境", "安克创新", "亚香股份"
    ]
    
    found_stocks = []
    for i, stock in enumerate(stock_keywords, 1):
        if stock in content:
            # 提取板块信息
            sector = "未分类"
            if "电新" in content and stock == "三花智控":
                sector = "电新"
            elif "汽车" in content and stock == "宇通客车":
                sector = "汽车"
            elif "机械" in content and stock == "杰瑞股份":
                sector = "机械"
            elif "科技" in content and stock == "道通科技":
                sector = "海外科技"
            elif "电子" in content and stock == "蓝思科技":
                sector = "电子"
            elif "军工" in content and stock == "华秦科技":
                sector = "军工"
            elif "建材" in content and stock == "上海港湾":
                sector = "建筑建材"
            elif "环保" in content and stock == "瀚蓝环境":
                sector = "环保公用"
            elif "商社" in content and stock == "安克创新":
                sector = "商社"
            elif "化工" in content and stock == "亚香股份":
                sector = "能源化工"
            
            found_stocks.append({
                "序号": f"1.{i}",
                "板块": sector,
                "股票名称": stock,
                "推荐理由": f"详见{stock}投资建议"
            })
    
    return found_stocks

def complete_data_pipeline():
    """完整的数据解析流程演示"""
    
    print("🚀 东吴证券研报数据解析完整流程演示")
    print("="*60)
    print("📋 流程说明：输入原始JSON → 格式修复 → 内容解析 → 结构化输出")
    print("🎯 目标：建立标准化数据处理pipeline，支持批量处理")
    
    # 步骤1: 读取原始数据
    print_step_header(1, "读取原始研报数据")
    
    input_file = project_root / "user_data" / "json_files" / "dongwu_complete_fixed.json"
    if not input_file.exists():
        print(f"❌ 错误：输入文件不存在: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✅ 成功读取原始数据")
    print(f"   📄 文件: {input_file.name}")
    print(f"   📊 数据大小: {len(str(raw_data))} 字符")
    print(f"   🏢 券商: {raw_data.get('metadata', {}).get('broker', '未知')}")
    
    # 步骤2: 使用增强版预处理器解析
    print_step_header(2, "智能内容解析")
    
    try:
        from enhanced_report_preprocessor import EnhancedReportPreprocessor
        
        preprocessor = EnhancedReportPreprocessor()
        print("✅ 预处理器初始化成功")
        
        # 处理数据
        processed_data = preprocessor.process_report(raw_data)
        print("✅ 数据解析完成")
        
        # 显示解析结果摘要
        print_result_summary(processed_data)
        
    except Exception as e:
        print(f"❌ 解析过程出错: {str(e)}")
        return
    
    # 步骤3: 数据优化和清理
    print_step_header(3, "数据优化和清理")
    
    # 提取关键股票信息
    key_stocks = extract_key_stocks(processed_data)
    processed_data['key_stocks'] = key_stocks
    
    print(f"✅ 提取到 {len(key_stocks)} 只关键股票")
    for stock in key_stocks:
        print(f"   📈 {stock['股票名称']} ({stock['板块']})")
    
    # 优化主题提取（截取前100字符）
    if 'basic_info' in processed_data and 'theme' in processed_data['basic_info']:
        theme = processed_data['basic_info']['theme']
        if len(theme) > 100:
            processed_data['basic_info']['theme'] = theme[:100] + "..."
    
    print("✅ 数据清理完成")
    
    # 步骤4: 生成标准化输出
    print_step_header(4, "生成标准化输出")
    
    # 创建最终输出结构
    final_output = {
        "report_metadata": {
            "report_id": processed_data.get('report_id'),
            "source_broker": "东吴证券",
            "processing_time": datetime.now().isoformat(),
            "pipeline_version": "1.0",
            "data_quality_score": calculate_quality_score(processed_data)
        },
        "extracted_data": {
            "basic_info": processed_data.get('basic_info', {}),
            "core_content": processed_data.get('core_content', {}),
            "key_stocks": key_stocks,
            "risk_factors": extract_risk_factors(processed_data),
            "financial_metrics": extract_financial_metrics(processed_data)
        },
        "processing_status": {
            "format_fixed": True,
            "content_extracted": True,
            "stocks_identified": len(key_stocks) > 0,
            "ready_for_analysis": True
        }
    }
    
    # 保存标准化输出
    output_file = project_root / "user_data" / "json_files" / "dongwu_standardized_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 标准化输出已保存: {output_file.name}")
    
    # 步骤5: 生成处理报告
    print_step_header(5, "生成处理报告")
    
    generate_processing_report(final_output)
    
    print(f"\n🎉 **数据解析流程完成！**")
    print(f"📊 质量评分: {final_output['report_metadata']['data_quality_score']}/100")
    print(f"📁 输出文件: {output_file}")
    print(f"📈 可分析状态: {'是' if final_output['processing_status']['ready_for_analysis'] else '否'}")

def calculate_quality_score(data: Dict[str, Any]) -> int:
    """计算数据质量评分"""
    score = 0
    
    # 基本信息完整性 (40分)
    basic_info = data.get('basic_info', {})
    if basic_info.get('theme'): score += 10
    if basic_info.get('type'): score += 10
    if basic_info.get('investment_advice'): score += 10
    if basic_info.get('publish_date'): score += 10
    
    # 核心内容质量 (40分)
    core_content = data.get('core_content', {})
    content_length = len(str(core_content))
    if content_length > 1000: score += 15
    elif content_length > 500: score += 10
    elif content_length > 200: score += 5
    
    if core_content.get('investment_highlights'): score += 10
    if core_content.get('risk_warnings'): score += 10
    if core_content.get('financial_forecast'): score += 5
    
    # 结构化程度 (20分)
    if data.get('recommended_stocks'): score += 10
    if len(str(data)) > 5000: score += 10  # 丰富度
    
    return min(score, 100)

def extract_risk_factors(data: Dict[str, Any]) -> List[str]:
    """提取风险因素"""
    risk_text = data.get('core_content', {}).get('risk_warnings', '')
    
    # 简单的风险因素提取
    risk_keywords = [
        "市场风险", "政策风险", "技术风险", "经济风险", 
        "竞争风险", "汇率风险", "流动性风险"
    ]
    
    found_risks = []
    for risk in risk_keywords:
        if risk in risk_text:
            found_risks.append(risk)
    
    return found_risks

def extract_financial_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """提取财务指标"""
    forecast_text = data.get('core_content', {}).get('financial_forecast', '')
    
    import re
    
    metrics = {}
    
    # 提取PE值
    pe_matches = re.findall(r'PE[^0-9]*(\d+\.?\d*)', forecast_text)
    if pe_matches:
        metrics['pe_ratios'] = pe_matches
    
    # 提取增长率
    growth_matches = re.findall(r'增长[^0-9]*(\d+\.?\d*)%', forecast_text)
    if growth_matches:
        metrics['growth_rates'] = growth_matches
    
    # 提取收入
    revenue_matches = re.findall(r'收入[^0-9]*(\d+\.?\d*)亿', forecast_text)
    if revenue_matches:
        metrics['revenue_billions'] = revenue_matches
    
    return metrics

def generate_processing_report(data: Dict[str, Any]):
    """生成处理报告"""
    
    report = f"""
📊 **数据处理报告**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **基本信息**
   • 报告ID: {data['report_metadata']['report_id']}
   • 处理时间: {data['report_metadata']['processing_time']}
   • 数据质量: {data['report_metadata']['data_quality_score']}/100

📈 **内容摘要**
   • 主题: {data['extracted_data']['basic_info'].get('theme', '未识别')[:50]}...
   • 类型: {data['extracted_data']['basic_info'].get('type', '未识别')}
   • 投资建议: {data['extracted_data']['basic_info'].get('investment_advice', '未识别')}

📊 **结构化数据**
   • 关键股票: {len(data['extracted_data']['key_stocks'])} 只
   • 风险因素: {len(data['extracted_data']['risk_factors'])} 项
   • 财务指标: {len(data['extracted_data']['financial_metrics'])} 类

✅ **处理状态**
   • 格式修复: {'✅' if data['processing_status']['format_fixed'] else '❌'}
   • 内容提取: {'✅' if data['processing_status']['content_extracted'] else '❌'}
   • 股票识别: {'✅' if data['processing_status']['stocks_identified'] else '❌'}
   • 分析就绪: {'✅' if data['processing_status']['ready_for_analysis'] else '❌'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    print(report)
    
    # 保存报告到文件
    report_file = project_root / "user_data" / "results" / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 处理报告已保存: {report_file}")

if __name__ == "__main__":
    try:
        complete_data_pipeline()
    except Exception as e:
        print(f"❌ 流程执行出错: {str(e)}")
        import traceback
        traceback.print_exc() 
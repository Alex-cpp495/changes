#!/usr/bin/env python3
"""
批量处理研报脚本
支持导入和分析大量研报数据
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data_processing.processors.report_processor import get_report_processor
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量处理研报数据')
    
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('--format', choices=['json', 'csv', 'excel', 'txt', 'auto'], 
                       default='auto', help='数据格式')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='批处理大小')
    parser.add_argument('--output-dir', default='data/results', 
                       help='输出目录')
    parser.add_argument('--export-format', choices=['json', 'csv'], 
                       default='json', help='导出格式')
    parser.add_argument('--config-file', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        print("🚀 开始批量处理研报数据")
        print(f"📁 输入文件: {args.input_file}")
        print(f"📊 数据格式: {args.format}")
        print(f"⚙️ 批处理大小: {args.batch_size}")
        
        # 加载配置
        config = {'batch_size': args.batch_size}
        if args.config_file:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                config.update(user_config)
        
        # 初始化处理器
        processor = get_report_processor(config)
        
        # 批量处理
        results = await processor.process_batch_data(
            data_source=args.input_file,
            data_format=args.format
        )
        
        # 显示处理结果
        print("\n" + "="*60)
        print("📊 处理结果摘要")
        print("="*60)
        
        summary = results['summary']
        print(f"📈 总计处理: {summary['total_processed']} 篇")
        print(f"✅ 成功处理: {summary['successful']} 篇")
        print(f"❌ 处理失败: {summary['failed']} 篇")
        print(f"⚠️ 异常报告: {summary['anomalous']} 篇")
        print(f"✔️ 正常报告: {summary['normal']} 篇")
        print(f"⏱️ 处理耗时: {summary['processing_time']:.2f} 秒")
        
        if 'reports_per_second' in summary:
            print(f"🚄 处理速度: {summary['reports_per_second']:.2f} 篇/秒")
        
        if 'anomaly_rate' in summary:
            print(f"📊 异常率: {summary['anomaly_rate']:.2%}")
        
        # 异常级别分布
        if results['anomaly_distribution']:
            print(f"\n📈 异常级别分布:")
            for level, count in results['anomaly_distribution'].items():
                print(f"   {level}: {count} 篇")
        
        # 性能统计
        if results['performance_stats']:
            perf = results['performance_stats']
            print(f"\n⚡ 性能统计:")
            print(f"   平均处理时间: {perf['avg_processing_time']:.3f} 秒")
            print(f"   最快处理时间: {perf['min_processing_time']:.3f} 秒")
            print(f"   最慢处理时间: {perf['max_processing_time']:.3f} 秒")
        
        # 建议
        if results['recommendations']:
            print(f"\n💡 处理建议:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # 导出结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"batch_results_{timestamp}.{args.export_format}"
        
        processor.export_results(results, str(output_file), args.export_format)
        
        print(f"\n💾 结果已保存到: {output_file}")
        
        # 显示异常报告列表
        anomalous_reports = [
            r for r in results['detailed_results'] 
            if r['status'] == 'success' and r.get('anomaly_result', {}).get('is_anomalous', False)
        ]
        
        if anomalous_reports:
            print(f"\n⚠️ 发现 {len(anomalous_reports)} 篇异常报告:")
            for report in anomalous_reports[:10]:  # 只显示前10个
                anomaly_result = report['anomaly_result']
                print(f"   • {report['title']} (异常分数: {anomaly_result['overall_anomaly_score']:.3f})")
            
            if len(anomalous_reports) > 10:
                print(f"   ... 还有 {len(anomalous_reports) - 10} 篇（详见导出文件）")
        
        print("\n🎉 批量处理完成！")
        
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        logger.error(f"批量处理失败: {e}", exc_info=True)
        sys.exit(1)


def create_sample_data():
    """创建示例数据文件"""
    sample_data = [
        {
            "report_id": "sample_001",
            "title": "某科技公司Q3财报分析",
            "content": """
            某科技公司2023年第三季度财务报告显示，公司营业收入达到15.2亿元，
            同比增长28.5%。但值得注意的是，公司现金流量出现异常：
            
            - 经营活动现金流量为负3.8亿元，与营收增长趋势严重不符
            - 应收账款大幅增加至12.5亿元，占营收比例达82%
            - 存货周转率仅为2.1次，远低于行业平均水平
            - 应付账款周转天数异常延长至180天
            
            此外，公司在本季度突然宣布大额收购计划，但未提供详细的尽职调查报告。
            审计师对部分关联交易的会计处理提出了关注意见。
            """,
            "company": "某科技公司",
            "industry": "科技",
            "report_date": "2023-10-31",
            "analyst": "张三"
        },
        {
            "report_id": "sample_002", 
            "title": "传统制造企业转型报告",
            "content": """
            某传统制造企业2023年业绩稳健，营业收入8.6亿元，同比增长5.2%。
            公司财务指标表现良好：
            
            - 经营活动现金流量为正2.1亿元，现金流健康
            - 应收账款周转率稳定在6.5次
            - 毛利率保持在22.8%的合理水平
            - 负债率控制在45%以内
            
            公司正在推进数字化转型，研发投入占营收比例提升至4.2%。
            管理层表示将继续专注主业发展，审慎考虑投资项目。
            """,
            "company": "某制造企业",
            "industry": "制造业", 
            "report_date": "2023-10-31",
            "analyst": "李四"
        }
    ]
    
    # 保存示例数据
    sample_file = Path("data/sample_reports.json")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 示例数据已创建: {sample_file}")
    return sample_file


if __name__ == "__main__":
    # 检查是否需要创建示例数据
    if len(sys.argv) == 1:
        print("📋 使用方法:")
        print("python scripts/batch_process_reports.py <输入文件> [选项]")
        print("\n📝 选项:")
        print("  --format: 数据格式 (json/csv/excel/txt/auto)")
        print("  --batch-size: 批处理大小 (默认: 10)")
        print("  --output-dir: 输出目录 (默认: data/results)")
        print("  --export-format: 导出格式 (json/csv)")
        print("\n🔧 创建示例数据并运行:")
        
        choice = input("是否创建示例数据并运行演示? (y/N): ")
        if choice.lower() == 'y':
            sample_file = create_sample_data()
            sys.argv = ['batch_process_reports.py', str(sample_file)]
        else:
            sys.exit(0)
    
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
100篇研报批量处理演示脚本
演示系统处理大量真实数据的能力
"""

import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data_processing.processors.report_processor import get_report_processor
from src.optimization.performance_optimizer import get_performance_optimizer

def create_sample_reports_data(count=100):
    """创建示例研报数据"""
    print(f"🎯 创建 {count} 篇示例研报数据...")
    
    # 基础模板
    normal_template = """
    公司2023年第{quarter}季度财务报告显示，公司营业收入达到{revenue}亿元，
    同比增长{growth}%。财务指标表现{performance}：
    
    - 经营活动现金流量为{cash_flow}亿元
    - 应收账款周转率稳定在{receivables_ratio}次
    - 毛利率保持在{margin}%的合理水平
    - 负债率控制在{debt_ratio}%以内
    
    {additional_info}
    """
    
    anomaly_template = """
    公司2023年第{quarter}季度财务报告显示，公司营业收入达到{revenue}亿元，
    同比增长{growth}%。但值得注意的是，存在以下异常情况：
    
    - 经营活动现金流量为负{cash_flow}亿元，与营收增长趋势严重不符
    - 应收账款大幅增加至{receivables}亿元，占营收比例达{receivables_ratio}%
    - {anomaly_type}
    - 应付账款周转天数异常延长至{payable_days}天
    
    {risk_info}
    """
    
    reports = []
    
    for i in range(count):
        is_anomaly = i % 4 == 0  # 25%的异常率
        
        if is_anomaly:
            # 异常报告
            report = {
                "report_id": f"report_{i+1:03d}",
                "title": f"某{'科技' if i % 2 == 0 else '制造'}公司Q{(i%4)+1}财报分析",
                "content": anomaly_template.format(
                    quarter=(i%4)+1,
                    revenue=round(5 + i * 0.3, 1),
                    growth=round(15 + i * 0.5, 1),
                    cash_flow=round(1 + i * 0.1, 1),
                    receivables=round(8 + i * 0.2, 1),
                    receivables_ratio=round(70 + i * 2, 0),
                    anomaly_type=[
                        "存货周转率仅为2.1次，远低于行业平均水平",
                        "关联交易金额异常增大，缺乏合理解释",
                        "研发费用突然大幅下降，引发关注",
                        "重要子公司业绩严重下滑，未充分披露"
                    ][i % 4],
                    payable_days=round(120 + i * 3, 0),
                    risk_info=[
                        "审计师对部分会计处理提出了关注意见。",
                        "公司突然宣布大额收购计划，但未提供详细说明。",
                        "管理层变动频繁，内控制度有待完善。",
                        "行业政策变化对公司未来经营存在不确定性。"
                    ][i % 4]
                ),
                "company": f"示例公司{i+1}",
                "industry": "科技" if i % 2 == 0 else "制造业",
                "report_date": f"2023-{((i%4)+1)*3:02d}-30",
                "analyst": f"分析师{(i%10)+1}",
                "source": "demo_data"
            }
        else:
            # 正常报告
            report = {
                "report_id": f"report_{i+1:03d}",
                "title": f"某{'金融' if i % 3 == 0 else '能源'}企业Q{(i%4)+1}业绩报告",
                "content": normal_template.format(
                    quarter=(i%4)+1,
                    revenue=round(8 + i * 0.2, 1),
                    growth=round(3 + i * 0.3, 1),
                    performance="良好",
                    cash_flow=round(2 + i * 0.05, 1),
                    receivables_ratio=round(5 + i * 0.1, 1),
                    margin=round(20 + i * 0.2, 1),
                    debt_ratio=round(40 + i * 0.3, 0),
                    additional_info=[
                        "公司正在推进数字化转型，研发投入占营收比例提升。",
                        "管理层表示将继续专注主业发展，审慎考虑投资项目。",
                        "公司积极履行社会责任，ESG评级持续提升。",
                        "未来将加大市场拓展力度，提升品牌影响力。"
                    ][i % 4]
                ),
                "company": f"示例企业{i+1}",
                "industry": "金融" if i % 3 == 0 else "能源",
                "report_date": f"2023-{((i%4)+1)*3:02d}-30",
                "analyst": f"分析师{(i%10)+1}",
                "source": "demo_data"
            }
        
        reports.append(report)
    
    return reports

async def demo_batch_processing():
    """演示批量处理"""
    print("🚀 东吴证券研报异常检测系统 - 100篇研报批量处理演示\n")
    
    # 创建演示数据
    reports_data = create_sample_reports_data(100)
    
    # 保存到文件供后续使用
    data_file = project_root / "data" / "demo_100_reports.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(reports_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 演示数据已保存到: {data_file}")
    
    # 初始化性能优化器
    print("\n⚡ 初始化性能优化器...")
    optimizer = get_performance_optimizer({
        'max_concurrent': 20,
        'cpu_warning': 75.0,
        'memory_warning': 80.0
    })
    
    optimizer.start_monitoring(interval=0.5)
    optimizer.optimize_for_batch_processing()
    
    # 初始化批量处理器
    print("🔧 初始化批量处理器...")
    processor = get_report_processor({
        'batch_size': 15,  # 每批处理15篇
    })
    
    print(f"\n📊 开始批量处理 {len(reports_data)} 篇研报...")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 执行批量处理
        results = await processor.process_batch_data(
            data_source=reports_data,
            data_format='list'
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 显示处理结果
        print("\n" + "="*60)
        print("📈 处理结果摘要")
        print("="*60)
        
        summary = results['summary']
        print(f"📋 总计处理: {summary['total_processed']} 篇")
        print(f"✅ 成功处理: {summary['successful']} 篇")
        print(f"❌ 处理失败: {summary['failed']} 篇")
        print(f"⚠️ 检出异常: {summary['anomalous']} 篇")
        print(f"✔️ 正常报告: {summary['normal']} 篇")
        print(f"⏱️ 处理耗时: {total_time:.2f} 秒")
        print(f"🚄 处理速度: {summary['total_processed']/total_time:.2f} 篇/秒")
        print(f"📊 异常检出率: {summary['anomalous']/summary['successful']:.2%}")
        
        # 异常级别分布
        if results['anomaly_distribution']:
            print(f"\n📈 异常级别分布:")
            for level, count in results['anomaly_distribution'].items():
                percentage = count / summary['successful'] * 100
                print(f"   {level}: {count} 篇 ({percentage:.1f}%)")
        
        # 性能统计
        if results['performance_stats']:
            perf = results['performance_stats']
            print(f"\n⚡ 性能统计:")
            print(f"   平均处理时间: {perf['avg_processing_time']:.3f} 秒/篇")
            print(f"   最快处理时间: {perf['min_processing_time']:.3f} 秒")
            print(f"   最慢处理时间: {perf['max_processing_time']:.3f} 秒")
        
        # 系统性能
        print(f"\n🖥️ 系统性能:")
        perf_report = optimizer.get_performance_report()
        if 'cpu_percent' in perf_report:
            print(f"   CPU使用率: {perf_report['cpu_percent']['current']:.1f}%")
        if 'memory_percent' in perf_report:
            print(f"   内存使用率: {perf_report['memory_percent']['current']:.1f}%")
        
        # 导出结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = project_root / "data" / "results" / f"demo_results_{timestamp}.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        processor.export_results(results, str(export_file), 'json')
        
        print(f"\n💾 详细结果已保存到: {export_file}")
        
        # 显示前5个异常报告
        anomalous_reports = [
            r for r in results['detailed_results']
            if r['status'] == 'success' and r.get('anomaly_result', {}).get('is_anomalous', False)
        ]
        
        if anomalous_reports:
            print(f"\n⚠️ 前5个异常报告示例:")
            for i, report in enumerate(anomalous_reports[:5], 1):
                anomaly_result = report['anomaly_result']
                print(f"   {i}. {report['title']}")
                print(f"      异常分数: {anomaly_result['overall_anomaly_score']:.3f}")
                print(f"      异常级别: {anomaly_result['overall_anomaly_level']}")
                print(f"      置信度: {anomaly_result['confidence']:.3f}")
        
        # 优化建议
        recommendations = optimizer.get_optimization_recommendations()
        if recommendations:
            print(f"\n💡 系统优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎉 演示完成！系统成功处理了100篇研报数据")
        print(f"🔗 您可以通过以下方式使用系统:")
        print(f"   • 批量处理脚本: python scripts/batch_process_reports.py {data_file}")
        print(f"   • Web界面: python start_system.py")
        print(f"   • API接口: 启动Web服务后访问 /docs 查看API文档")
        
    except Exception as e:
        print(f"\n❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止性能监控
        optimizer.stop_monitoring()
        print(f"\n🛑 性能监控已停止")

if __name__ == "__main__":
    asyncio.run(demo_batch_processing()) 
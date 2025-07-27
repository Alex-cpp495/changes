"""
东吴证券研报异常检测系统使用示例

演示如何使用四层异常检测体系对研报进行综合异常检测
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.anomaly_detection.ensemble_detector import get_ensemble_detector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_report_data():
    """
    创建示例研报数据
    
    Returns:
        示例研报数据列表
    """
    sample_reports = [
        {
            'text': """
            东吴证券研报：某科技股投资价值分析
            
            基于我们的深度调研，该公司Q3业绩超预期增长35%，营收达到85亿元，
            净利润12.3亿元，同比增长28%。公司在AI芯片领域的技术突破值得关注。
            
            核心投资逻辑：
            1. 技术护城河深厚，AI芯片性能领先同行20%
            2. 下游需求旺盛，订单饱满至明年Q2
            3. 毛利率持续提升，达到43.2%
            4. 管理层执行力强，战略清晰
            
            风险提示：
            1. 行业竞争加剧风险
            2. 技术迭代风险
            3. 下游需求波动风险
            
            综合考虑，我们维持"买入"评级，目标价65元，较当前价格有30%上涨空间。
            """,
            'publication_time': datetime.now() - timedelta(days=1),
            'author': '张三',
            'stock_codes': ['000001'],
            'rating': '买入',
            'target_prices': {'000001': 65.0},
            'industry': '科技',
            'publication_date': datetime.now() - timedelta(days=1)
        },
        {
            'text': """
            紧急调研报告：某科技股重大变化
            
            据可靠消息，该公司即将发布革命性产品，预计将带来巨大变革。
            内部人士透露，新产品性能超越市场预期数倍。
            
            基于最新获得的独家信息，我们预计：
            - Q4营收将暴涨200%
            - 明年净利润增长500%
            - 股价有望翻倍
            
            但是我们同时注意到，公司近期面临监管压力，业务发展存在不确定性。
            财务数据显示盈利能力下降，现金流紧张。
            
            综合判断，我们给予"强烈推荐"评级，目标价120元。
            同时提醒投资者注意风险，建议谨慎投资。
            """,
            'publication_time': datetime.now(),
            'author': '李四',
            'stock_codes': ['000001'],
            'rating': '强烈推荐',
            'target_prices': {'000001': 120.0},
            'industry': '科技',
            'publication_date': datetime.now()
        }
    ]
    
    return sample_reports


def create_sample_market_data(detector):
    """
    创建示例市场数据
    
    Args:
        detector: 集成检测器实例
    """
    # 添加示例市场数据
    base_date = datetime.now() - timedelta(days=30)
    base_price = 50.0
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        
        # 模拟价格波动
        price_change = (i - 15) * 0.5 + np.random.normal(0, 1)
        close_price = base_price + price_change
        open_price = close_price + np.random.normal(0, 0.5)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
        volume = int(10000000 + np.random.normal(0, 2000000))
        
        detector.market_detector.add_market_data(
            stock_code='000001',
            date=date,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            volume=max(volume, 1000000)  # 确保成交量为正
        )


def create_sample_market_events(detector):
    """
    创建示例市场事件
    
    Args:
        detector: 集成检测器实例
    """
    # 添加市场事件
    detector.behavioral_detector.add_market_event(
        event_time=datetime.now() - timedelta(hours=6),
        event_type='earnings',
        description='科技板块财报季开始',
        affected_stocks=['000001']
    )
    
    detector.behavioral_detector.add_market_event(
        event_time=datetime.now() - timedelta(days=2),
        event_type='policy',
        description='AI产业政策利好发布',
        affected_stocks=['000001']
    )


def demonstrate_single_detection():
    """
    演示单个研报的异常检测
    """
    print("\n" + "="*80)
    print("东吴证券研报异常检测系统演示")
    print("="*80)
    
    # 获取集成检测器
    detector = get_ensemble_detector()
    
    # 创建示例数据
    import numpy as np
    create_sample_market_data(detector)
    create_sample_market_events(detector)
    
    # 获取示例研报
    sample_reports = create_sample_report_data()
    
    print(f"\n📊 开始检测 {len(sample_reports)} 个研报样本")
    
    for i, report in enumerate(sample_reports):
        print(f"\n{'='*60}")
        print(f"📄 研报 {i+1}: {report['author']} - {report['rating']}")
        print(f"📅 发布时间: {report['publication_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"📈 股票代码: {', '.join(report['stock_codes'])}")
        print(f"📝 文本长度: {len(report['text'])} 字符")
        
        # 执行异常检测
        result = detector.detect_anomalies(report)
        
        # 显示检测结果
        print(f"\n🔍 异常检测结果:")
        print(f"   总体异常分数: {result['overall_anomaly_score']:.3f}")
        print(f"   异常等级: {result['anomaly_level']}")
        print(f"   是否异常: {'是' if result['is_anomaly'] else '否'}")
        
        # 显示各层检测结果
        print(f"\n📋 分层检测分数:")
        for layer, score in result['layer_scores'].items():
            emoji = "⚠️" if score > 0.5 else "✅" if score < 0.3 else "⚡"
            print(f"   {emoji} {layer:12}: {score:.3f}")
        
        # 显示异常层
        if result['layer_anomalies']:
            print(f"\n🚨 检测到异常的层级:")
            for layer in result['layer_anomalies']:
                print(f"   • {layer}")
        
        # 显示异常模式
        pattern = result['anomaly_pattern']
        if pattern['pattern_description'] != '未检测到显著异常':
            print(f"\n🔍 异常模式分析:")
            print(f"   主要异常类型: {pattern['primary_anomaly_type']}")
            print(f"   风险等级: {pattern['risk_level']}")
            print(f"   模式描述: {pattern['pattern_description']}")
        
        # 显示建议
        if result['recommendations']:
            print(f"\n💡 处理建议:")
            for rec in result['recommendations'][:3]:  # 显示前3个建议
                print(f"   • {rec}")
        
        # 显示详细错误（如果有）
        if 'detection_errors' in result:
            print(f"\n⚠️  检测过程中的错误:")
            for error in result['detection_errors']:
                print(f"   • {error}")


def demonstrate_batch_detection():
    """
    演示批量异常检测
    """
    print(f"\n{'='*60}")
    print("📊 批量异常检测演示")
    print(f"{'='*60}")
    
    detector = get_ensemble_detector()
    sample_reports = create_sample_report_data()
    
    # 更新历史数据
    print("📈 更新历史数据...")
    detector.update_historical_data(sample_reports)
    
    # 批量检测
    print("🔍 执行批量检测...")
    batch_results = detector.batch_detect(sample_reports)
    
    # 统计结果
    anomaly_count = sum(1 for r in batch_results if r.get('is_anomaly', False))
    avg_score = np.mean([r.get('overall_anomaly_score', 0) for r in batch_results])
    
    print(f"\n📈 批量检测统计:")
    print(f"   总研报数: {len(batch_results)}")
    print(f"   异常研报数: {anomaly_count}")
    print(f"   异常比例: {anomaly_count/len(batch_results)*100:.1f}%")
    print(f"   平均异常分数: {avg_score:.3f}")
    
    # 按异常等级分类
    level_counts = {}
    for result in batch_results:
        level = result.get('anomaly_level', 'NORMAL')
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"\n📊 异常等级分布:")
    for level, count in sorted(level_counts.items()):
        print(f"   {level:8}: {count} 个")


def demonstrate_system_status():
    """
    演示系统状态查看
    """
    print(f"\n{'='*60}")
    print("📊 系统状态演示")
    print(f"{'='*60}")
    
    detector = get_ensemble_detector()
    summary = detector.get_detection_summary()
    
    print("⚙️  检测器配置:")
    weights = summary['ensemble_config']['anomaly_weights']
    for detector_type, weight in weights.items():
        print(f"   {detector_type:25}: {weight:.2f}")
    
    print("\n📈 检测器状态:")
    status = summary['detectors_status']
    
    if 'statistical' in status:
        stat_status = status['statistical']
        print(f"   统计检测器:")
        print(f"     历史文档数: {stat_status.get('total_documents', 0)}")
        print(f"     词汇表大小: {stat_status.get('vocabulary_size', 0)}")
    
    if 'behavioral' in status:
        behav_status = status['behavioral']
        print(f"   行为检测器:")
        print(f"     历史发布数: {behav_status.get('total_publications', 0)}")
        print(f"     覆盖股票数: {behav_status.get('unique_stocks', 0)}")
        print(f"     作者数量: {behav_status.get('unique_authors', 0)}")
    
    if 'market' in status:
        market_status = status['market']
        print(f"   市场检测器:")
        print(f"     股票数据: {market_status.get('total_stocks', 0)}")
        print(f"     预测记录: {market_status.get('total_predictions', 0)}")


def main():
    """
    主函数
    """
    try:
        # 演示单个检测
        demonstrate_single_detection()
        
        # 演示批量检测
        demonstrate_batch_detection()
        
        # 演示系统状态
        demonstrate_system_status()
        
        print(f"\n{'='*80}")
        print("✅ 演示完成！")
        print("📚 更多功能请参考项目文档")
        print("="*80)
        
    except Exception as e:
        logger.error(f"演示过程出错: {str(e)}")
        print(f"❌ 演示失败: {str(e)}")
        print("💡 请检查依赖包是否正确安装")


if __name__ == "__main__":
    main() 
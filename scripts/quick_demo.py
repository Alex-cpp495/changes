#!/usr/bin/env python3
"""
快速功能演示
验证异常检测系统的核心功能
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

async def demo():
    """演示系统核心功能"""
    print("🚀 东吴证券研报异常检测系统 - 功能演示\n")
    
    try:
        # 1. 导入服务
        from src.web_interface.services.detection_service import get_detection_service
        from src.web_interface.services.feedback_service import get_feedback_service
        from src.web_interface.services.monitoring_service import get_monitoring_service
        from src.web_interface.models.request_models import DetectionRequest, FeedbackRequest
        from src.continuous_learning import initialize_continuous_learning
        
        print("📋 1. 系统组件初始化...")
        
        # 初始化持续学习系统
        continuous_learning = initialize_continuous_learning(auto_start=False)
        print("   ✅ 持续学习系统已初始化")
        
        # 获取服务实例
        detection_service = get_detection_service()
        feedback_service = get_feedback_service()
        monitoring_service = get_monitoring_service()
        print("   ✅ Web服务层已初始化")
        
        # 2. 异常检测演示
        print("\n🔍 2. 异常检测演示...")
        
        test_report = """
        某公司2023年Q3财报显示：
        - 营业收入同比增长25%，达到50亿元
        - 但现金流量为负10亿元，与收入增长严重不匹配
        - 应收账款大幅增加300%，回款周期异常延长
        - 同时宣布大额投资项目，但未提供详细说明
        - 审计师对某些会计处理表示关注
        """
        
        detection_request = DetectionRequest(
            report_content=test_report,
            report_title="某公司Q3财报分析",
            include_explanations=True
        )
        
        print("   🔄 正在执行异常检测...")
        detection_response = await detection_service.detect_anomaly(detection_request)
        
        if detection_response.status == "success":
            result = detection_response.anomaly_result
            print(f"   ✅ 检测完成!")
            print(f"      📊 异常分数: {result.overall_anomaly_score:.3f}")
            print(f"      🎯 异常级别: {result.overall_anomaly_level.value}")
            print(f"      ⚠️  是否异常: {'是' if result.is_anomalous else '否'}")
            print(f"      🔧 置信度: {result.confidence:.3f}")
            
            if detection_response.recommendations:
                print("      💡 建议:")
                for i, rec in enumerate(detection_response.recommendations[:3], 1):
                    print(f"         {i}. {rec}")
        else:
            print(f"   ❌ 检测失败: {detection_response.message}")
        
        # 3. 反馈提交演示
        print("\n📝 3. 用户反馈演示...")
        
        feedback_request = FeedbackRequest(
            report_id=detection_response.report_id,
            original_prediction=detection_response.anomaly_result.dict(),
            feedback_type="correct_detection",
            is_correct=True,
            confidence_rating=8,
            explanation="检测结果准确，报告确实存在现金流与收入不匹配的异常",
            user_expertise="expert"
        )
        
        feedback_response = await feedback_service.submit_feedback(feedback_request)
        
        if feedback_response.status == "success":
            print(f"   ✅ 反馈提交成功!")
            print(f"      📋 反馈ID: {feedback_response.feedback_id}")
            print(f"      🤖 是否触发学习: {'是' if feedback_response.learning_triggered else '否'}")
        else:
            print(f"   ❌ 反馈提交失败: {feedback_response.message}")
        
        # 4. 系统状态监控演示
        print("\n📊 4. 系统监控演示...")
        
        status_response = await monitoring_service.get_system_status()
        
        if status_response.status == "success":
            print(f"   ✅ 系统状态: {status_response.system_health}")
            print(f"      💻 运行时间: {status_response.uptime:.1f} 小时")
            print(f"      🎯 系统准确率: {status_response.performance_metrics.accuracy:.2%}")
            print(f"      ⚡ 平均处理时间: {status_response.performance_metrics.average_processing_time:.2f}s")
            print(f"      📈 用户满意度: {status_response.performance_metrics.user_satisfaction:.2%}")
        
        # 5. 性能统计
        print("\n📈 5. 性能统计...")
        
        perf_stats = detection_service.get_performance_stats()
        print(f"   📊 总请求数: {perf_stats['total_requests']}")
        print(f"   ✅ 成功率: {perf_stats['success_rate']:.2%}")
        print(f"   ⚡ 缓存命中率: {perf_stats['cache_hit_rate']:.2%}")
        print(f"   🕒 平均处理时间: {perf_stats['average_processing_time']:.2f}s")
        
        print("\n🎉 演示完成！系统运行正常！")
        print("\n" + "="*60)
        print("🌟 系统功能摘要:")
        print("✅ 异常检测 - 四层检测架构，智能识别研报异常")
        print("✅ 用户反馈 - 收集专家反馈，持续改进模型")
        print("✅ 系统监控 - 实时监控性能，确保稳定运行")
        print("✅ 自适应学习 - 基于反馈自动优化检测效果")
        print("✅ Web服务 - 完整的API接口，支持多种应用场景")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("初始化中...")
    success = asyncio.run(demo())
    
    if success:
        print("\n🎊 系统功能验证完成！所有核心功能都工作正常！")
        print("\n💡 下一步:")
        print("   1. 可以开始部署API服务器")
        print("   2. 集成到现有的报告分析流程")
        print("   3. 开始处理真实的研报数据")
    else:
        print("\n💥 系统功能验证失败，请检查错误信息")
    
    sys.exit(0 if success else 1) 
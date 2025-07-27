#!/usr/bin/env python3
"""
导入测试脚本
验证所有模块是否能正常导入
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """测试所有主要模块的导入"""
    print("🔍 开始测试导入...")
    
    try:
        # 测试工具模块
        print("\n📋 测试工具模块:")
        from src.utils.logger import get_logger
        print("✅ utils.logger")
        
        from src.utils.config_loader import load_config
        print("✅ utils.config_loader")
        
        from src.utils.file_utils import get_file_manager
        print("✅ utils.file_utils")
        
        # 测试持续学习模块
        print("\n🤖 测试持续学习模块:")
        from src.continuous_learning import get_continuous_learning_system
        print("✅ continuous_learning")
        
        from src.continuous_learning.feedback_collector import get_feedback_collector
        print("✅ feedback_collector")
        
        from src.continuous_learning.model_monitor import get_model_monitor
        print("✅ model_monitor")
        
        from src.continuous_learning.adaptive_learner import get_adaptive_learner
        print("✅ adaptive_learner")
        
        from src.continuous_learning.performance_tracker import get_performance_tracker
        print("✅ performance_tracker")
        
        # 测试Web界面模块
        print("\n🌐 测试Web界面模块:")
        from src.web_interface.models import DetectionRequest, DetectionResponse
        print("✅ web_interface.models")
        
        from src.web_interface.services import DetectionService
        print("✅ web_interface.services.DetectionService")
        
        from src.web_interface.services import FeedbackService
        print("✅ web_interface.services.FeedbackService")
        
        from src.web_interface.services import MonitoringService
        print("✅ web_interface.services.MonitoringService")
        
        from src.web_interface.services import ReportService
        print("✅ web_interface.services.ReportService")
        
        from src.web_interface.services import ConfigService
        print("✅ web_interface.services.ConfigService")
        
        from src.web_interface.services import AuthService
        print("✅ web_interface.services.AuthService")
        
        # 测试异常检测模块（如果存在）
        print("\n🔍 测试异常检测模块:")
        try:
            from src.anomaly_detection import get_ensemble_detector
            print("✅ anomaly_detection")
        except ImportError as e:
            print(f"⚠️ anomaly_detection: {e}")
        
        print("\n🎉 所有主要模块导入成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ 所有导入测试通过！系统已经可以运行了。")
    else:
        print("\n❌ 导入测试失败，请检查错误信息。")
    
    sys.exit(0 if success else 1) 
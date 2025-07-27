#!/usr/bin/env python3
"""
🏢 东吴证券研报异常检测系统 - 主启动程序
一键启动所有功能模块
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def show_welcome():
    """显示欢迎信息"""
    print("="*70)
    print("🏢 东吴证券研报异常检测系统")
    print("   专注于文本分析，股价信息可选")
    print("="*70)
    print("🚀 系统功能:")
    print("  ✅ AI研报解析 (东吴证券定制版)")
    print("  ✅ 批量数据处理")
    print("  ✅ 异常检测引擎")
    print("  ✅ Web管理界面")
    print("  ✅ 持续学习系统")
    print("="*70)

def show_main_menu():
    """显示主菜单"""
    print("\n🎯 请选择启动模式:")
    print("  1. 🤖 AI研报解析器 (东吴证券专用)")
    print("  2. 🔍 异常检测系统 (完整功能)")
    print("  3. 🌐 Web管理界面")
    print("  4. 🧪 快速功能测试")
    print("  5. 📊 生成系统状态报告")
    print("  6. ❓ 查看帮助文档")
    print("  0. 🚪 退出")

def run_ai_parser():
    """运行AI解析器"""
    print("\n🤖 启动东吴证券AI研报解析器...")
    print("="*50)
    
    try:
        from tools.dongwu_integrated_system import DongwuIntegratedSystem
        system = DongwuIntegratedSystem()
        system.run()
    except Exception as e:
        print(f"❌ AI解析器启动失败: {e}")
        print("💡 尝试使用: python tools/ai_report_parser.py")

def run_anomaly_detection():
    """运行异常检测系统"""
    print("\n🔍 启动异常检测系统...")
    print("="*50)
    
    try:
        from start_system import main as start_main
        start_main()
    except Exception as e:
        print(f"❌ 异常检测系统启动失败: {e}")
        print("💡 请检查环境配置和依赖")

def run_web_interface():
    """运行Web界面"""
    print("\n🌐 启动Web管理界面...")
    print("="*50)
    print("📍 访问地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("⚠️  按 Ctrl+C 停止服务")
    
    try:
        import uvicorn
        from src.web_interface.main import create_app
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n🛑 Web服务已停止")
    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")
        print("💡 请检查环境配置")

def run_quick_test():
    """运行快速测试"""
    print("\n🧪 运行系统功能测试...")
    print("="*50)
    
    try:
        # 测试AI解析器
        print("1️⃣ 测试AI解析器...")
        from tools.quick_test import test_parser
        test_parser()
        
        print("\n2️⃣ 测试完成!")
        print("✅ 如果测试通过，您可以:")
        print("  - 使用模板文件输入您的研报内容")
        print("  - 批量处理多个文件")
        print("  - 启动Web界面进行可视化操作")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def generate_status_report():
    """生成系统状态报告"""
    print("\n📊 生成系统状态报告...")
    print("="*50)
    
    try:
        import json
        from datetime import datetime
        
        # 检查各组件状态
        status = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "components": {}
        }
        
        # 检查AI解析器
        try:
            from tools.ai_report_parser import AIReportParser
            parser = AIReportParser()
            status["components"]["ai_parser"] = "✅ 可用"
        except Exception:
            status["components"]["ai_parser"] = "❌ 不可用"
        
        # 检查异常检测
        try:
            from src.inference.inference_app import InferenceApp
            status["components"]["anomaly_detection"] = "✅ 可用"
        except Exception:
            status["components"]["anomaly_detection"] = "❌ 不可用"
        
        # 检查Web界面
        try:
            from src.web_interface.main import create_app
            status["components"]["web_interface"] = "✅ 可用"
        except Exception:
            status["components"]["web_interface"] = "❌ 不可用"
        
        # 保存报告
        report_file = Path("user_data/results") / f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        
        # 显示结果
        print("📋 系统组件状态:")
        for component, state in status["components"].items():
            print(f"  {component}: {state}")
        
        print(f"\n💾 详细报告已保存: {report_file}")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")

def show_help():
    """显示帮助信息"""
    print("\n❓ 系统帮助文档")
    print("="*50)
    print("📖 文档位置:")
    print("  - tools/使用指南.md (AI解析器使用指南)")
    print("  - user_data/使用说明.md (用户数据处理)")
    print("  - README.md (项目总览)")
    print("  - 系统完成情况报告.md (系统状态)")
    
    print("\n🔧 快速命令:")
    print("  python tools/quick_test.py - 快速测试")
    print("  python tools/ai_report_parser.py 文件.json - 解析单个文件")
    print("  python tools/batch_parser.py user_data/json_files/ - 批量处理")
    print("  python tools/dongwu_integrated_system.py - 完整AI系统")
    
    print("\n📁 重要文件:")
    print("  - user_data/json_files/dongwu_simple_template.json (输入模板)")
    print("  - user_data/json_files/示例_*.json (示例文件)")
    
    print("\n🆘 常见问题:")
    print("  Q: AI解析器如何使用？")
    print("  A: 复制研报内容到模板JSON文件，然后运行解析器")
    print("  Q: 如何批量处理？")
    print("  A: 准备多个JSON文件，使用批量处理器")
    print("  Q: 股价信息是必需的吗？")
    print("  A: 不是，系统专门优化了晨会纪要等非股价分析报告")

def main():
    parser = argparse.ArgumentParser(description='东吴证券研报异常检测系统')
    parser.add_argument('--mode', type=str, choices=['ai', 'anomaly', 'web', 'test'], 
                        help='直接启动指定模式')
    
    args = parser.parse_args()
    
    # 直接启动模式
    if args.mode == 'ai':
        run_ai_parser()
        return
    elif args.mode == 'anomaly':
        run_anomaly_detection()
        return
    elif args.mode == 'web':
        run_web_interface()
        return
    elif args.mode == 'test':
        run_quick_test()
        return
    
    # 交互模式
    show_welcome()
    
    while True:
        try:
            show_main_menu()
            choice = input("\n请选择 (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用东吴证券研报异常检测系统!")
                print("🎯 您的文本分析专家助手")
                break
            elif choice == '1':
                run_ai_parser()
            elif choice == '2':
                run_anomaly_detection()
            elif choice == '3':
                run_web_interface()
            elif choice == '4':
                run_quick_test()
            elif choice == '5':
                generate_status_report()
            elif choice == '6':
                show_help()
            else:
                print("❌ 无效选择，请输入 0-6")
            
            if choice != '0':
                input("\n按 Enter 键继续...")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，系统退出")
            break
        except Exception as e:
            print(f"\n❌ 系统错误: {e}")
            input("按 Enter 键继续...")

if __name__ == "__main__":
    main() 
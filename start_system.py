#!/usr/bin/env python3
"""
东吴证券研报异常检测系统启动脚本
"""

import sys
import os
import time
import uvicorn
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🏛️  东吴证券研报异常检测系统 v1.0.0                    ║
║                                                              ║
║        🤖 AI-Powered Research Report Anomaly Detection       ║
║                                                              ║
║        ✨ Features:                                          ║
║           • 四层异常检测架构                                   ║
║           • 持续学习与自适应优化                              ║
║           • 实时性能监控                                      ║
║           • 批量数据处理                                      ║
║           • 用户友好的Web界面                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    
    print(f"✅ Python版本: {sys.version.split()[0]}")
    
    # 检查关键依赖
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 
        'jieba', 'psutil', 'pydantic', 'jinja2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n💡 请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # 检查必要目录
    required_dirs = ['data', 'logs', 'configs']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            print(f"📁 创建目录: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ 目录存在: {dir_name}")
    
    return True

def create_default_config():
    """创建默认配置文件"""
    config_dir = project_root / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # 创建基本配置文件（如果不存在）
    configs = {
        "anomaly_thresholds.yaml": """
# 异常检测阈值配置
statistical:
  z_score_threshold: 2.5
  outlier_percentile: 0.95

behavioral:
  pattern_threshold: 0.7
  frequency_threshold: 0.8

market:
  volatility_threshold: 0.6
  correlation_threshold: 0.5

semantic:
  similarity_threshold: 0.8
  confidence_threshold: 0.75

ensemble:
  final_threshold: 0.6
  weight_threshold: 0.8
""",
        "model_config.yaml": """
# 模型配置
model:
  name: "qwen2.5-7b"
  max_length: 2048
  temperature: 0.1
  
processing:
  batch_size: 32
  max_workers: 4
  timeout: 30
""",
        "web_config.yaml": """
# Web服务配置
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  
security:
  cors_origins: ["*"]
  session_timeout: 3600
"""
    }
    
    for filename, content in configs.items():
        config_file = config_dir / filename
        if not config_file.exists():
            print(f"📝 创建配置文件: {filename}")
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动东吴证券研报异常检测系统')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')
    parser.add_argument('--reload', action='store_true', help='开启热重载（开发模式）')
    parser.add_argument('--log-level', default='info', help='日志级别')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境，不启动服务')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请修复上述问题后重试")
        sys.exit(1)
    
    # 创建默认配置
    create_default_config()
    
    if args.check_only:
        print("\n✅ 环境检查完成，系统可以正常启动")
        return
    
    print(f"\n🚀 启动系统...")
    print(f"📡 服务地址: http://{args.host}:{args.port}")
    print(f"📚 API文档: http://{args.host}:{args.port}/docs")
    print(f"🎛️ 仪表板: http://{args.host}:{args.port}/")
    print("\n💡 提示:")
    print("   • 按 Ctrl+C 停止服务")
    print("   • 首次启动可能需要较长时间初始化模型")
    print("   • 建议在生产环境中使用反向代理（如Nginx）")
    
    # 启动FastAPI应用
    try:
        uvicorn.run(
            "src.web_interface.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 系统已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
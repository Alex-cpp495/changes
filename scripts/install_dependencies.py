#!/usr/bin/env python3
"""
依赖安装脚本 - 东吴证券研报异常检测系统

自动安装项目所需的Python依赖包

Usage:
    python scripts/install_dependencies.py [--gpu|--cpu] [--china-mirror]
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, description: str = "", check: bool = True) -> bool:
    """执行命令并处理结果"""
    print(f"🔄 {description}")
    print(f"   执行: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()[:200]}...")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 执行失败: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} - 意外错误: {e}")
        return False


def install_pytorch(use_gpu: bool = True, china_mirror: bool = False):
    """安装PyTorch"""
    print("\n" + "="*50)
    print("🔥 安装PyTorch")
    print("="*50)
    
    if use_gpu:
        if china_mirror:
            # 使用清华镜像
            pytorch_cmd = (
                "pip install torch torchvision torchaudio "
                "--index-url https://pypi.tuna.tsinghua.edu.cn/simple/ "
                "--extra-index-url https://download.pytorch.org/whl/cu118"
            )
        else:
            pytorch_cmd = (
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu118"
            )
        description = "安装PyTorch (GPU版本)"
    else:
        if china_mirror:
            pytorch_cmd = (
                "pip install torch torchvision torchaudio "
                "--index-url https://pypi.tuna.tsinghua.edu.cn/simple/ "
                "--extra-index-url https://download.pytorch.org/whl/cpu"
            )
        else:
            pytorch_cmd = (
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        description = "安装PyTorch (CPU版本)"
    
    return run_command(pytorch_cmd, description)


def install_transformers_ecosystem(china_mirror: bool = False):
    """安装Transformers生态系统"""
    print("\n" + "="*50)
    print("🤗 安装Transformers生态系统")
    print("="*50)
    
    packages = [
        "transformers>=4.35.0",
        "accelerate>=0.20.0", 
        "peft>=0.6.2",
        "trl>=0.7.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.14.0"
    ]
    
    base_cmd = "pip install"
    if china_mirror:
        base_cmd += " -i https://pypi.tuna.tsinghua.edu.cn/simple/"
    
    success = True
    for package in packages:
        cmd = f"{base_cmd} {package}"
        if not run_command(cmd, f"安装 {package}"):
            success = False
    
    return success


def install_bitsandbytes(china_mirror: bool = False):
    """安装bitsandbytes (量化库)"""
    print("\n" + "="*50)
    print("⚡ 安装量化库 bitsandbytes")
    print("="*50)
    
    base_cmd = "pip install bitsandbytes>=0.41.0"
    if china_mirror:
        base_cmd += " -i https://pypi.tuna.tsinghua.edu.cn/simple/"
    
    return run_command(base_cmd, "安装bitsandbytes")


def install_other_dependencies(china_mirror: bool = False):
    """安装其他依赖"""
    print("\n" + "="*50)
    print("📦 安装其他依赖包")
    print("="*50)
    
    # 获取requirements.txt路径
    current_dir = Path(__file__).parent.parent.parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ 找不到requirements.txt文件: {requirements_file}")
        return False
    
    base_cmd = f"pip install -r {requirements_file}"
    if china_mirror:
        base_cmd += " -i https://pypi.tuna.tsinghua.edu.cn/simple/"
    
    return run_command(base_cmd, "安装requirements.txt中的依赖")


def verify_installation():
    """验证安装"""
    print("\n" + "="*50)
    print("🔍 验证安装")
    print("="*50)
    
    # 运行环境检查脚本
    check_script = Path(__file__).parent / "check_environment.py"
    if check_script.exists():
        return run_command(f"python {check_script}", "运行环境检查", check=False)
    else:
        print("⚠️  环境检查脚本不存在，手动验证...")
        
        # 基本验证
        test_imports = [
            "import torch; print('PyTorch版本:', torch.__version__)",
            "import transformers; print('Transformers版本:', transformers.__version__)",
            "import numpy; print('NumPy版本:', numpy.__version__)",
            "import pandas; print('Pandas版本:', pandas.__version__)"
        ]
        
        success = True
        for test_import in test_imports:
            if not run_command(f"python -c \"{test_import}\"", f"测试导入", check=False):
                success = False
        
        return success


def main():
    """主安装函数"""
    parser = argparse.ArgumentParser(description="东吴证券研报异常检测系统 - 依赖安装脚本")
    parser.add_argument('--gpu', action='store_true', help='安装GPU版本的PyTorch (默认)')
    parser.add_argument('--cpu', action='store_true', help='安装CPU版本的PyTorch')
    parser.add_argument('--china-mirror', action='store_true', help='使用中国镜像源加速下载')
    parser.add_argument('--skip-pytorch', action='store_true', help='跳过PyTorch安装')
    parser.add_argument('--skip-verification', action='store_true', help='跳过安装验证')
    
    args = parser.parse_args()
    
    # 确定GPU选项
    use_gpu = not args.cpu  # 默认使用GPU，除非明确指定CPU
    
    print("=" * 60)
    print("🚀 东吴证券研报异常检测系统 - 依赖安装")
    print("=" * 60)
    print(f"📊 安装配置:")
    print(f"   PyTorch模式: {'GPU' if use_gpu else 'CPU'}")
    print(f"   镜像源: {'中国镜像' if args.china_mirror else '官方源'}")
    print(f"   跳过PyTorch: {'是' if args.skip_pytorch else '否'}")
    print()
    
    # 升级pip
    print("🔄 升级pip...")
    if args.china_mirror:
        run_command("python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/", "升级pip")
    else:
        run_command("python -m pip install --upgrade pip", "升级pip")
    
    success = True
    
    # 安装PyTorch
    if not args.skip_pytorch:
        if not install_pytorch(use_gpu, args.china_mirror):
            success = False
            print("⚠️  PyTorch安装失败，继续安装其他组件...")
    
    # 安装Transformers生态
    if not install_transformers_ecosystem(args.china_mirror):
        success = False
        print("⚠️  Transformers生态系统安装失败，继续...")
    
    # 安装bitsandbytes
    if not install_bitsandbytes(args.china_mirror):
        success = False
        print("⚠️  bitsandbytes安装失败，可能影响量化功能...")
    
    # 安装其他依赖
    if not install_other_dependencies(args.china_mirror):
        success = False
        print("⚠️  其他依赖安装失败...")
    
    # 验证安装
    if not args.skip_verification:
        if not verify_installation():
            success = False
    
    # 总结
    print("\n" + "="*60)
    print("📋 安装总结")
    print("="*60)
    
    if success:
        print("🎉 所有依赖包安装成功！")
        print("\n💡 下一步:")
        print("   1. 运行环境检查: python eastmoney_anomaly_detection/scripts/check_environment.py")
        print("   2. 运行示例: python eastmoney_anomaly_detection/examples/detection_example.py")
    else:
        print("⚠️  部分依赖包安装失败，请检查错误信息")
        print("\n🔧 故障排除:")
        print("   1. 检查网络连接")
        print("   2. 尝试使用中国镜像: --china-mirror")
        print("   3. 手动安装失败的包")
        print("   4. 检查Python版本 (需要3.8+)")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
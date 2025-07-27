#!/usr/bin/env python3
"""
环境检查脚本 - 东吴证券研报异常检测系统

检查Python环境和依赖包是否正确安装

Usage:
    python scripts/check_environment.py
"""

import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version_str} (支持)"
    else:
        return False, f"❌ Python {version_str} (需要Python 3.8+)"


def check_gpu_availability() -> Tuple[bool, str]:
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB, {gpu_count}个设备)"
        else:
            return False, "⚠️  CUDA不可用，将使用CPU模式"
    except ImportError:
        return False, "❌ PyTorch未安装，无法检查GPU"


def check_package_installation(packages: List[str]) -> Dict[str, Tuple[bool, str]]:
    """检查包安装状态"""
    results = {}
    
    for package in packages:
        try:
            # 特殊处理某些包名
            import_name = package
            if package == "peft":
                import_name = "peft"
            elif package == "bitsandbytes":
                import_name = "bitsandbytes"
            elif package == "colorlog":
                import_name = "colorlog"
            elif package == "scikit-learn":
                import_name = "sklearn"
            elif package == "pyyaml":
                import_name = "yaml"
            
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                # 尝试获取版本信息
                try:
                    module = importlib.import_module(import_name)
                    version = getattr(module, '__version__', 'unknown')
                    results[package] = (True, f"✅ {version}")
                except Exception:
                    results[package] = (True, "✅ 已安装")
            else:
                results[package] = (False, "❌ 未安装")
                
        except Exception as e:
            results[package] = (False, f"❌ 错误: {str(e)}")
    
    return results


def check_cuda_installation() -> Tuple[bool, str]:
    """检查CUDA安装"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            # 提取CUDA版本信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    return True, f"✅ CUDA {cuda_version}"
            return True, "✅ NVIDIA驱动已安装"
        else:
            return False, "❌ NVIDIA驱动未安装或不可用"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "❌ nvidia-smi命令不可用"


def main():
    """主检查函数"""
    print("=" * 60)
    print("🔍 东吴证券研报异常检测系统 - 环境检查")
    print("=" * 60)
    
    # 系统信息
    print(f"\n📊 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   架构: {platform.machine()}")
    
    # Python版本检查
    print(f"\n🐍 Python环境:")
    py_ok, py_msg = check_python_version()
    print(f"   {py_msg}")
    
    # CUDA检查
    print(f"\n🔧 CUDA环境:")
    cuda_ok, cuda_msg = check_cuda_installation()
    print(f"   {cuda_msg}")
    
    # GPU检查
    print(f"\n🖥️  GPU检查:")
    gpu_ok, gpu_msg = check_gpu_availability()
    print(f"   {gpu_msg}")
    
    # 核心依赖包检查
    print(f"\n📦 核心依赖包:")
    core_packages = [
        'torch',
        'transformers', 
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'peft',
        'bitsandbytes',
        'accelerate',
        'jieba',
        'fastapi',
        'streamlit',
        'plotly',
        'pyyaml',
        'colorlog'
    ]
    
    package_results = check_package_installation(core_packages)
    
    installed_count = 0
    for package, (is_installed, message) in package_results.items():
        print(f"   {package:15} {message}")
        if is_installed:
            installed_count += 1
    
    # 总结
    print(f"\n📋 检查总结:")
    print(f"   Python版本: {'✅' if py_ok else '❌'}")
    print(f"   CUDA支持: {'✅' if cuda_ok else '❌'}")
    print(f"   GPU可用: {'✅' if gpu_ok else '⚠️'}")
    print(f"   依赖包: {installed_count}/{len(core_packages)} 已安装")
    
    # 建议
    print(f"\n💡 建议:")
    if not py_ok:
        print("   ⚠️  请升级到Python 3.8或更高版本")
    
    if installed_count < len(core_packages):
        print("   📦 运行以下命令安装缺失的依赖包:")
        print("      pip install -r requirements.txt")
    
    if not gpu_ok and cuda_ok:
        print("   🔧 PyTorch可能没有正确安装CUDA支持，请重新安装:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if installed_count == len(core_packages) and gpu_ok:
        print("   🎉 环境配置完善，可以开始使用系统！")
    
    print("=" * 60)
    
    return installed_count == len(core_packages) and py_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
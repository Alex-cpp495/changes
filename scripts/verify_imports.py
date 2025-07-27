#!/usr/bin/env python3
"""
导入验证脚本
用于诊断Python包导入问题，帮助解决Pylance错误
"""

import sys
import importlib
import importlib.util
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_import(module_name, import_alias=None):
    """测试模块导入"""
    try:
        if import_alias:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} -> {import_alias}: {getattr(module, '__version__', 'OK')}")
        else:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}: {getattr(module, '__version__', 'OK')}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: {e}")
        return False

def test_relative_import(module_path):
    """测试相对导入"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None:
            print(f"❌ 无法找到模块: {module_path}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        print(f"✅ 项目模块: {module_path}")
        return True
    except Exception as e:
        print(f"❌ 项目模块 {module_path}: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Python环境和导入验证")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"项目根目录: {project_root}")
    print()
    
    # 测试核心依赖
    print("🔍 测试核心依赖包:")
    core_packages = [
        "numpy",
        "pandas", 
        "torch",
        "transformers",
        "sklearn",
        "jieba",
        "peft",
        "scipy",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "yaml",
        "colorlog",
        "tqdm",
        "requests"
    ]
    
    success_count = 0
    for package in core_packages:
        if test_import(package):
            success_count += 1
    
    print(f"\n核心包导入成功率: {success_count}/{len(core_packages)} ({success_count/len(core_packages)*100:.1f}%)")
    
    # 测试特殊导入
    print("\n🔍 测试特殊导入:")
    special_imports = [
        ("sklearn.metrics.pairwise", "cosine_similarity"),
        ("collections", "defaultdict"),
        ("datetime", "datetime"),
        ("pathlib", "Path"),
        ("json", None),
        ("pickle", None),
        ("re", None),
        ("logging", None),
        ("typing", None)
    ]
    
    for module_name, attr in special_imports:
        try:
            module = importlib.import_module(module_name)
            if attr:
                getattr(module, attr)
                print(f"✅ from {module_name} import {attr}")
            else:
                print(f"✅ import {module_name}")
        except Exception as e:
            print(f"❌ {module_name}.{attr if attr else ''}: {e}")
    
    # 测试项目模块结构
    print("\n🔍 测试项目模块结构:")
    project_modules = [
        project_root / "src" / "utils" / "__init__.py",
        project_root / "src" / "utils" / "logger.py", 
        project_root / "src" / "utils" / "config_loader.py",
        project_root / "src" / "utils" / "text_utils.py",
        project_root / "src" / "utils" / "file_utils.py",
        project_root / "src" / "anomaly_detection" / "__init__.py",
        project_root / "src" / "models" / "__init__.py"
    ]
    
    for module_path in project_modules:
        if module_path.exists():
            test_relative_import(module_path)
        else:
            print(f"❌ 文件不存在: {module_path}")
    
    # 测试项目内部导入
    print("\n🔍 测试项目内部导入:")
    try:
        # 确保src目录在Python路径中
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # 测试utils模块导入
        import importlib
        
        # 测试utils子模块
        utils_modules = [
            'utils.logger',
            'utils.config_loader', 
            'utils.text_utils',
            'utils.file_utils'
        ]
        
        for module_name in utils_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    print(f"✅ {module_name} 模块可用")
                else:
                    print(f"❌ {module_name} 模块未找到")
            except Exception as e:
                print(f"❌ {module_name} 导入测试失败: {e}")
        
        # 尝试实际导入（如果模块存在）
        try:
            from src.utils.logger import get_logger
            logger = get_logger('test')
            print("✅ src.utils.logger 实际导入成功")
        except Exception as e:
            print(f"⚠️ src.utils.logger 实际导入失败: {e}")
        
    except Exception as e:
        print(f"❌ 项目内部导入测试失败: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    
    # 给出建议
    print("\n💡 解决建议:")
    print("1. 重启VS Code")
    print("2. 按Ctrl+Shift+P，选择'Python: Select Interpreter'")
    print("3. 选择正确的Python解释器路径")
    print("4. 按Ctrl+Shift+P，选择'Python: Restart Language Server'")
    print("5. 如果问题依然存在，尝试重新安装缺失的包")

if __name__ == "__main__":
    main() 
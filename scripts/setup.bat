@echo off
chcp 65001 >nul
echo ================================================================
echo 🚀 东吴证券研报异常检测系统 - Windows一键安装
echo ================================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ 检测到Python环境
python --version

REM 升级pip
echo.
echo 🔄 升级pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

REM 检查CUDA
echo.
echo 🔍 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  未检测到NVIDIA GPU或驱动，将安装CPU版本PyTorch
    set USE_GPU=false
) else (
    echo ✅ 检测到NVIDIA GPU
    set USE_GPU=true
)

REM 安装依赖
echo.
echo 📦 开始安装依赖包...

if "%USE_GPU%"=="true" (
    echo 🔥 安装GPU版本PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo 💻 安装CPU版本PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo 🤗 安装Transformers生态系统...
python -m pip install transformers>=4.35.0 accelerate>=0.20.0 peft>=0.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo.
echo ⚡ 安装量化库...
python -m pip install bitsandbytes>=0.41.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo.
echo 📦 安装其他依赖...
cd /d "%~dp0.."
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

REM 验证安装
echo.
echo 🔍 验证安装...
python scripts\check_environment.py

echo.
echo ================================================================
if errorlevel 1 (
    echo ❌ 安装过程中出现错误，请检查上面的错误信息
    echo 💡 常见解决方案:
    echo    1. 以管理员身份运行此脚本
    echo    2. 检查网络连接
    echo    3. 查看 SETUP.md 获取详细帮助
) else (
    echo 🎉 安装完成！
    echo.
    echo 💡 下一步:
    echo    1. 运行测试: python examples\detection_example.py
    echo    2. 查看文档: README.md
)
echo ================================================================
echo.
pause 
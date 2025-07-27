#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "================================================================"
echo "🚀 东吴证券研报异常检测系统 - Linux/macOS一键安装"
echo "================================================================"
echo

# 检查Python
print_info "检查Python环境..."
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    print_error "未找到Python，请先安装Python 3.8或更高版本"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_success "检测到Python $PYTHON_VERSION"

# 检查pip
print_info "检查pip..."
if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    print_error "未找到pip，请先安装pip"
    exit 1
fi

# 升级pip
print_info "升级pip..."
$PYTHON_CMD -m pip install --upgrade pip

# 检查CUDA (可选)
print_info "检查CUDA环境..."
if command_exists nvidia-smi; then
    print_success "检测到NVIDIA GPU"
    USE_GPU=true
else
    print_warning "未检测到NVIDIA GPU，将安装CPU版本PyTorch"
    USE_GPU=false
fi

# 创建虚拟环境 (可选)
read -p "是否创建Python虚拟环境? (推荐) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    print_info "创建虚拟环境 'eastmoney_env'..."
    $PYTHON_CMD -m venv eastmoney_env
    source eastmoney_env/bin/activate
    print_success "虚拟环境已激活"
fi

# 使用镜像源
read -p "是否使用中国镜像源加速下载? [Y/n]: " use_mirror
if [[ ! $use_mirror =~ ^[Nn]$ ]]; then
    MIRROR_ARG="-i https://pypi.tuna.tsinghua.edu.cn/simple/"
    print_info "将使用清华镜像源"
else
    MIRROR_ARG=""
fi

echo
print_info "开始安装依赖包..."

# 安装PyTorch
if $USE_GPU; then
    print_info "安装GPU版本PyTorch..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_info "安装CPU版本PyTorch..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装Transformers生态系统
print_info "安装Transformers生态系统..."
$PIP_CMD install transformers>=4.35.0 accelerate>=0.20.0 peft>=0.6.2 $MIRROR_ARG

# 安装bitsandbytes (Linux用户)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "安装量化库 bitsandbytes..."
    $PIP_CMD install bitsandbytes>=0.41.0 $MIRROR_ARG
fi

# 安装其他依赖
print_info "安装其他依赖..."
cd "$(dirname "$0")/.."
$PIP_CMD install -r requirements.txt $MIRROR_ARG

# 验证安装
echo
print_info "验证安装..."
if $PYTHON_CMD scripts/check_environment.py; then
    print_success "所有依赖安装完成！"
    echo
    echo "💡 下一步:"
    echo "   1. 运行测试: $PYTHON_CMD examples/detection_example.py"
    echo "   2. 查看文档: README.md"
    
    if [[ $create_venv =~ ^[Yy]$ ]]; then
        echo "   3. 下次使用前激活环境: source eastmoney_env/bin/activate"
    fi
else
    print_error "安装验证失败，请检查错误信息"
    echo
    echo "💡 常见解决方案:"
    echo "   1. 检查网络连接"
    echo "   2. 尝试使用镜像源"
    echo "   3. 查看 SETUP.md 获取详细帮助"
    exit 1
fi

echo "================================================================" 
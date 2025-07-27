# 环境设置指南 - 东吴证券研报异常检测系统

本指南将帮助您快速设置项目所需的Python环境和依赖包。

## 📋 系统要求

### 基础要求
- **Python版本**: 3.8+ (推荐 3.9 或 3.10)
- **操作系统**: Windows 10/11, macOS, Linux
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间

### GPU要求 (可选但推荐)
- **NVIDIA GPU**: RTX 4060/4070 或更高
- **显存**: 至少 6GB (推荐 8GB+)
- **CUDA版本**: 11.8 或 12.x
- **驱动版本**: 支持CUDA的最新驱动

## 🚀 快速安装

### 方法一: 自动安装 (推荐)

```bash
# 1. 克隆或下载项目
cd eastmoney_anomaly_detection

# 2. 运行自动安装脚本
python scripts/install_dependencies.py --china-mirror

# 3. 验证安装
python scripts/check_environment.py
```

### 方法二: 手动安装

```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 安装PyTorch (GPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 验证安装
python scripts/check_environment.py
```

## 🌍 中国用户优化

### 使用镜像源加速
```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt

# 或者使用阿里镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

### 永久配置镜像源
```bash
# Windows
mkdir %APPDATA%\pip
echo [global] > %APPDATA%\pip\pip.ini
echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple/ >> %APPDATA%\pip\pip.ini

# Linux/macOS
mkdir -p ~/.config/pip
echo "[global]" > ~/.config/pip/pip.conf
echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple/" >> ~/.config/pip/pip.conf
```

## 🔧 常见问题解决

### 问题 1: PyTorch GPU支持问题

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
```bash
# 卸载现有PyTorch
pip uninstall torch torchvision torchaudio

# 重新安装GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证CUDA
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 问题 2: bitsandbytes安装失败

**症状**: `ModuleNotFoundError: No module named 'bitsandbytes'`

**解决方案**:
```bash
# Windows用户
pip install bitsandbytes --force-reinstall

# Linux用户
pip install bitsandbytes>=0.41.0

# 如果仍然失败，使用预编译版本
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### 问题 3: transformers版本冲突

**症状**: `ImportError` 或版本兼容性错误

**解决方案**:
```bash
# 更新transformers
pip install --upgrade transformers>=4.35.0

# 如果冲突，重新安装
pip uninstall transformers accelerate peft
pip install transformers>=4.35.0 accelerate>=0.20.0 peft>=0.6.2
```

### 问题 4: jieba中文分词问题

**症状**: 中文分词结果不准确

**解决方案**:
```bash
# 重新安装jieba
pip install --upgrade jieba>=0.42.1

# 下载词典文件 (在Python中执行)
python -c "import jieba; jieba.initialize()"
```

### 问题 5: 内存不足错误

**症状**: `CUDA out of memory` 或 `RuntimeError: CUDA error`

**解决方案**:
1. **减少batch_size**: 在配置文件中设置更小的批处理大小
2. **启用量化**: 确保4bit量化已启用
3. **使用CPU模式**: 修改配置使用CPU推理

```python
# 在代码中添加内存清理
import torch
torch.cuda.empty_cache()
```

## 🐍 虚拟环境设置 (推荐)

### 使用conda
```bash
# 创建新环境
conda create -n eastmoney python=3.10

# 激活环境
conda activate eastmoney

# 安装依赖
python scripts/install_dependencies.py --china-mirror
```

### 使用venv
```bash
# 创建虚拟环境
python -m venv eastmoney_env

# 激活环境 (Windows)
eastmoney_env\Scripts\activate

# 激活环境 (Linux/macOS)
source eastmoney_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📊 环境验证

运行以下命令验证环境是否正确配置:

```bash
# 全面环境检查
python scripts/check_environment.py

# 快速测试
python -c "
import torch
import transformers
import numpy as np
print('✅ 基础库导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'GPU可用: {torch.cuda.is_available()}')
print(f'Transformers版本: {transformers.__version__}')
"
```

## 🎯 性能优化配置

### 针对8GB显存优化
```yaml
# 在 configs/model_config.yaml 中设置
model:
  quantization:
    enabled: true
    method: "4bit_nf4"
    compute_dtype: "float16"
  
inference:
  batch_size: 1
  max_new_tokens: 512
```

### 系统级优化
```bash
# 设置PyTorch优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Windows用户在PowerShell中执行
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

## 📞 获取帮助

如果您仍然遇到问题，请：

1. **检查日志**: 查看详细的错误信息
2. **运行诊断**: `python scripts/check_environment.py`
3. **查看文档**: 阅读 README.md 了解更多信息
4. **提交Issue**: 在项目仓库中提交问题

## 🔄 更新依赖

```bash
# 更新所有依赖到最新版本
pip install --upgrade -r requirements.txt

# 更新特定包
pip install --upgrade torch transformers

# 检查过时的包
pip list --outdated
```

---

**💡 提示**: 建议在虚拟环境中安装项目依赖，避免与系统Python环境冲突。 
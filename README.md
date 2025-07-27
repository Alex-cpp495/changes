# 东吴证券研报异常检测系统

## 项目概述

东吴证券研报异常检测系统是一个基于AI的智能分析平台，专门用于检测券商研报中的异常模式和关键信息。系统采用四层异常检测体系，结合深度学习和传统机器学习方法，为研报质量控制和合规监管提供技术支持。

## 核心功能

### 🔍 四层异常检测体系

1. **统计异常检测**
   - 文本长度异常检测
   - 词汇新颖性分析
   - 情感强度异常识别
   - 数值预测合理性验证

2. **行为异常检测**
   - 发布时机异常（重大事件前后）
   - 发布频率变化监控
   - 关注转移异常识别
   - 写作习惯突变检测

3. **市场关联异常检测**
   - 异常收益率分析
   - 成交量异常监控
   - 预测准确性评估
   - 价格趋势一致性验证

4. **语义异常检测**
   - 逻辑矛盾识别
   - 历史观点转变分析
   - 信息来源可靠性评估
   - 语义一致性检查

### 🧠 AI技术栈

- **Qwen2.5-7B**: 大语言模型，支持4bit量化和LoRA微调
- **多任务学习**: 情感分析、风格识别、异常预检测
- **集成学习**: 加权融合多层检测结果
- **可解释性**: 注意力可视化、反事实解释、相似案例分析

### ⚡ 性能优化

- **8GB显存适配**: 4bit NF4量化技术
- **推理优化**: 模型缓存、批处理、结果缓存
- **并发处理**: 异步IO、多进程数据加载
- **资源监控**: GPU使用率、内存占用监控

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- 8GB+ GPU显存 (RTX 4060/4070)
- 16GB+ 系统内存

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from src.anomaly_detection.ensemble_detector import get_ensemble_detector

# 初始化检测器
detector = get_ensemble_detector()

# 准备研报数据
report_data = {
    'text': '研报内容...',
    'publication_time': datetime.now(),
    'author': '分析师姓名',
    'stock_codes': ['000001'],
    'rating': '买入'
}

# 执行异常检测
result = detector.detect_anomalies(report_data)

# 查看结果
print(f"异常分数: {result['overall_anomaly_score']:.3f}")
print(f"异常等级: {result['anomaly_level']}")
print(f"处理建议: {result['recommendations']}")
```

### 批量检测

```python
# 批量检测多个研报
reports_data = [report1, report2, report3]
batch_results = detector.batch_detect(reports_data)

# 统计分析
anomaly_count = sum(1 for r in batch_results if r['is_anomaly'])
print(f"异常研报数量: {anomaly_count}/{len(batch_results)}")
```

## 项目结构

```
eastmoney_anomaly_detection/
├── configs/                    # 配置文件
│   ├── model_config.yaml      # 模型配置
│   ├── training_config.yaml   # 训练配置
│   ├── anomaly_thresholds.yaml # 异常检测阈值
│   └── web_config.yaml        # Web服务配置
├── data/                       # 数据目录
│   ├── raw_reports/           # 原始研报TXT
│   ├── processed_reports/     # 预处理后数据
│   ├── market_data/           # 市场数据
│   ├── training_data/         # 训练数据
│   ├── models/                # 模型文件
│   └── results/               # 分析结果
├── src/                       # 源代码
│   ├── utils/                 # 工具模块
│   │   ├── logger.py         # 日志工具
│   │   ├── config_loader.py  # 配置加载
│   │   ├── file_utils.py     # 文件操作
│   │   └── text_utils.py     # 文本处理
│   ├── models/               # 模型定义
│   │   └── qwen_wrapper.py   # Qwen模型封装
│   ├── anomaly_detection/    # 异常检测
│   │   ├── statistical_detector.py    # 统计异常检测
│   │   ├── behavioral_detector.py     # 行为异常检测
│   │   ├── market_detector.py         # 市场关联异常
│   │   ├── semantic_detector.py       # 语义异常检测
│   │   └── ensemble_detector.py       # 集成检测器
│   ├── training/             # 训练模块
│   ├── inference/            # 推理模块
│   ├── explainability/       # 可解释性
│   ├── continuous_learning/  # 持续学习
│   └── web_interface/        # Web界面
├── examples/                 # 示例代码
│   └── detection_example.py # 使用示例
└── requirements.txt          # 依赖包
```

## 配置说明

### 模型配置 (model_config.yaml)

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  quantization:
    enabled: true
    method: "4bit_nf4"
  max_sequence_length: 2048

lora_config:
  rank: 32
  alpha: 64
  dropout: 0.1

inference:
  batch_size: 1
  temperature: 0.7
  top_p: 0.95
```

### 异常检测阈值 (anomaly_thresholds.yaml)

```yaml
anomaly_weights:
  statistical_anomaly: 0.15
  behavioral_anomaly: 0.25
  market_correlation_anomaly: 0.35
  semantic_anomaly: 0.25

anomaly_levels:
  CRITICAL: {min: 0.8, max: 1.0}
  HIGH: {min: 0.6, max: 0.8}
  MEDIUM: {min: 0.4, max: 0.6}
  LOW: {min: 0.2, max: 0.4}
  NORMAL: {min: 0.0, max: 0.2}
```

## 异常等级说明

| 等级 | 分数范围 | 描述 | 建议操作 |
|------|---------|------|----------|
| CRITICAL | 0.8-1.0 | 极高异常度 | 立即暂停发布，全面审核 |
| HIGH | 0.6-0.8 | 高异常度 | 暂缓发布，要求详细解释 |
| MEDIUM | 0.4-0.6 | 中等异常度 | 要求作者提供说明 |
| LOW | 0.2-0.4 | 轻度异常 | 标记关注，跟踪表现 |
| NORMAL | 0.0-0.2 | 正常范围 | 正常发布流程 |

## 技术特点

### 🚀 先进技术

- **大模型应用**: 基于Qwen2.5-7B的深度语义理解
- **多模态融合**: 文本、数值、时序数据综合分析
- **量化优化**: 4bit量化技术，适配8GB显存
- **LoRA微调**: 高效的模型适配方法

### 📊 检测精度

- **统计异常**: 基于Z-score和百分位数的精确检测
- **行为模式**: 时间序列分析和频率异常识别
- **市场关联**: 收益率和成交量的量化分析
- **语义理解**: 基于大模型的逻辑一致性检查

### 🔄 持续学习

- **用户反馈**: 收集人工标注结果
- **模型更新**: 增量学习和模型漂移检测
- **性能监控**: 实时跟踪检测效果
- **历史数据**: 动态更新基准数据

## 使用示例

### 运行演示程序

```bash
cd eastmoney_anomaly_detection
python examples/detection_example.py
```

### 关键API

#### 1. 初始化检测器

```python
from src.anomaly_detection.ensemble_detector import get_ensemble_detector
detector = get_ensemble_detector()
```

#### 2. 单个研报检测

```python
result = detector.detect_anomalies(report_data)
```

#### 3. 批量检测

```python
batch_results = detector.batch_detect(reports_list)
```

#### 4. 更新历史数据

```python
detector.update_historical_data(historical_reports)
```

#### 5. 添加市场数据

```python
detector.market_detector.add_market_data(
    stock_code='000001',
    date=datetime.now(),
    open_price=50.0,
    close_price=52.0,
    high_price=53.0,
    low_price=49.5,
    volume=1000000
)
```

## 开发指南

### 代码规范

- 遵循PEP 8规范
- 使用类型注解
- 完整的docstring文档
- 异常处理机制
- 单元测试覆盖

### 日志使用

```python
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info("开始处理研报文件: {}", file_path)
logger.warning("检测到质量较低的文本，质量分数: {:.2f}", score)
logger.error("模型推理失败: {}", str(error))
```

### 配置管理

```python
from src.utils.config_loader import load_config, MODEL_CONFIG
config = load_config(MODEL_CONFIG)
learning_rate = config['training']['learning_rate']
```

## 性能基准

### 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|-------|---------|----------|
| GPU | GTX 1660 (6GB) | RTX 4060 (8GB) |
| 内存 | 16GB | 32GB |
| 存储 | 50GB可用空间 | 100GB SSD |
| CPU | 4核心 | 8核心以上 |

### 性能指标

- **单个研报检测**: ~30-60秒
- **批量检测**: ~100个研报/小时
- **模型加载**: ~2-5分钟
- **显存占用**: ~6-7GB (量化模式)

## 常见问题

### Q: 如何处理显存不足？

A: 系统已针对8GB显存优化：
- 启用4bit量化
- 调整batch_size=1
- 使用梯度累积
- 定期清理显存缓存

### Q: 如何提高检测准确性？

A: 建议：
- 提供充足的历史数据
- 定期更新市场数据
- 收集用户反馈进行模型调优
- 根据业务场景调整阈值

### Q: 如何集成到现有系统？

A: 可通过以下方式：
- REST API接口
- Python SDK调用
- 批量文件处理
- 数据库直连

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交代码更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目负责人: AI工程团队
- 邮箱: ai@example.com
- 技术支持: support@example.com

---

**注意**: 本系统为专业金融工具，使用前请确保符合相关法规要求。系统检测结果仅供参考，最终决策需结合人工审核。 
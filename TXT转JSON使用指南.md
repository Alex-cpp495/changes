# 🔄 TXT转JSON完整使用指南

## 🎯 为您的100篇研报数据准备的专用工具

### 📁 第一步：文件组织

1. **将您的TXT文件放入指定目录**：
   ```bash
   # 将您的100篇TXT研报文件复制到这里
   cp /您的文件路径/*.txt user_data/txt_files/
   ```

2. **目录结构**：
   ```
   user_data/
   ├── txt_files/          # 👈 放您的TXT文件这里
   ├── json_files/         # 转换后的JSON文件
   ├── results/           # 分析结果
   └── 使用说明.md        
   ```

### 🔧 第二步：转换为JSON格式

#### 方法1：一键转换（推荐）
```bash
# 转换所有TXT文件为JSON格式
python tools/txt_to_json_converter.py user_data/txt_files/ -o user_data/json_files/my_100_reports.json
```

#### 方法2：先预览再转换（安全）
```bash
# 1. 先预览转换结果
python tools/txt_to_json_converter.py user_data/txt_files/ --preview

# 2. 确认无误后正式转换
python tools/txt_to_json_converter.py user_data/txt_files/ -o user_data/json_files/my_100_reports.json
```

#### 方法3：分批测试（稳妥）
```bash
# 先测试转换5个文件
python tools/txt_to_json_converter.py user_data/txt_files/ --sample 5 -o user_data/json_files/test.json

# 确认无误后转换全部
python tools/txt_to_json_converter.py user_data/txt_files/ -o user_data/json_files/my_100_reports.json
```

### 📊 第三步：批量异常检测

```bash
# 分析您的100篇研报
python scripts/batch_process_reports.py user_data/json_files/my_100_reports.json --output-dir user_data/results/
```

### 🚀 第四步：启动Web界面查看结果

```bash
# 启动系统
python start_system.py

# 访问 http://localhost:8000 查看结果
```

## 🔍 工具特性说明

### 自动智能识别
转换工具会自动识别：

✅ **公司名称**：
- 从文件名：`贵州茅台_2023-10-31_研报.txt` → "贵州茅台"
- 从内容：`关于招商银行的分析` → "招商银行"

✅ **报告日期**：
- `2023-10-31`、`20231031`、`2023年10月31日` 等格式

✅ **分析师姓名**：
- `分析师：张三`、`研究员：李四` 等格式

✅ **报告标题**：
- 智能识别文档标题行

### 支持的文件命名格式
```
✅ 推荐格式：
公司名_日期_类型.txt
贵州茅台_2023-10-31_三季报.txt
比亚迪_20231031_深度研究.txt

✅ 也支持：
某公司2023年第三季度分析报告.txt
20231031_招商银行_调研报告.txt
深度研究_宁德时代_2023.txt
```

## 🧪 快速测试

运行快速测试脚本验证功能：
```bash
python tools/quick_test.py
```

## 💡 实用技巧

### 1. 文件编码处理
确保TXT文件为UTF-8编码：
```bash
# Windows用户：用记事本打开文件，另存为时选择UTF-8编码
# Mac/Linux用户：使用iconv转换编码
iconv -f GBK -t UTF-8 原文件.txt > 新文件.txt
```

### 2. 批量重命名文件
```bash
# 如果文件名不规范，可以批量重命名
# 例如：将 "report_001.txt" 重命名为 "公司A_2023-10-31_研报.txt"
```

### 3. 处理大量文件
```bash
# 如果文件很多，可以分批处理
python tools/txt_to_json_converter.py user_data/txt_files/ --sample 50 -o user_data/json_files/batch1.json
python tools/txt_to_json_converter.py user_data/txt_files/ --sample 100 -o user_data/json_files/batch2.json
```

## 📋 转换结果示例

### 输入：TXT文件
```
贵州茅台2023年第三季度财务分析报告

分析师：李明
日期：2023-10-31

营业收入385.2亿元，同比增长18.5%...
```

### 输出：JSON格式
```json
{
  "report_id": "txt_abc12345",
  "title": "贵州茅台2023年第三季度财务分析报告",
  "content": "贵州茅台2023年第三季度...",
  "company": "贵州茅台",
  "report_date": "2023-10-31",
  "analyst": "李明",
  "original_filename": "贵州茅台_2023-10-31_三季报.txt"
}
```

## ❓ 常见问题解决

### Q1: 公司名称识别不准确
**解决方案**：
1. 重命名文件使用标准格式：`公司名_日期_类型.txt`
2. 或在TXT内容开头添加：`公司：XXX公司`
3. 转换后手动编辑JSON文件

### Q2: 日期格式无法识别
**解决方案**：
1. 使用标准日期格式：`2023-10-31`
2. 在文件名中包含日期
3. 在内容中添加：`日期：2023-10-31`

### Q3: 转换后内容乱码
**解决方案**：
1. 确保TXT文件是UTF-8编码
2. 使用文本编辑器重新保存为UTF-8
3. 检查是否有特殊字符

## 🎊 完成！

现在您已经掌握了：
✅ 如何将TXT文件转换为JSON格式  
✅ 如何批量处理100篇研报  
✅ 如何解决常见问题  
✅ 如何优化转换效果  

**准备好处理您的100篇研报数据了吗？开始吧！** 🚀 
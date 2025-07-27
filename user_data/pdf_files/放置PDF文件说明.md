# 📄 PDF文件处理说明

## 📁 PDF文件存放位置

请将您的PDF研报文件放在这个目录中：`user_data/pdf_files/`

## 🔍 PDF识别工具特性

我们的PDF识别工具结合了多种先进技术：

### 🛠️ 多重识别方法
1. **pdfplumber** - 推荐方法，效果最佳
2. **PyMuPDF** - 高性能PDF处理
3. **PyPDF2** - 兼容性强
4. **OCR识别** - 处理扫描版PDF

### 📊 智能元数据提取
- ✅ 券商名称：东吴证券、招商证券等
- ✅ 报告类型：策略、深度、快评等
- ✅ 发布日期：多种日期格式
- ✅ 分析师姓名
- ✅ 投资评级
- ✅ 公司名称
- ✅ 表格数据

### 🔧 使用方法

```bash
# 检查依赖安装
python tools/pdf_to_json_converter.py --check-deps

# 安装PDF处理依赖
pip install PyPDF2 pdfplumber PyMuPDF

# 如需OCR功能（扫描版PDF）
pip install pytesseract pillow pdf2image

# 转换单个PDF文件
python tools/pdf_to_json_converter.py 您的PDF文件.pdf --preview

# 转换目录中所有PDF
python tools/pdf_to_json_converter.py user_data/pdf_files/ -o user_data/json_files/pdf_reports.json
```

## 💡 使用技巧

1. **文本型PDF**：大多数证券研报都是文本型，识别效果最佳
2. **扫描版PDF**：自动使用OCR识别，时间较长但效果良好
3. **复杂布局**：工具会自动处理表格和多列布局
4. **批量处理**：支持一次处理多个PDF文件

现在可以开始测试您的PDF文件了！ 
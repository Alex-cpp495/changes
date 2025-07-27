#!/usr/bin/env python3
"""
批量转换工具测试脚本
验证TXT到JSON转换功能的正确性
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.batch_txt_to_json import TxtToJsonConverter


def test_converter():
    """测试TXT到JSON转换工具"""
    print("=" * 50)
    print("🧪 测试东吴证券研报TXT到JSON转换工具")
    print("=" * 50)
    
    # 创建临时测试目录
    test_dir = Path("user_data/test_conversion")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试输出目录
    output_dir = "user_data/test_json_output"
    
    # 初始化转换器
    converter = TxtToJsonConverter(output_dir=output_dir)
    
    # 测试单个文件转换
    print("\n🔍 测试单个文件转换...")
    
    # 使用已有的TXT文件
    test_file = Path("user_data/txt_files/1.txt")
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        print("请确保 user_data/txt_files 目录中有TXT文件")
        return
    
    # 转换单个文件
    json_data = converter.convert_single_file(test_file)
    
    # 验证JSON结构
    print("\n✅ 验证JSON结构...")
    expected_keys = ["report_id", "metadata", "content", "dongwu_style_features"]
    for key in expected_keys:
        if key in json_data:
            print(f"  ✓ 包含 '{key}' 字段")
        else:
            print(f"  ✗ 缺少 '{key}' 字段")
    
    # 验证东吴风格特征
    print("\n✅ 验证东吴风格特征...")
    style_features = json_data.get("dongwu_style_features", {})
    
    # 验证文本统计特征
    if "text_stats" in style_features:
        print("  ✓ 包含文本统计特征")
        for key in ["total_length", "paragraph_count", "avg_paragraph_length"]:
            if key in style_features["text_stats"]:
                print(f"    ✓ 包含 '{key}': {style_features['text_stats'][key]}")
    else:
        print("  ✗ 缺少文本统计特征")
    
    # 验证结构特征
    if "structure_features" in style_features:
        print("  ✓ 包含结构特征")
        for key, value in style_features["structure_features"].items():
            print(f"    ✓ {key}: {value}")
    else:
        print("  ✗ 缺少结构特征")
    
    # 验证内容特征
    if "content_features" in style_features:
        print("  ✓ 包含内容特征")
        content_features = style_features["content_features"]
        
        # 验证投资评级
        if "investment_rating" in content_features:
            print(f"    ✓ 投资评级: {content_features['investment_rating']}")
        
        # 验证驱动因素
        if "driving_factors" in content_features:
            factors = content_features["driving_factors"]
            print(f"    ✓ 驱动因素: {len(factors)} 项")
            for i, factor in enumerate(factors[:3], 1):
                print(f"      {i}. {factor[:50]}..." if len(factor) > 50 else f"      {i}. {factor}")
        
        # 验证风险因素
        if "risk_factors" in content_features:
            risks = content_features["risk_factors"]
            print(f"    ✓ 风险因素: {len(risks)} 项")
            for i, risk in enumerate(risks[:3], 1):
                print(f"      {i}. {risk[:50]}..." if len(risk) > 50 else f"      {i}. {risk}")
        
        # 验证目标公司
        if "target_companies" in content_features:
            companies = content_features["target_companies"]
            print(f"    ✓ 目标公司: {len(companies)} 家")
            for i, company in enumerate(companies[:3], 1):
                print(f"      {i}. {company.get('code', '未知')} {company.get('name', '未知')}")
    else:
        print("  ✗ 缺少内容特征")
    
    # 显示下一步操作
    show_next_steps()


def show_next_steps():
    """显示下一步操作建议"""
    print("\n" + "=" * 50)
    print("🚀 下一步操作建议")
    print("=" * 50)
    print("1. 检查提取的投资评级和驱动因素是否准确")
    print("2. 如果满意，可以使用以下命令批量转换所有TXT文件:")
    print("   python -m tools.batch_txt_to_json --txt_dir user_data/txt_files --output_dir user_data/json_files")
    print("3. 如需转换单个文件，可以使用:")
    print("   python -m tools.batch_txt_to_json --single user_data/txt_files/1.txt")
    print("=" * 50)


if __name__ == "__main__":
    test_converter() 
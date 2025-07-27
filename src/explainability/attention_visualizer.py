"""
注意力可视化器
可视化模型的注意力机制，帮助理解模型在异常检测时关注的文本区域
"""

import json
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from io import BytesIO
import colorsys

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib不可用，将使用文本替代方案")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


@dataclass
class AttentionData:
    """注意力数据结构"""
    tokens: List[str]
    attention_weights: List[List[float]]  # [layer, head, token, token] 或简化版本
    layer_names: Optional[List[str]] = None
    head_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 颜色设置
    colormap: str = "Reds"
    background_color: str = "white"
    text_color: str = "black"
    highlight_color: str = "red"
    
    # 尺寸设置
    figure_width: int = 12
    figure_height: int = 8
    font_size: int = 10
    token_spacing: float = 0.1
    
    # 显示设置
    show_values: bool = False
    normalize_weights: bool = True
    threshold: float = 0.1  # 显示阈值
    max_tokens_display: int = 100
    
    # 输出设置
    output_format: str = "png"  # png, svg, html
    dpi: int = 300
    save_raw_data: bool = True


class AttentionVisualizer:
    """
    注意力可视化器
    
    提供多种注意力可视化功能：
    1. 注意力热图 - 显示token之间的注意力权重
    2. 文本高亮 - 在原文中高亮重要区域
    3. 层级分析 - 分析不同层的注意力模式
    4. 头部对比 - 对比不同注意力头的行为
    5. 交互式可视化 - 生成可交互的HTML报告
    6. 统计分析 - 提供注意力分布的统计信息
    
    Args:
        config: 可视化配置
        
    Attributes:
        config: 可视化配置
        output_dir: 输出目录
        color_schemes: 颜色方案
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """初始化注意力可视化器"""
        self.config = config or VisualizationConfig()
        self.file_manager = get_file_manager()
        
        # 输出目录
        self.output_dir = Path("data/visualizations/attention")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色方案
        self.color_schemes = self._initialize_color_schemes()
        
        # 可视化历史
        self.visualization_history = []
        
        logger.info("注意力可视化器初始化完成")
    
    def _initialize_color_schemes(self) -> Dict[str, Dict[str, Any]]:
        """初始化颜色方案"""
        return {
            "default": {
                "low": "#E8F4FD",
                "medium": "#81C7E8", 
                "high": "#1E88E5",
                "text": "#000000"
            },
            "heat": {
                "low": "#FFF5F5",
                "medium": "#FF6B6B",
                "high": "#E53E3E",
                "text": "#000000"
            },
            "cool": {
                "low": "#F0FFF4",
                "medium": "#68D391",
                "high": "#38A169",
                "text": "#000000"
            },
            "purple": {
                "low": "#FAF5FF",
                "medium": "#B794F6",
                "high": "#805AD5",
                "text": "#000000"
            }
        }
    
    def visualize_attention_heatmap(self, attention_data: AttentionData,
                                  layer_idx: int = 0,
                                  head_idx: int = 0,
                                  title: Optional[str] = None) -> str:
        """
        生成注意力热图
        
        Args:
            attention_data: 注意力数据
            layer_idx: 层索引
            head_idx: 头索引
            title: 图表标题
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if not MATPLOTLIB_AVAILABLE:
                return self._create_text_heatmap(attention_data, layer_idx, head_idx)
            
            # 提取注意力权重
            if len(attention_data.attention_weights) <= layer_idx:
                logger.error(f"层索引超出范围: {layer_idx}")
                return ""
            
            layer_attention = attention_data.attention_weights[layer_idx]
            
            # 处理多头注意力
            if isinstance(layer_attention[0], list) and len(layer_attention) > head_idx:
                attention_matrix = np.array(layer_attention[head_idx])
            else:
                attention_matrix = np.array(layer_attention)
            
            # 限制显示的token数量
            max_tokens = min(len(attention_data.tokens), self.config.max_tokens_display)
            attention_matrix = attention_matrix[:max_tokens, :max_tokens]
            display_tokens = attention_data.tokens[:max_tokens]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
            
            # 绘制热图
            im = ax.imshow(attention_matrix, cmap=self.config.colormap, aspect='auto')
            
            # 设置坐标轴
            ax.set_xticks(range(len(display_tokens)))
            ax.set_yticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(display_tokens, fontsize=8)
            
            # 设置标题
            if title is None:
                title = f"注意力热图 - 层{layer_idx}, 头{head_idx}"
            ax.set_title(title, fontsize=self.config.font_size + 2)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('注意力权重', fontsize=self.config.font_size)
            
            # 添加数值标注（如果启用）
            if self.config.show_values:
                for i in range(len(display_tokens)):
                    for j in range(len(display_tokens)):
                        if attention_matrix[i, j] > self.config.threshold:
                            text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                                         ha="center", va="center", color="white", fontsize=6)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_heatmap_L{layer_idx}_H{head_idx}_{timestamp}.{self.config.output_format}"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            # 保存原始数据
            if self.config.save_raw_data:
                data_file = filepath.with_suffix('.json')
                self._save_attention_data(attention_data, layer_idx, head_idx, data_file)
            
            logger.info(f"注意力热图已保存: {filepath}")
            
            # 记录历史
            self.visualization_history.append({
                'type': 'heatmap',
                'file': str(filepath),
                'layer': layer_idx,
                'head': head_idx,
                'timestamp': timestamp
            })
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"注意力热图生成失败: {e}")
            return ""
    
    def visualize_token_attention(self, attention_data: AttentionData,
                                target_token_idx: int,
                                layer_idx: int = 0,
                                head_idx: int = 0) -> str:
        """
        可视化单个token的注意力分布
        
        Args:
            attention_data: 注意力数据
            target_token_idx: 目标token索引
            layer_idx: 层索引
            head_idx: 头索引
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if not MATPLOTLIB_AVAILABLE:
                return self._create_text_token_attention(attention_data, target_token_idx)
            
            # 提取注意力权重
            layer_attention = attention_data.attention_weights[layer_idx]
            if isinstance(layer_attention[0], list):
                attention_weights = np.array(layer_attention[head_idx][target_token_idx])
            else:
                attention_weights = np.array(layer_attention[target_token_idx])
            
            # 限制显示的token数量
            max_tokens = min(len(attention_data.tokens), self.config.max_tokens_display)
            attention_weights = attention_weights[:max_tokens]
            display_tokens = attention_data.tokens[:max_tokens]
            
            # 创建条形图
            fig, ax = plt.subplots(figsize=(self.config.figure_width, 6))
            
            # 按注意力权重排序
            sorted_indices = np.argsort(attention_weights)[::-1]
            top_n = min(20, len(sorted_indices))  # 显示前20个
            
            top_indices = sorted_indices[:top_n]
            top_weights = attention_weights[top_indices]
            top_tokens = [display_tokens[i] for i in top_indices]
            
            # 绘制条形图
            bars = ax.barh(range(len(top_tokens)), top_weights, color=plt.cm.Reds(top_weights))
            
            # 设置标签
            ax.set_yticks(range(len(top_tokens)))
            ax.set_yticklabels(top_tokens, fontsize=self.config.font_size)
            ax.set_xlabel('注意力权重', fontsize=self.config.font_size)
            ax.set_title(f'Token "{attention_data.tokens[target_token_idx]}" 的注意力分布', 
                        fontsize=self.config.font_size + 2)
            
            # 添加数值标注
            for i, (bar, weight) in enumerate(zip(bars, top_weights)):
                ax.text(weight + 0.001, i, f'{weight:.3f}', 
                       va='center', fontsize=8)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"token_attention_{target_token_idx}_{timestamp}.{self.config.output_format}"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Token注意力图已保存: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Token注意力可视化失败: {e}")
            return ""
    
    def create_attention_text_highlight(self, text: str, 
                                      attention_weights: List[float],
                                      tokens: List[str]) -> str:
        """
        创建高亮文本的HTML
        
        Args:
            text: 原始文本
            attention_weights: 注意力权重
            tokens: token列表
            
        Returns:
            str: HTML字符串
        """
        try:
            # 标准化注意力权重
            if self.config.normalize_weights:
                max_weight = max(attention_weights) if attention_weights else 1.0
                min_weight = min(attention_weights) if attention_weights else 0.0
                weight_range = max_weight - min_weight
                
                if weight_range > 0:
                    normalized_weights = [(w - min_weight) / weight_range for w in attention_weights]
                else:
                    normalized_weights = [0.5] * len(attention_weights)
            else:
                normalized_weights = attention_weights
            
            # 生成HTML
            html_parts = ['<div style="font-family: Arial, sans-serif; line-height: 1.5;">']
            
            for i, (token, weight) in enumerate(zip(tokens, normalized_weights)):
                if weight > self.config.threshold:
                    # 计算颜色强度
                    intensity = min(weight, 1.0)
                    
                    # 生成背景颜色
                    r, g, b = self._weight_to_color(intensity)
                    background_color = f"rgba({r}, {g}, {b}, {intensity * 0.7})"
                    
                    # 创建高亮span
                    html_parts.append(
                        f'<span style="background-color: {background_color}; '
                        f'padding: 2px; margin: 1px; border-radius: 3px; '
                        f'title="权重: {weight:.3f}">{token}</span>'
                    )
                else:
                    html_parts.append(f'<span>{token}</span>')
                
                # 添加空格（除了最后一个token）
                if i < len(tokens) - 1:
                    html_parts.append(' ')
            
            html_parts.append('</div>')
            
            # 添加图例
            legend_html = self._create_attention_legend()
            
            return '\n'.join(html_parts) + '\n' + legend_html
            
        except Exception as e:
            logger.error(f"文本高亮生成失败: {e}")
            return f"<div>文本高亮生成失败: {str(e)}</div>"
    
    def _weight_to_color(self, weight: float) -> Tuple[int, int, int]:
        """将权重转换为RGB颜色"""
        # 使用红色渐变
        r = int(255 * weight)
        g = int(255 * (1 - weight * 0.8))
        b = int(255 * (1 - weight * 0.8))
        
        return r, g, b
    
    def _create_attention_legend(self) -> str:
        """创建注意力图例"""
        return '''
        <div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <strong>注意力权重图例:</strong><br>
            <span style="background-color: rgba(255, 200, 200, 0.3); padding: 2px;">低注意力</span>
            <span style="background-color: rgba(255, 150, 150, 0.5); padding: 2px;">中等注意力</span>
            <span style="background-color: rgba(255, 100, 100, 0.7); padding: 2px;">高注意力</span>
        </div>
        '''
    
    def create_multi_layer_comparison(self, attention_data: AttentionData,
                                    target_token_idx: int,
                                    layers: Optional[List[int]] = None) -> str:
        """
        创建多层注意力对比图
        
        Args:
            attention_data: 注意力数据
            target_token_idx: 目标token索引
            layers: 要对比的层（默认为所有层）
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if not MATPLOTLIB_AVAILABLE:
                return self._create_text_layer_comparison(attention_data, target_token_idx, layers)
            
            if layers is None:
                layers = list(range(len(attention_data.attention_weights)))
            
            # 限制层数
            layers = layers[:6]  # 最多显示6层
            
            fig, axes = plt.subplots(len(layers), 1, figsize=(self.config.figure_width, 3 * len(layers)))
            
            if len(layers) == 1:
                axes = [axes]
            
            target_token = attention_data.tokens[target_token_idx]
            
            for i, layer_idx in enumerate(layers):
                ax = axes[i]
                
                # 提取该层的注意力权重
                layer_attention = attention_data.attention_weights[layer_idx]
                if isinstance(layer_attention[0], list):
                    # 多头注意力，取平均
                    attention_weights = np.mean(layer_attention, axis=0)[target_token_idx]
                else:
                    attention_weights = np.array(layer_attention[target_token_idx])
                
                # 限制显示的token数量
                max_tokens = min(len(attention_data.tokens), 50)
                attention_weights = attention_weights[:max_tokens]
                display_tokens = attention_data.tokens[:max_tokens]
                
                # 绘制条形图
                bars = ax.bar(range(len(display_tokens)), attention_weights, 
                             color=plt.cm.Blues(0.7), alpha=0.8)
                
                # 设置标签
                ax.set_title(f'层 {layer_idx} - Token "{target_token}" 的注意力', fontsize=10)
                ax.set_ylabel('注意力权重', fontsize=9)
                
                # 只在最后一个子图显示x轴标签
                if i == len(layers) - 1:
                    ax.set_xticks(range(len(display_tokens)))
                    ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=7)
                else:
                    ax.set_xticks([])
                
                # 高亮最高注意力的token
                max_idx = np.argmax(attention_weights)
                bars[max_idx].set_color('red')
                bars[max_idx].set_alpha(1.0)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_layer_attention_{target_token_idx}_{timestamp}.{self.config.output_format}"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"多层注意力对比图已保存: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"多层注意力对比生成失败: {e}")
            return ""
    
    def _create_text_heatmap(self, attention_data: AttentionData, 
                           layer_idx: int, head_idx: int) -> str:
        """创建文本版本的注意力热图"""
        try:
            # 提取注意力权重
            layer_attention = attention_data.attention_weights[layer_idx]
            if isinstance(layer_attention[0], list):
                attention_matrix = layer_attention[head_idx]
            else:
                attention_matrix = layer_attention
            
            # 生成文本报告
            report_lines = [
                f"注意力热图 - 层{layer_idx}, 头{head_idx}",
                "=" * 50,
                f"Tokens: {len(attention_data.tokens)}",
                ""
            ]
            
            # 显示top attention pairs
            top_pairs = []
            for i, token_i in enumerate(attention_data.tokens[:20]):  # 限制显示
                for j, token_j in enumerate(attention_data.tokens[:20]):
                    if i < len(attention_matrix) and j < len(attention_matrix[i]):
                        weight = attention_matrix[i][j]
                        if weight > self.config.threshold:
                            top_pairs.append((weight, token_i, token_j, i, j))
            
            # 按权重排序
            top_pairs.sort(reverse=True)
            
            report_lines.append("Top 注意力对:")
            for weight, token_i, token_j, i, j in top_pairs[:10]:
                report_lines.append(f"  {token_i} -> {token_j}: {weight:.3f}")
            
            # 保存文本报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_heatmap_L{layer_idx}_H{head_idx}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"文本注意力报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"文本热图生成失败: {e}")
            return ""
    
    def _create_text_token_attention(self, attention_data: AttentionData, 
                                   target_token_idx: int) -> str:
        """创建文本版本的token注意力分析"""
        try:
            target_token = attention_data.tokens[target_token_idx]
            
            # 假设使用第一层第一头的注意力
            layer_attention = attention_data.attention_weights[0]
            if isinstance(layer_attention[0], list):
                attention_weights = layer_attention[0][target_token_idx]
            else:
                attention_weights = layer_attention[target_token_idx]
            
            # 创建排序列表
            token_weights = list(zip(attention_data.tokens, attention_weights))
            token_weights.sort(key=lambda x: x[1], reverse=True)
            
            # 生成报告
            report_lines = [
                f'Token "{target_token}" 注意力分析',
                "=" * 40,
                ""
            ]
            
            for i, (token, weight) in enumerate(token_weights[:20]):
                if weight > self.config.threshold:
                    report_lines.append(f"{i+1:2d}. {token:<15} {weight:.4f}")
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"token_attention_{target_token_idx}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"文本token注意力分析失败: {e}")
            return ""
    
    def _create_text_layer_comparison(self, attention_data: AttentionData,
                                    target_token_idx: int, 
                                    layers: Optional[List[int]]) -> str:
        """创建文本版本的层对比分析"""
        try:
            if layers is None:
                layers = list(range(min(len(attention_data.attention_weights), 6)))
            
            target_token = attention_data.tokens[target_token_idx]
            
            report_lines = [
                f'多层注意力对比 - Token "{target_token}"',
                "=" * 50,
                ""
            ]
            
            for layer_idx in layers:
                layer_attention = attention_data.attention_weights[layer_idx]
                if isinstance(layer_attention[0], list):
                    # 多头平均
                    attention_weights = [sum(head[target_token_idx] for head in layer_attention) / len(layer_attention)]
                else:
                    attention_weights = layer_attention[target_token_idx]
                
                # 找到最高注意力的token
                max_attention_idx = attention_weights.index(max(attention_weights))
                max_token = attention_data.tokens[max_attention_idx]
                max_weight = attention_weights[max_attention_idx]
                
                report_lines.append(f"层 {layer_idx}: 最关注 '{max_token}' (权重: {max_weight:.4f})")
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"layer_comparison_{target_token_idx}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"文本层对比分析失败: {e}")
            return ""
    
    def _save_attention_data(self, attention_data: AttentionData,
                           layer_idx: int, head_idx: int, filepath: Path):
        """保存注意力原始数据"""
        try:
            data = {
                'tokens': attention_data.tokens,
                'layer_idx': layer_idx,
                'head_idx': head_idx,
                'attention_weights': attention_data.attention_weights[layer_idx],
                'metadata': attention_data.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存注意力数据失败: {e}")
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """获取可视化摘要"""
        return {
            'total_visualizations': len(self.visualization_history),
            'output_directory': str(self.output_dir),
            'config': {
                'colormap': self.config.colormap,
                'figure_size': (self.config.figure_width, self.config.figure_height),
                'output_format': self.config.output_format,
                'threshold': self.config.threshold
            },
            'recent_visualizations': self.visualization_history[-5:] if self.visualization_history else [],
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'seaborn_available': SEABORN_AVAILABLE
        }


# 工厂函数
def create_attention_visualizer(config_dict: Optional[Dict[str, Any]] = None) -> AttentionVisualizer:
    """
    创建注意力可视化器实例
    
    Args:
        config_dict: 配置字典
        
    Returns:
        AttentionVisualizer: 可视化器实例
    """
    config = VisualizationConfig(**config_dict) if config_dict else VisualizationConfig()
    return AttentionVisualizer(config)


if __name__ == "__main__":
    # 使用示例
    print("注意力可视化器测试:")
    
    # 创建可视化器
    visualizer = AttentionVisualizer()
    
    # 模拟注意力数据
    tokens = ["这", "是", "一", "份", "测试", "研报", "内容"]
    
    # 模拟注意力权重 (简化版本)
    attention_weights = [
        [  # 第一层
            [0.1, 0.2, 0.15, 0.1, 0.25, 0.15, 0.05],  # token 0的注意力
            [0.05, 0.3, 0.2, 0.15, 0.1, 0.15, 0.05],  # token 1的注意力
            [0.1, 0.15, 0.4, 0.1, 0.1, 0.1, 0.05],    # token 2的注意力
            [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05],  # token 3的注意力
            [0.2, 0.1, 0.1, 0.1, 0.3, 0.15, 0.05],    # token 4的注意力
            [0.1, 0.1, 0.1, 0.15, 0.2, 0.3, 0.05],    # token 5的注意力
            [0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.5]   # token 6的注意力
        ]
    ]
    
    attention_data = AttentionData(
        tokens=tokens,
        attention_weights=attention_weights,
        metadata={'model': 'test_model', 'layer_count': 1}
    )
    
    print(f"创建注意力数据: {len(tokens)}个tokens, {len(attention_weights)}层")
    
    # 测试文本高亮
    highlight_html = visualizer.create_attention_text_highlight(
        text="这是一份测试研报内容",
        attention_weights=[0.1, 0.2, 0.15, 0.1, 0.25, 0.15, 0.05],
        tokens=tokens
    )
    
    if highlight_html:
        print("✅ 文本高亮生成成功")
        
        # 保存HTML示例
        html_file = visualizer.output_dir / "test_highlight.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>注意力高亮测试</title></head>
            <body>
            <h1>注意力高亮测试</h1>
            {highlight_html}
            </body>
            </html>
            """)
        print(f"HTML示例已保存: {html_file}")
    
    # 测试可视化（如果matplotlib可用）
    if MATPLOTLIB_AVAILABLE:
        heatmap_file = visualizer.visualize_attention_heatmap(attention_data)
        if heatmap_file:
            print(f"✅ 注意力热图生成成功: {heatmap_file}")
    else:
        text_heatmap = visualizer._create_text_heatmap(attention_data, 0, 0)
        if text_heatmap:
            print(f"✅ 文本热图生成成功: {text_heatmap}")
    
    # 获取摘要
    summary = visualizer.get_visualization_summary()
    print(f"\n可视化摘要: {summary}") 
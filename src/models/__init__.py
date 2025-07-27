"""
模型模块
包含Qwen模型封装、多模态融合、注意力层等
"""

# 目前只有qwen_wrapper可用，其他模块待实现
try:
    from .qwen_wrapper import QwenWrapper, get_qwen_model
    __all__ = ['QwenWrapper', 'get_qwen_model']
except ImportError:
    # 如果导入失败，提供空的__all__
    __all__ = []

# 未来将添加:
# from .multimodal_fusion import MultimodalFusion
# from .attention_layers import CrossModalAttention 
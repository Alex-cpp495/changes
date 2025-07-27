"""
Qwen2.5-7B模型封装器
实现模型加载、推理、微调等核心功能，支持CPU/GPU模式和8位量化
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
import json
import numpy as np
from pathlib import Path
import gc
import psutil
from datetime import datetime

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class QwenWrapper:
    """
    Qwen2.5-7B模型封装器
    
    支持功能：
    - 模型加载与量化
    - 文本生成和多任务推理
    - LoRA微调支持
    - CPU/GPU自适应
    - 内存优化管理
    
    Args:
        model_config_path: 模型配置文件路径
        device: 计算设备，"auto"为自动选择
        
    Attributes:
        model: 加载的模型实例
        tokenizer: 分词器
        device: 当前设备
        generation_config: 生成配置
        is_loaded: 模型是否已加载
    """
    
    def __init__(self, model_config_path: Optional[str] = None, device: str = "auto"):
        """初始化Qwen模型封装器"""
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.is_loaded = False
        self.file_manager = get_file_manager()
        
        # 加载配置
        if model_config_path is None:
            model_config_path = "configs/model_config.yaml"
        
        try:
            self.config = load_config(model_config_path)
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            # 使用默认配置
            self.config = self._get_default_config()
        
        # 设备配置
        self.device = self._setup_device(device)
        logger.info(f"使用设备: {self.device}")
        
        # 监控配置
        self.memory_threshold = 0.85  # 内存使用阈值
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "model": {
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "quantization": {
                    "enabled": True,
                    "method": "4bit_nf4",
                    "compute_dtype": "float16"
                },
                "max_sequence_length": 2048,
                "device_map": "auto",
                "low_cpu_mem_usage": True
            },
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "inference": {
                "batch_size": 1,
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 512,
                "do_sample": True
            }
        }
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("未检测到CUDA设备，使用CPU模式")
        
        # CPU模式优化
        if device == "cpu":
            torch.set_num_threads(min(psutil.cpu_count(logical=False), 24))
            logger.info(f"CPU线程数设置为: {torch.get_num_threads()}")
        
        return device
    
    def load_model(self, enable_lora: bool = False) -> bool:
        """
        加载Qwen模型和分词器
        
        Args:
            enable_lora: 是否启用LoRA配置
            
        Returns:
            bool: 加载是否成功
            
        Raises:
            RuntimeError: 模型加载失败
            MemoryError: 内存不足
        """
        try:
            logger.info("开始加载Qwen2.5-7B模型...")
            start_time = datetime.now()
            
            model_name = self.config["model"]["base_model"]
            max_length = self.config["model"]["max_sequence_length"]
            
            # 配置量化
            quantization_config = None
            if self.config["model"]["quantization"]["enabled"]:
                quantization_config = self._get_quantization_config()
                logger.info("启用4bit量化以节省内存")
            
            # 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side=self.config.get("tokenizer", {}).get("padding_side", "left"),
                use_fast=True
            )
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info("加载模型...")
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": self.config["model"]["low_cpu_mem_usage"],
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.device == "cpu":
                model_kwargs["device_map"] = None
            else:
                model_kwargs["device_map"] = self.config["model"]["device_map"]
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # CPU模式下手动移动到CPU
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            # 配置LoRA
            if enable_lora:
                self._setup_lora()
            
            # 设置生成配置
            self._setup_generation_config()
            
            # 设置评估模式
            self.model.eval()
            
            self.is_loaded = True
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"模型加载成功！耗时: {load_time:.2f}秒")
            
            # 显示内存使用情况
            self._log_memory_usage()
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self._cleanup()
            raise RuntimeError(f"Qwen模型加载失败: {str(e)}")
    
    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """获取量化配置"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True if self.device == "cpu" else False
        )
    
    def _setup_lora(self):
        """设置LoRA配置"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config["lora_config"]["rank"],
            lora_alpha=self.config["lora_config"]["alpha"],
            lora_dropout=self.config["lora_config"]["dropout"],
            target_modules=self.config["lora_config"]["target_modules"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA配置已启用")
    
    def _setup_generation_config(self):
        """设置文本生成配置"""
        inference_config = self.config["inference"]
        
        self.generation_config = GenerationConfig(
            max_new_tokens=inference_config["max_new_tokens"],
            temperature=inference_config["temperature"],
            top_p=inference_config["top_p"],
            do_sample=inference_config["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
        )
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: 核采样参数
            do_sample: 是否采样
            
        Returns:
            str: 生成的文本
            
        Raises:
            RuntimeError: 模型未加载或生成失败
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            # 动态配置生成参数
            gen_config = self.generation_config
            if max_new_tokens:
                gen_config.max_new_tokens = max_new_tokens
            if temperature:
                gen_config.temperature = temperature
            if top_p:
                gen_config.top_p = top_p
            if do_sample is not None:
                gen_config.do_sample = do_sample
            
            # 编码输入
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.config["model"]["max_sequence_length"] - gen_config.max_new_tokens
            )
            inputs = inputs.to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=gen_config,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            raise RuntimeError(f"文本生成失败: {str(e)}")
    
    def multi_task_inference(
        self, 
        text: str, 
        tasks: List[str] = None
    ) -> Dict[str, Any]:
        """
        多任务推理
        
        Args:
            text: 输入文本
            tasks: 任务列表，默认为["sentiment_analysis", "style_recognition", "anomaly_pre_detection"]
            
        Returns:
            Dict[str, Any]: 各任务的推理结果
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        if tasks is None:
            tasks = ["sentiment_analysis", "style_recognition", "anomaly_pre_detection"]
        
        results = {}
        
        try:
            for task in tasks:
                prompt = self._build_task_prompt(task, text)
                result = self.generate_text(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True
                )
                results[task] = self._parse_task_result(task, result)
                
        except Exception as e:
            logger.error(f"多任务推理失败: {str(e)}")
            raise
        
        return results
    
    def _build_task_prompt(self, task: str, text: str) -> str:
        """构建任务特定的提示"""
        prompts = {
            "sentiment_analysis": f"请分析以下研报文本的情感倾向，回答积极、中性或消极：\n\n{text}\n\n情感倾向：",
            "style_recognition": f"请分析以下研报文本的写作风格，回答保守、中性或激进：\n\n{text}\n\n写作风格：",
            "anomaly_pre_detection": f"请评估以下研报文本是否有异常，回答正常、可疑或异常：\n\n{text}\n\n异常评估："
        }
        
        return prompts.get(task, f"请分析以下文本：\n\n{text}\n\n分析结果：")
    
    def _parse_task_result(self, task: str, result: str) -> Dict[str, Any]:
        """解析任务结果"""
        result = result.strip().lower()
        
        if task == "sentiment_analysis":
            if "积极" in result or "positive" in result:
                return {"sentiment": "positive", "confidence": 0.8}
            elif "消极" in result or "negative" in result:
                return {"sentiment": "negative", "confidence": 0.8}
            else:
                return {"sentiment": "neutral", "confidence": 0.6}
                
        elif task == "style_recognition":
            if "激进" in result or "aggressive" in result:
                return {"style": "aggressive", "confidence": 0.8}
            elif "保守" in result or "conservative" in result:
                return {"style": "conservative", "confidence": 0.8}
            else:
                return {"style": "neutral", "confidence": 0.6}
                
        elif task == "anomaly_pre_detection":
            if "异常" in result or "anomaly" in result:
                return {"anomaly": "anomaly", "confidence": 0.8}
            elif "可疑" in result or "suspicious" in result:
                return {"anomaly": "suspicious", "confidence": 0.7}
            else:
                return {"anomaly": "normal", "confidence": 0.6}
        
        return {"result": result, "confidence": 0.5}
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 嵌入向量矩阵
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        embeddings = []
        
        try:
            for text in texts:
                inputs = self.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config["model"]["max_sequence_length"]
                )
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(inputs, output_hidden_states=True)
                    # 使用最后一层的隐藏状态
                    hidden_states = outputs.hidden_states[-1]
                    # 平均池化
                    embedding = hidden_states.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"获取文本嵌入失败: {str(e)}")
            raise
    
    def _log_memory_usage(self):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"GPU内存使用: {gpu_memory:.2f}GB / 峰值: {gpu_memory_max:.2f}GB")
        
        ram_usage = psutil.virtual_memory()
        ram_used = ram_usage.used / 1024**3
        ram_total = ram_usage.total / 1024**3
        ram_percent = ram_usage.percent
        
        logger.info(f"RAM使用: {ram_used:.2f}GB / {ram_total:.2f}GB ({ram_percent:.1f}%)")
        
        if ram_percent > self.memory_threshold * 100:
            logger.warning(f"内存使用率过高: {ram_percent:.1f}%")
    
    def _cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("模型资源已清理")
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # 保存配置
            config_path = save_path / "qwen_wrapper_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模型已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def __del__(self):
        """析构函数，清理资源"""
        self._cleanup()


# 全局模型实例管理
_global_qwen_model = None


def get_qwen_model() -> QwenWrapper:
    """
    获取全局Qwen模型实例
    
    Returns:
        QwenWrapper: Qwen模型实例
    """
    global _global_qwen_model
    
    if _global_qwen_model is None:
        _global_qwen_model = QwenWrapper()
        if not _global_qwen_model.is_loaded:
            _global_qwen_model.load_model()
    
    return _global_qwen_model


def clear_qwen_model():
    """清理全局模型实例"""
    global _global_qwen_model
    
    if _global_qwen_model is not None:
        _global_qwen_model._cleanup()
        _global_qwen_model = None


if __name__ == "__main__":
    # 使用示例
    model = QwenWrapper()
    
    # 加载模型
    model.load_model()
    
    # 文本生成测试
    text = "近期A股市场表现"
    result = model.generate_text(f"请分析：{text}")
    print(f"生成结果：{result}")
    
    # 多任务推理测试  
    multi_result = model.multi_task_inference(text)
    print(f"多任务结果：{multi_result}") 
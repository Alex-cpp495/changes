"""
Qwen模型微调器
专门用于Qwen模型的LoRA微调和多任务学习
"""

import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import numpy as np

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager
from .trainer import Trainer, TrainingConfig, TrainingPhase
from .loss_functions import MultiTaskLoss, AnomalyDetectionLoss

logger = get_logger(__name__)


@dataclass
class QwenFineTuningConfig:
    """Qwen微调配置"""
    # 基础配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_path: Optional[str] = None
    
    # LoRA配置
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 量化配置
    use_quantization: bool = True
    quantization_type: str = "nf4"  # nf4, fp4, int8
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # 训练配置
    max_seq_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    epochs: int = 3
    
    # 多任务配置
    enable_multi_task: bool = True
    task_weights: Dict[str, float] = None
    
    # 保存配置
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 2
    
    # 验证配置
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    
    # 其他配置
    fp16: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    report_to: str = "none"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Qwen2.5的默认LoRA目标模块
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.task_weights is None:
            self.task_weights = {
                "sentiment_analysis": 0.3,
                "style_recognition": 0.3,
                "anomaly_detection": 0.4
            }


class QwenFineTuner:
    """
    Qwen模型微调器
    
    提供Qwen模型的专业微调功能：
    1. LoRA微调 - 高效参数微调，节省显存
    2. 量化支持 - 4bit/8bit量化，适应有限硬件
    3. 多任务学习 - 同时训练多个下游任务
    4. 数据处理 - 自动数据预处理和格式化
    5. 模型评估 - 多维度模型性能评估
    6. 增量训练 - 支持从检查点继续训练
    
    Args:
        config: Qwen微调配置
        
    Attributes:
        config: 微调配置
        model: Qwen模型实例
        tokenizer: 分词器
        trainer: 底层训练器
    """
    
    def __init__(self, config: QwenFineTuningConfig):
        """初始化Qwen微调器"""
        self.config = config
        self.file_manager = get_file_manager()
        
        # 模型组件
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # 训练组件
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # 任务处理器
        self.task_processors = {}
        
        # 输出目录
        self.output_dir = Path(f"data/qwen_finetuning/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.training_stats = {
            'total_training_time': 0.0,
            'total_examples': 0,
            'task_performance': {},
            'best_metrics': {}
        }
        
        logger.info("Qwen微调器初始化完成")
    
    def load_model_and_tokenizer(self) -> bool:
        """
        加载模型和分词器
        
        Returns:
            bool: 加载是否成功
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, TaskType
            
            logger.info(f"开始加载模型: {self.config.model_name}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 量化配置
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                    bnb_4bit_quant_type=self.config.quantization_type,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 设置LoRA
            if self.config.use_lora:
                self._setup_lora()
            
            logger.info("模型和分词器加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _setup_lora(self):
        """设置LoRA配置"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # 创建LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none"
            )
            
            # 应用LoRA
            self.peft_model = get_peft_model(self.model, lora_config)
            self.model = self.peft_model
            
            # 打印可训练参数
            self.model.print_trainable_parameters()
            
            logger.info("LoRA配置设置成功")
            
        except Exception as e:
            logger.error(f"LoRA设置失败: {e}")
            raise
    
    def prepare_datasets(self, train_data: List[Dict[str, Any]], 
                        eval_data: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        准备训练数据集
        
        Args:
            train_data: 训练数据
            eval_data: 验证数据
            
        Returns:
            bool: 准备是否成功
        """
        try:
            logger.info(f"准备数据集: 训练{len(train_data)}条, 验证{len(eval_data) if eval_data else 0}条")
            
            # 处理训练数据
            self.train_dataset = self._process_dataset(train_data, is_training=True)
            
            # 处理验证数据
            if eval_data:
                self.eval_dataset = self._process_dataset(eval_data, is_training=False)
            
            self.training_stats['total_examples'] = len(train_data)
            
            logger.info("数据集准备完成")
            return True
            
        except Exception as e:
            logger.error(f"数据集准备失败: {e}")
            return False
    
    def _process_dataset(self, data: List[Dict[str, Any]], is_training: bool = True) -> List[Dict[str, Any]]:
        """
        处理数据集
        
        Args:
            data: 原始数据
            is_training: 是否为训练数据
            
        Returns:
            List[Dict[str, Any]]: 处理后的数据
        """
        processed_data = []
        
        for item in data:
            try:
                if self.config.enable_multi_task:
                    # 多任务数据处理
                    processed_item = self._process_multi_task_item(item)
                else:
                    # 单任务数据处理
                    processed_item = self._process_single_task_item(item)
                
                if processed_item:
                    processed_data.append(processed_item)
                    
            except Exception as e:
                logger.warning(f"数据处理失败: {e}")
                continue
        
        logger.info(f"数据处理完成: {len(processed_data)}/{len(data)}")
        return processed_data
    
    def _process_multi_task_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理多任务数据项"""
        try:
            # 基础文本
            content = item.get('content', '')
            if not content:
                return None
            
            # 构建多任务prompt
            prompts = []
            labels = []
            
            # 情感分析任务
            if 'sentiment' in item:
                sentiment_prompt = f"请分析以下研报的情感倾向：\n{content}\n情感："
                sentiment_label = item['sentiment']
                prompts.append(('sentiment_analysis', sentiment_prompt, sentiment_label))
            
            # 风格识别任务
            if 'style' in item:
                style_prompt = f"请识别以下研报的写作风格：\n{content}\n风格："
                style_label = item['style']
                prompts.append(('style_recognition', style_prompt, style_label))
            
            # 异常检测任务
            if 'is_anomaly' in item:
                anomaly_prompt = f"请判断以下研报是否存在异常：\n{content}\n异常："
                anomaly_label = "是" if item['is_anomaly'] else "否"
                prompts.append(('anomaly_detection', anomaly_prompt, anomaly_label))
            
            if not prompts:
                return None
            
            # 随机选择一个任务（训练时）或使用所有任务（评估时）
            import random
            selected_task, prompt, label = random.choice(prompts)
            
            # 分词
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            # 添加标签
            label_tokenized = self.tokenizer(
                label,
                truncation=True,
                max_length=50,  # 标签长度限制
                padding=False,
                return_tensors=None
            )
            
            # 合并输入和标签
            input_ids = tokenized['input_ids'] + label_tokenized['input_ids']
            attention_mask = tokenized['attention_mask'] + label_tokenized['attention_mask']
            
            # 创建labels（用于计算损失）
            labels = [-100] * len(tokenized['input_ids']) + label_tokenized['input_ids']
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'task_type': selected_task,
                'original_item': item
            }
            
        except Exception as e:
            logger.error(f"多任务数据处理失败: {e}")
            return None
    
    def _process_single_task_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单任务数据项"""
        try:
            content = item.get('content', '')
            label = item.get('label', '')
            
            if not content or not label:
                return None
            
            # 构建prompt
            prompt = f"请分析以下研报内容：\n{content}\n分析结果："
            
            # 分词
            tokenized = self.tokenizer(
                prompt + label,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            # 创建labels
            prompt_length = len(self.tokenizer(prompt)['input_ids'])
            labels = [-100] * prompt_length + tokenized['input_ids'][prompt_length:]
            
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels,
                'original_item': item
            }
            
        except Exception as e:
            logger.error(f"单任务数据处理失败: {e}")
            return None
    
    def create_trainer(self) -> bool:
        """
        创建训练器
        
        Returns:
            bool: 创建是否成功
        """
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # 创建训练参数
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                num_train_epochs=self.config.epochs,
                fp16=self.config.fp16,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                evaluation_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                logging_steps=10,
                report_to=self.config.report_to,
                dataloader_num_workers=self.config.dataloader_num_workers,
                remove_unused_columns=self.config.remove_unused_columns,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # 创建数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics if self.config.enable_multi_task else None
            )
            
            logger.info("训练器创建成功")
            return True
            
        except Exception as e:
            logger.error(f"训练器创建失败: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """
        开始微调训练
        
        Returns:
            Dict[str, Any]: 训练结果
        """
        if not self.trainer:
            raise ValueError("训练器未初始化，请先调用create_trainer()")
        
        try:
            logger.info("开始Qwen模型微调...")
            start_time = time.time()
            
            # 开始训练
            train_result = self.trainer.train()
            
            # 记录训练时间
            training_time = time.time() - start_time
            self.training_stats['total_training_time'] = training_time
            
            # 保存模型
            self._save_model()
            
            # 评估模型
            if self.eval_dataset:
                eval_results = self.trainer.evaluate()
                self.training_stats['best_metrics'] = eval_results
            
            # 生成训练报告
            training_report = self._generate_training_report(train_result)
            
            logger.info(f"微调完成，耗时: {training_time:.2f}秒")
            return training_report
            
        except Exception as e:
            logger.error(f"微调训练失败: {e}")
            raise Exception(f"训练失败: {e}")
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """计算评估指标"""
        try:
            predictions, labels = eval_pred
            
            # 简化的指标计算
            # 实际应用中需要根据具体任务计算相应指标
            metrics = {}
            
            # 计算困惑度
            import torch
            import torch.nn.functional as F
            
            predictions = torch.tensor(predictions)
            labels = torch.tensor(labels)
            
            # 只计算非-100标签的损失
            mask = labels != -100
            if mask.sum() > 0:
                loss = F.cross_entropy(
                    predictions[mask].view(-1, predictions.size(-1)),
                    labels[mask].view(-1),
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
                metrics['perplexity'] = perplexity
            
            return metrics
            
        except Exception as e:
            logger.error(f"指标计算失败: {e}")
            return {}
    
    def _save_model(self):
        """保存微调后的模型"""
        try:
            # 保存LoRA适配器
            if self.config.use_lora and self.peft_model:
                adapter_path = self.output_dir / "lora_adapter"
                self.peft_model.save_pretrained(adapter_path)
                logger.info(f"LoRA适配器已保存: {adapter_path}")
            
            # 保存分词器
            tokenizer_path = self.output_dir / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            
            # 保存配置
            config_path = self.output_dir / "finetuning_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info("模型保存完成")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def _generate_training_report(self, train_result: Any) -> Dict[str, Any]:
        """生成训练报告"""
        report = {
            'training_summary': {
                'model_name': self.config.model_name,
                'use_lora': self.config.use_lora,
                'use_quantization': self.config.use_quantization,
                'multi_task': self.config.enable_multi_task,
                'total_examples': self.training_stats['total_examples'],
                'training_time': self.training_stats['total_training_time'],
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'model_config': asdict(self.config),
            'training_stats': self.training_stats,
            'output_directory': str(self.output_dir),
            'generated_at': datetime.now().isoformat()
        }
        
        if hasattr(train_result, 'training_loss'):
            report['final_training_loss'] = train_result.training_loss
        
        # 保存报告
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"训练报告已保存: {report_path}")
        return report
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估微调后的模型
        
        Args:
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            logger.info(f"开始模型评估，测试数据: {len(test_data)}条")
            
            # 处理测试数据
            test_dataset = self._process_dataset(test_data, is_training=False)
            
            # 执行评估
            if self.trainer:
                eval_results = self.trainer.evaluate(test_dataset)
            else:
                eval_results = self._manual_evaluate(test_dataset)
            
            logger.info("模型评估完成")
            return eval_results
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {'error': str(e)}
    
    def _manual_evaluate(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """手动评估模型"""
        # 简化的评估实现
        return {
            'test_samples': len(test_dataset),
            'evaluation_method': 'manual',
            'timestamp': datetime.now().isoformat()
        }
    
    def inference(self, text: str, task_type: str = "anomaly_detection") -> str:
        """
        使用微调后的模型进行推理
        
        Args:
            text: 输入文本
            task_type: 任务类型
            
        Returns:
            str: 推理结果
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("模型或分词器未加载")
            
            # 构建prompt
            if task_type == "sentiment_analysis":
                prompt = f"请分析以下研报的情感倾向：\n{text}\n情感："
            elif task_type == "style_recognition":
                prompt = f"请识别以下研报的写作风格：\n{text}\n风格："
            elif task_type == "anomaly_detection":
                prompt = f"请判断以下研报是否存在异常：\n{text}\n异常："
            else:
                prompt = f"请分析以下研报内容：\n{text}\n分析结果："
            
            # 分词
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            )
            
            # 推理
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码结果
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分
            generated_part = response[len(prompt):].strip()
            
            return generated_part
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"推理错误: {str(e)}"


# 工厂函数
def create_qwen_fine_tuner(config_dict: Dict[str, Any]) -> QwenFineTuner:
    """
    创建Qwen微调器实例
    
    Args:
        config_dict: 配置字典
        
    Returns:
        QwenFineTuner: 微调器实例
    """
    config = QwenFineTuningConfig(**config_dict)
    return QwenFineTuner(config)


if __name__ == "__main__":
    # 使用示例
    print("Qwen微调器测试:")
    
    # 创建配置
    config = QwenFineTuningConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_lora=True,
        epochs=1,
        batch_size=2
    )
    
    # 创建微调器
    fine_tuner = QwenFineTuner(config)
    
    print(f"微调器创建成功")
    print(f"配置: LoRA={config.use_lora}, 量化={config.use_quantization}")
    print(f"输出目录: {fine_tuner.output_dir}")
    
    # 测试数据处理
    test_data = [
        {
            'content': '这是一份测试研报内容',
            'sentiment': '积极',
            'style': '正式',
            'is_anomaly': False
        }
    ]
    
    processed = fine_tuner._process_multi_task_item(test_data[0])
    if processed:
        print(f"数据处理成功: {list(processed.keys())}")
    
    print("注意：实际训练需要安装transformers、peft等依赖") 
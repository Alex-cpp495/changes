#!/usr/bin/env python3
"""
使用JSON研报数据进行模型训练
支持异常检测模型的训练和验证
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.training.trainer import Trainer, TrainingConfig
from src.data_processing.processors.report_processor import ReportProcessor
from src.anomaly_detection import get_ensemble_detector
from src.continuous_learning import get_continuous_learning_system

logger = get_logger(__name__)


class JSONDataTrainer:
    """JSON数据训练器"""
    
    def __init__(self, json_files_dir: str = "user_data/json_files"):
        """
        初始化训练器
        
        Args:
            json_files_dir: JSON文件目录
        """
        self.json_files_dir = Path(project_root) / json_files_dir
        self.training_data = []
        self.validation_data = []
        self.test_data = []
        
        # 加载配置
        self.config = load_config()
        
        # 初始化组件
        self.report_processor = ReportProcessor(self.config.get('data_processing', {}))
        self.detector = get_ensemble_detector()
        self.continuous_learning = get_continuous_learning_system()
        
        logger.info(f"JSON数据训练器初始化完成，数据目录: {self.json_files_dir}")
    
    def load_json_files(self) -> List[Dict[str, Any]]:
        """
        加载所有JSON文件
        
        Returns:
            List[Dict]: 研报数据列表
        """
        json_files = list(self.json_files_dir.glob("*.json"))
        logger.info(f"发现 {len(json_files)} 个JSON文件")
        
        all_data = []
        
        for json_file in json_files:
            try:
                logger.info(f"加载文件: {json_file.name}")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 添加文件源信息
                data['source_file'] = json_file.name
                data['load_timestamp'] = datetime.now().isoformat()
                
                all_data.append(data)
                logger.info(f"成功加载: {data.get('basic_info', {}).get('title', json_file.name)}")
                
            except Exception as e:
                logger.error(f"加载文件 {json_file.name} 失败: {e}")
                continue
        
        logger.info(f"总共加载了 {len(all_data)} 个研报数据")
        return all_data
    
    def prepare_training_data(self, data: List[Dict[str, Any]], 
                            train_ratio: float = 0.7, 
                            val_ratio: float = 0.2, 
                            test_ratio: float = 0.1) -> None:
        """
        准备训练数据
        
        Args:
            data: 原始数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("数据集比例之和必须等于1")
        
        # 随机打乱数据
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        
        # 计算分割点
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        
        # 分割数据
        self.training_data = [data[i] for i in indices[:train_end]]
        self.validation_data = [data[i] for i in indices[train_end:val_end]]
        self.test_data = [data[i] for i in indices[val_end:]]
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(self.training_data)} 个")
        logger.info(f"  验证集: {len(self.validation_data)} 个")
        logger.info(f"  测试集: {len(self.test_data)} 个")
    
    def extract_training_features(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        提取训练特征
        
        Args:
            data: 研报数据列表
            
        Returns:
            Dict: 特征数据
        """
        features = {
            'numerical_features': [],
            'text_features': [],
            'categorical_features': [],
            'meta_features': []
        }
        
        for report in data:
            try:
                # 提取数值特征
                numerical = self._extract_numerical_features(report)
                features['numerical_features'].append(numerical)
                
                # 提取文本特征
                text = self._extract_text_features(report)
                features['text_features'].append(text)
                
                # 提取分类特征
                categorical = self._extract_categorical_features(report)
                features['categorical_features'].append(categorical)
                
                # 提取元数据特征
                meta = self._extract_meta_features(report)
                features['meta_features'].append(meta)
                
            except Exception as e:
                logger.warning(f"提取特征失败: {e}")
                continue
        
        logger.info(f"特征提取完成，共 {len(features['numerical_features'])} 个样本")
        return features
    
    def _extract_numerical_features(self, report: Dict[str, Any]) -> Dict[str, float]:
        """提取数值特征"""
        numerical = {}
        
        try:
            # 财务指标
            financial_metrics = report.get('financial_metrics', {})
            for key, values in financial_metrics.items():
                if isinstance(values, list) and values:
                    # 取最新值
                    numerical[f'financial_{key}_latest'] = float(values[0]) if values[0] else 0.0
                    # 计算平均值
                    valid_values = [float(v) for v in values if v and str(v).replace('.', '').isdigit()]
                    if valid_values:
                        numerical[f'financial_{key}_avg'] = np.mean(valid_values)
                        numerical[f'financial_{key}_std'] = np.std(valid_values) if len(valid_values) > 1 else 0.0
            
            # 元信息中的数值特征
            meta_info = report.get('meta_info', {})
            for key in ['text_length', 'table_count', 'quality_score', 'confidence', 'processing_time']:
                if key in meta_info:
                    numerical[f'meta_{key}'] = float(meta_info[key])
            
            # 投资建议数值特征
            investment_advice = report.get('investment_advice', {})
            if 'current_price' in investment_advice and investment_advice['current_price']:
                numerical['current_price'] = float(investment_advice['current_price'])
            
        except Exception as e:
            logger.warning(f"提取数值特征时出错: {e}")
        
        return numerical
    
    def _extract_text_features(self, report: Dict[str, Any]) -> Dict[str, str]:
        """提取文本特征"""
        text_features = {}
        
        try:
            # 基本信息文本
            basic_info = report.get('basic_info', {})
            text_features['title'] = basic_info.get('title', '')
            text_features['stock_name'] = basic_info.get('stock_name', '')
            text_features['report_type'] = basic_info.get('report_type', '')
            
            # 核心内容文本
            core_content = report.get('core_content', {})
            text_features['investment_summary'] = core_content.get('investment_summary', '')
            text_features['risk_warning'] = core_content.get('risk_warning', '')
            
            # 主要逻辑（转换为文本）
            main_logic = core_content.get('main_logic', [])
            text_features['main_logic'] = ' '.join(main_logic) if isinstance(main_logic, list) else str(main_logic)
            
            # 深度分析文本
            deep_analysis = report.get('deep_analysis', {})
            if 'analysis_summary' in deep_analysis:
                summary = deep_analysis['analysis_summary']
                text_features['key_insights'] = ' '.join(summary.get('key_insights', []))
                text_features['analysis_strengths'] = ' '.join(summary.get('analysis_strengths', []))
                text_features['potential_weaknesses'] = ' '.join(summary.get('potential_weaknesses', []))
            
        except Exception as e:
            logger.warning(f"提取文本特征时出错: {e}")
        
        return text_features
    
    def _extract_categorical_features(self, report: Dict[str, Any]) -> Dict[str, str]:
        """提取分类特征"""
        categorical = {}
        
        try:
            # 基本信息分类
            basic_info = report.get('basic_info', {})
            categorical['institution'] = basic_info.get('institution', '')
            categorical['report_type'] = basic_info.get('report_type', '')
            
            # 投资建议分类
            investment_advice = report.get('investment_advice', {})
            categorical['rating'] = investment_advice.get('rating', '')
            categorical['rating_change'] = investment_advice.get('rating_change', '')
            
            # 元信息分类
            meta_info = report.get('meta_info', {})
            categorical['extraction_method'] = meta_info.get('extraction_method', '')
            categorical['success'] = str(meta_info.get('success', ''))
            
        except Exception as e:
            logger.warning(f"提取分类特征时出错: {e}")
        
        return categorical
    
    def _extract_meta_features(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """提取元数据特征"""
        meta = {}
        
        try:
            # 时间特征
            basic_info = report.get('basic_info', {})
            if 'publish_date' in basic_info:
                meta['publish_date'] = basic_info['publish_date']
            
            # 文件源特征
            meta['source_file'] = report.get('source_file', '')
            meta['load_timestamp'] = report.get('load_timestamp', '')
            
            # 质量特征
            meta_info = report.get('meta_info', {})
            meta['file_path'] = meta_info.get('file_path', '')
            meta['processor_version'] = meta_info.get('processor_version', '')
            
        except Exception as e:
            logger.warning(f"提取元数据特征时出错: {e}")
        
        return meta
    
    async def train_anomaly_detection_model(self) -> Dict[str, Any]:
        """
        训练异常检测模型
        
        Returns:
            Dict: 训练结果
        """
        try:
            logger.info("开始训练异常检测模型...")
            
            # 提取训练特征
            train_features = self.extract_training_features(self.training_data)
            val_features = self.extract_training_features(self.validation_data)
            
            # 创建训练配置
            training_config = TrainingConfig(
                model_type="ensemble_anomaly_detector",
                model_config={
                    "n_estimators": 10,
                    "contamination": 0.1,
                    "random_state": 42
                },
                epochs=5,
                batch_size=len(self.training_data),  # 小数据集使用全批次
                learning_rate=1e-4,
                validation_split=0.0,  # 我们已经手动分割了
                early_stopping=True,
                patience=2
            )
            
            # 初始化训练器
            trainer = Trainer(training_config)
            
            # 设置模型
            trainer.set_model(self.detector)
            
            # 模拟数据加载器（因为异常检测通常是无监督的）
            train_data = self._prepare_anomaly_detection_data(train_features)
            val_data = self._prepare_anomaly_detection_data(val_features)
            
            # 训练模型
            training_result = await self._train_ensemble_detector(train_data, val_data)
            
            # 保存训练结果
            self._save_training_results(training_result)
            
            logger.info("异常检测模型训练完成")
            return training_result
            
        except Exception as e:
            logger.error(f"训练异常检测模型失败: {e}")
            raise
    
    def _prepare_anomaly_detection_data(self, features: Dict[str, Any]) -> np.ndarray:
        """准备异常检测数据"""
        try:
            # 合并数值特征
            numerical_data = []
            for sample in features['numerical_features']:
                # 标准化特征向量
                feature_vector = []
                for key in sorted(sample.keys()):
                    value = sample[key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(value)
                    else:
                        feature_vector.append(0.0)
                
                if feature_vector:  # 确保有特征
                    numerical_data.append(feature_vector)
            
            if not numerical_data:
                logger.warning("没有有效的数值特征，使用默认特征")
                # 创建默认特征向量
                numerical_data = [[1.0, 0.0, 0.5] for _ in range(len(features['text_features']))]
            
            return np.array(numerical_data)
            
        except Exception as e:
            logger.error(f"准备异常检测数据失败: {e}")
            # 返回默认数据
            return np.array([[1.0, 0.0, 0.5] for _ in range(max(1, len(features.get('text_features', []))))])
    
    async def _train_ensemble_detector(self, train_data: np.ndarray, val_data: np.ndarray) -> Dict[str, Any]:
        """训练集成异常检测器"""
        try:
            start_time = datetime.now()
            
            # 训练检测器
            self.detector.fit(train_data)
            
            # 验证性能
            train_scores = self.detector.decision_function(train_data)
            val_scores = self.detector.decision_function(val_data)
            
            # 计算训练指标
            train_anomalies = self.detector.predict(train_data)
            val_anomalies = self.detector.predict(val_data)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'training_time': training_time,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'train_anomaly_ratio': np.mean(train_anomalies == -1),
                'val_anomaly_ratio': np.mean(val_anomalies == -1),
                'train_score_mean': np.mean(train_scores),
                'train_score_std': np.std(train_scores),
                'val_score_mean': np.mean(val_scores),
                'val_score_std': np.std(val_scores),
                'model_params': self.detector.get_params() if hasattr(self.detector, 'get_params') else {},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"训练完成，用时 {training_time:.2f} 秒")
            logger.info(f"训练集异常比例: {result['train_anomaly_ratio']:.2%}")
            logger.info(f"验证集异常比例: {result['val_anomaly_ratio']:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"训练集成检测器失败: {e}")
            raise
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """保存训练结果"""
        try:
            # 创建输出目录
            output_dir = Path(project_root) / "data" / "training_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存训练结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"training_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练结果已保存到: {results_file}")
            
            # 保存模型（如果支持）
            try:
                model_file = output_dir / f"trained_model_{timestamp}.pkl"
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(self.detector, f)
                logger.info(f"训练模型已保存到: {model_file}")
            except Exception as e:
                logger.warning(f"保存模型失败: {e}")
            
        except Exception as e:
            logger.error(f"保存训练结果失败: {e}")
    
    async def evaluate_model(self) -> Dict[str, Any]:
        """评估模型性能"""
        try:
            logger.info("开始评估模型性能...")
            
            if not self.test_data:
                logger.warning("没有测试数据，跳过模型评估")
                return {}
            
            # 提取测试特征
            test_features = self.extract_training_features(self.test_data)
            test_data = self._prepare_anomaly_detection_data(test_features)
            
            # 预测
            predictions = self.detector.predict(test_data)
            scores = self.detector.decision_function(test_data)
            
            # 计算评估指标
            evaluation_result = {
                'test_samples': len(test_data),
                'anomaly_count': np.sum(predictions == -1),
                'normal_count': np.sum(predictions == 1),
                'anomaly_ratio': np.mean(predictions == -1),
                'score_mean': np.mean(scores),
                'score_std': np.std(scores),
                'score_min': np.min(scores),
                'score_max': np.max(scores),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"模型评估完成:")
            logger.info(f"  测试样本: {evaluation_result['test_samples']}")
            logger.info(f"  异常样本: {evaluation_result['anomaly_count']}")
            logger.info(f"  正常样本: {evaluation_result['normal_count']}")
            logger.info(f"  异常比例: {evaluation_result['anomaly_ratio']:.2%}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}
    
    async def run_full_training_pipeline(self) -> Dict[str, Any]:
        """运行完整的训练流程"""
        try:
            logger.info("开始完整训练流程...")
            
            # 1. 加载数据
            logger.info("步骤 1: 加载JSON数据")
            all_data = self.load_json_files()
            
            if len(all_data) < 3:
                logger.warning(f"数据量过少({len(all_data)})，建议至少有5个样本进行训练")
            
            # 2. 准备训练数据
            logger.info("步骤 2: 准备训练数据")
            self.prepare_training_data(all_data)
            
            # 3. 训练模型
            logger.info("步骤 3: 训练异常检测模型")
            training_result = await self.train_anomaly_detection_model()
            
            # 4. 评估模型
            logger.info("步骤 4: 评估模型性能")
            evaluation_result = await self.evaluate_model()
            
            # 5. 汇总结果
            final_result = {
                'pipeline_status': 'completed',
                'data_summary': {
                    'total_reports': len(all_data),
                    'training_reports': len(self.training_data),
                    'validation_reports': len(self.validation_data),
                    'test_reports': len(self.test_data)
                },
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存最终结果
            self._save_training_results(final_result)
            
            logger.info("完整训练流程结束")
            return final_result
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            raise


async def main():
    """主函数"""
    try:
        logger.info("启动JSON数据训练脚本")
        
        # 初始化训练器
        trainer = JSONDataTrainer()
        
        # 运行完整训练流程
        result = await trainer.run_full_training_pipeline()
        
        print("\n" + "="*50)
        print("训练完成！")
        print("="*50)
        print(f"总报告数: {result['data_summary']['total_reports']}")
        print(f"训练报告数: {result['data_summary']['training_reports']}")
        print(f"验证报告数: {result['data_summary']['validation_reports']}")
        print(f"测试报告数: {result['data_summary']['test_reports']}")
        
        if 'training_result' in result and result['training_result']:
            tr = result['training_result']
            print(f"训练时间: {tr.get('training_time', 0):.2f} 秒")
            print(f"训练集异常比例: {tr.get('train_anomaly_ratio', 0):.2%}")
            print(f"验证集异常比例: {tr.get('val_anomaly_ratio', 0):.2%}")
        
        if 'evaluation_result' in result and result['evaluation_result']:
            er = result['evaluation_result']
            print(f"测试集异常比例: {er.get('anomaly_ratio', 0):.2%}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"训练脚本执行失败: {e}")
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
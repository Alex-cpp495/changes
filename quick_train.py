#!/usr/bin/env python3
"""
快速训练脚本 - 使用7个JSON文件进行模型训练
简化版本，专门处理用户提供的研报数据
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_json_data(data_dir="user_data/json_files"):
    """加载JSON数据"""
    logger = logging.getLogger(__name__)
    
    json_dir = project_root / data_dir
    if not json_dir.exists():
        logger.error(f"数据目录不存在: {json_dir}")
        return []
    
    json_files = list(json_dir.glob("*.json"))
    logger.info(f"发现 {len(json_files)} 个JSON文件")
    
    data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            data.append({
                'file_name': json_file.name,
                'title': report_data.get('basic_info', {}).get('title', 'Unknown'),
                'data': report_data
            })
            logger.info(f"[OK] 加载: {json_file.name}")
            
        except Exception as e:
            logger.error(f"[FAIL] 加载失败 {json_file.name}: {e}")
    
    logger.info(f"成功加载 {len(data)} 个研报文件")
    return data

def extract_simple_features(data_list):
    """提取简单特征用于训练"""
    logger = logging.getLogger(__name__)
    
    features = []
    labels = []
    
    for item in data_list:
        try:
            report = item['data']
            
            # 提取数值特征
            feature_vector = []
            
            # 1. 质量评分特征
            meta_info = report.get('meta_info', {})
            feature_vector.append(meta_info.get('quality_score', 5.0))
            feature_vector.append(meta_info.get('confidence', 0.5))
            feature_vector.append(meta_info.get('text_length', 1000) / 10000.0)  # 标准化
            feature_vector.append(meta_info.get('table_count', 0))
            
            # 2. 财务指标特征
            financial_metrics = report.get('financial_metrics', {})
            pe_values = financial_metrics.get('PE', [])
            if pe_values and pe_values[0]:
                try:
                    pe_val = float(pe_values[0])
                    feature_vector.append(min(pe_val / 100.0, 2.0))  # 标准化PE
                except:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
            
            # 3. 投资评级特征
            investment_advice = report.get('investment_advice', {})
            rating = investment_advice.get('rating', '')
            rating_score = {'买入': 1.0, '增持': 0.8, '持有': 0.5, '减持': 0.2, '卖出': 0.0}.get(rating, 0.5)
            feature_vector.append(rating_score)
            
            # 4. 文本长度特征
            core_content = report.get('core_content', {})
            summary_len = len(str(core_content.get('investment_summary', '')))
            feature_vector.append(min(summary_len / 1000.0, 2.0))  # 标准化
            
            # 5. 分析深度特征
            deep_analysis = report.get('deep_analysis', {})
            if deep_analysis:
                reasoning_analysis = deep_analysis.get('reasoning_analysis', {})
                logic_strength = reasoning_analysis.get('logic_strength', 0.5)
                feature_vector.append(logic_strength)
            else:
                feature_vector.append(0.0)
            
            features.append(feature_vector)
            
            # 简单标签：根据质量评分判断是否异常
            quality_score = meta_info.get('quality_score', 5.0)
            is_normal = 1 if quality_score >= 7.0 else 0
            labels.append(is_normal)
            
        except Exception as e:
            logger.warning(f"特征提取失败 {item['file_name']}: {e}")
            continue
    
    logger.info(f"提取到 {len(features)} 个特征向量，每个向量 {len(features[0]) if features else 0} 维")
    return np.array(features), np.array(labels)

def simple_anomaly_detection(X, contamination=0.2):
    """简单的异常检测"""
    logger = logging.getLogger(__name__)
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练隔离森林
        detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        detector.fit(X_scaled)
        
        # 预测
        predictions = detector.predict(X_scaled)
        scores = detector.decision_function(X_scaled)
        
        # 统计结果
        normal_count = np.sum(predictions == 1)
        anomaly_count = np.sum(predictions == -1)
        
        logger.info(f"异常检测完成:")
        logger.info(f"  正常样本: {normal_count}")
        logger.info(f"  异常样本: {anomaly_count}")
        logger.info(f"  异常比例: {anomaly_count/(normal_count+anomaly_count):.2%}")
        
        return {
            'model': detector,
            'scaler': scaler,
            'predictions': predictions,
            'scores': scores,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count
        }
        
    except ImportError:
        logger.error("scikit-learn 未安装，请运行: pip install scikit-learn")
        return None
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        return None

def save_results(results, features, data_list):
    """保存训练结果"""
    logger = logging.getLogger(__name__)
    
    try:
        # 创建结果目录
        results_dir = project_root / "data" / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_results = []
        for i, item in enumerate(data_list):
            if i < len(results['predictions']):
                detailed_results.append({
                    'file_name': item['file_name'],
                    'title': item['title'],
                    'prediction': 'normal' if results['predictions'][i] == 1 else 'anomaly',
                    'score': float(results['scores'][i]),
                    'features': features[i].tolist()
                })
        
        # 保存JSON结果
        result_file = results_dir / f"training_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'summary': {
                    'total_samples': len(data_list),
                    'normal_count': int(results['normal_count']),
                    'anomaly_count': int(results['anomaly_count']),
                    'anomaly_ratio': float(results['anomaly_count'] / len(data_list))
                },
                'detailed_results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {result_file}")
        
        # 保存模型
        import pickle
        model_file = results_dir / f"model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': results['model'],
                'scaler': results['scaler'],
                'timestamp': timestamp
            }, f)
        
        logger.info(f"模型已保存到: {model_file}")
        
        return result_file, model_file
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        return None, None

def main():
    """主函数"""
    logger = setup_logging()
    
    print("="*60)
    print("          东财异常检测模型训练")
    print("="*60)
    
    try:
        # 1. 加载数据
        print("\n1. 加载JSON数据...")
        data_list = load_json_data()
        
        if len(data_list) == 0:
            print("[ERROR] 没有找到有效的JSON文件，请检查数据目录")
            return 1
        
        # 2. 提取特征
        print("\n2. 提取特征...")
        features, labels = extract_simple_features(data_list)
        
        if len(features) == 0:
            print("[ERROR] 特征提取失败")
            return 1
        
        print(f"[OK] 提取到 {len(features)} 个样本，每个样本 {features.shape[1]} 个特征")
        
        # 3. 训练模型
        print("\n3. 训练异常检测模型...")
        results = simple_anomaly_detection(features)
        
        if results is None:
            print("[ERROR] 模型训练失败")
            return 1
        
        # 4. 保存结果
        print("\n4. 保存训练结果...")
        result_file, model_file = save_results(results, features, data_list)
        
        # 5. 显示结果
        print("\n" + "="*60)
        print("           训练完成！")
        print("="*60)
        print(f"[总样本数] {len(data_list)}")
        print(f"[正常样本] {results['normal_count']}")
        print(f"[异常样本] {results['anomaly_count']}")
        print(f"[异常比例] {results['anomaly_count']/len(data_list):.1%}")
        
        if result_file:
            print(f"\n[结果文件] {result_file.name}")
        if model_file:
            print(f"[模型文件] {model_file.name}")
        
        print("\n详细异常检测结果:")
        print("-" * 40)
        for i, item in enumerate(data_list):
            if i < len(results['predictions']):
                status = "[正常]" if results['predictions'][i] == 1 else "[异常]"
                score = results['scores'][i]
                print(f"{status} | {item['file_name']:<15} | 评分: {score:>6.3f}")
        
        print("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        print(f"[ERROR] {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
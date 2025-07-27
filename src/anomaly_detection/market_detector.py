"""
市场异常检测器
检测与市场表现相关的异常，包括异常收益率、成交量异常、预测准确性异常等
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
import logging
import re
import math

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.text_utils import get_text_processor
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class MarketAnomalyDetector:
    """
    市场异常检测器
    
    检测与市场表现相关的异常：
    1. 异常收益率 - 实际收益与预期收益的偏离
    2. 成交量异常 - 异常的交易量放大或缩减
    3. 预测准确性异常 - 预测准确率的突然变化
    4. 价格趋势异常 - 价格走势与预测的背离
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 配置参数
        text_processor: 文本处理器
        market_data: 市场数据存储
        prediction_history: 预测历史记录
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化市场异常检测器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.text_processor = get_text_processor()
        self.file_manager = get_file_manager()
        
        # 市场数据存储
        self.market_data = {
            'stock_prices': defaultdict(list),  # 股票价格历史
            'volumes': defaultdict(list),       # 成交量历史
            'returns': defaultdict(list),       # 收益率历史
            'volatility': defaultdict(list),    # 波动率历史
            'last_update': None
        }
        
        # 预测历史记录
        self.prediction_history = {
            'predictions': [],           # 预测记录列表
            'accuracy_scores': deque(maxlen=100),  # 准确率历史
            'prediction_errors': defaultdict(list),  # 预测误差
            'prediction_count': 0,
            'last_update': None
        }
        
        # 统计模型
        self.is_trained = False
        self.market_baseline = {}
        self.prediction_baseline = {}
        
        # 加载历史数据
        self._load_historical_data()
        
        logger.info("市场异常检测器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('market_anomaly', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'return_anomaly': {
                'return_threshold': 0.05,      # 异常收益率阈值 (5%)
                'volatility_multiplier': 3.0,  # 波动率倍数阈值
                'lookback_days': 30            # 历史数据回看天数
            },
            'volume_anomaly': {
                'volume_multiplier_threshold': 3.0,  # 成交量倍数阈值
                'volume_change_threshold': 2.0,      # 成交量变化阈值
                'min_volume_data_points': 10         # 最少成交量数据点
            },
            'prediction_accuracy': {
                'accuracy_drop_threshold': 0.3,    # 准确率下降阈值
                'min_predictions_for_baseline': 20, # 基线计算最少预测数
                'recent_predictions_window': 10     # 最近预测窗口
            },
            'price_trend_anomaly': {
                'trend_deviation_threshold': 0.2,   # 趋势偏离阈值
                'price_shock_threshold': 0.1,       # 价格冲击阈值
                'trend_analysis_days': 14            # 趋势分析天数
            },
            'min_historical_data_points': 30,  # 最少历史数据点
            'anomaly_score_weights': {
                'return': 0.3,
                'volume': 0.25,
                'prediction_accuracy': 0.25,
                'price_trend': 0.2
            }
        }
    
    def _load_historical_data(self):
        """加载历史市场数据"""
        try:
            data_path = Path("data/models/market_history.pkl")
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.market_data.update(data.get('market', {}))
                self.prediction_history.update(data.get('predictions', {}))
                
                # 重建基线模型
                total_data_points = sum(len(prices) for prices in self.market_data['stock_prices'].values())
                if total_data_points >= self.config['min_historical_data_points']:
                    self._build_baseline_models()
                    logger.info(f"已加载市场历史数据，总数据点: {total_data_points}")
                
        except Exception as e:
            logger.warning(f"历史数据加载失败: {e}")
    
    def _save_historical_data(self):
        """保存历史市场数据"""
        try:
            data_path = Path("data/models/market_history.pkl")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换deque为list以便序列化
            prediction_data = dict(self.prediction_history)
            prediction_data['accuracy_scores'] = list(prediction_data['accuracy_scores'])
            
            data = {
                'market': self.market_data,
                'predictions': prediction_data
            }
            
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug("历史市场数据已保存")
            
        except Exception as e:
            logger.error(f"历史数据保存失败: {e}")
    
    def add_market_data(self, stock_code: str, date: datetime, 
                       open_price: float, close_price: float, 
                       high_price: float, low_price: float, 
                       volume: int, turnover: float = None):
        """
        添加市场数据
        
        Args:
            stock_code: 股票代码
            date: 日期
            open_price: 开盘价
            close_price: 收盘价
            high_price: 最高价
            low_price: 最低价
            volume: 成交量
            turnover: 成交额（可选）
        """
        try:
            # 计算收益率
            returns = self.market_data['returns'][stock_code]
            if len(self.market_data['stock_prices'][stock_code]) > 0:
                prev_close = self.market_data['stock_prices'][stock_code][-1]['close']
                daily_return = (close_price - prev_close) / prev_close
            else:
                daily_return = 0.0
            
            # 存储价格数据
            price_data = {
                'date': date,
                'open': open_price,
                'close': close_price,
                'high': high_price,
                'low': low_price,
                'volume': volume,
                'turnover': turnover,
                'return': daily_return
            }
            
            self.market_data['stock_prices'][stock_code].append(price_data)
            self.market_data['volumes'][stock_code].append(volume)
            self.market_data['returns'][stock_code].append(daily_return)
            
            # 计算波动率（滚动20日标准差）
            if len(returns) >= 20:
                recent_returns = returns[-20:]
                volatility = np.std(recent_returns) * np.sqrt(252)  # 年化波动率
                self.market_data['volatility'][stock_code].append(volatility)
            
            # 限制历史数据长度（保持最近1000个数据点）
            for key in ['stock_prices', 'volumes', 'returns', 'volatility']:
                if len(self.market_data[key][stock_code]) > 1000:
                    self.market_data[key][stock_code] = self.market_data[key][stock_code][-1000:]
            
            self.market_data['last_update'] = datetime.now()
            
            # 保存数据
            self._save_historical_data()
            
            logger.debug(f"添加市场数据: {stock_code} - {date.date()}")
            
        except Exception as e:
            logger.error(f"添加市场数据失败: {e}")
            raise
    
    def add_prediction_record(self, prediction_data: Dict[str, Any]):
        """
        添加预测记录
        
        Args:
            prediction_data: 预测数据，包含股票代码、预测值、实际值、日期等
        """
        try:
            # 计算预测准确性
            predicted_value = prediction_data.get('predicted_value')
            actual_value = prediction_data.get('actual_value')
            stock_code = prediction_data.get('stock_code')
            prediction_date = prediction_data.get('date')
            
            if predicted_value is not None and actual_value is not None:
                # 计算预测误差
                error = abs(predicted_value - actual_value) / max(abs(actual_value), 1e-6)
                accuracy = max(0, 1 - error)
                
                # 存储预测记录
                record = {
                    'stock_code': stock_code,
                    'date': prediction_date,
                    'predicted_value': predicted_value,
                    'actual_value': actual_value,
                    'error': error,
                    'accuracy': accuracy,
                    'prediction_type': prediction_data.get('type', 'price')
                }
                
                self.prediction_history['predictions'].append(record)
                self.prediction_history['accuracy_scores'].append(accuracy)
                self.prediction_history['prediction_errors'][stock_code].append(error)
                self.prediction_history['prediction_count'] += 1
                
                # 限制预测历史长度
                if len(self.prediction_history['predictions']) > 1000:
                    self.prediction_history['predictions'] = self.prediction_history['predictions'][-1000:]
                
                for stock in self.prediction_history['prediction_errors']:
                    if len(self.prediction_history['prediction_errors'][stock]) > 100:
                        self.prediction_history['prediction_errors'][stock] = \
                            self.prediction_history['prediction_errors'][stock][-100:]
                
                self.prediction_history['last_update'] = datetime.now()
                
                # 重建基线模型
                if self.prediction_history['prediction_count'] >= self.config['prediction_accuracy']['min_predictions_for_baseline']:
                    self._build_baseline_models()
                
                # 保存数据
                self._save_historical_data()
                
                logger.debug(f"添加预测记录: {stock_code} - 准确率: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"添加预测记录失败: {e}")
            raise
    
    def _build_baseline_models(self):
        """构建基线模型"""
        try:
            # 市场数据基线
            self.market_baseline = {}
            
            for stock_code in self.market_data['stock_prices']:
                if len(self.market_data['returns'][stock_code]) >= 30:
                    returns = np.array(self.market_data['returns'][stock_code])
                    volumes = np.array(self.market_data['volumes'][stock_code])
                    
                    self.market_baseline[stock_code] = {
                        'return_mean': np.mean(returns),
                        'return_std': np.std(returns),
                        'return_percentiles': np.percentile(returns, [5, 25, 50, 75, 95]),
                        'volume_mean': np.mean(volumes),
                        'volume_std': np.std(volumes),
                        'volume_percentiles': np.percentile(volumes, [5, 25, 50, 75, 95])
                    }
            
            # 预测准确性基线
            if len(self.prediction_history['accuracy_scores']) >= 20:
                accuracies = list(self.prediction_history['accuracy_scores'])
                self.prediction_baseline = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'accuracy_trend': self._calculate_accuracy_trend(accuracies),
                    'recent_accuracy': np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)
                }
            
            self.is_trained = True
            logger.info("市场基线模型构建完成")
            
        except Exception as e:
            logger.error(f"基线模型构建失败: {e}")
            raise
    
    def _calculate_accuracy_trend(self, accuracies: List[float]) -> float:
        """计算准确率趋势"""
        if len(accuracies) < 10:
            return 0.0
        
        # 使用线性回归计算趋势
        x = np.arange(len(accuracies))
        y = np.array(accuracies)
        
        # 简单线性回归
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def detect_market_anomalies(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测市场异常
        
        Args:
            report_data: 报告数据，包含股票代码、预测、实际表现等
            
        Returns:
            Dict[str, Any]: 市场异常检测结果
            
        Raises:
            RuntimeError: 检测器未训练
        """
        if not self.is_trained:
            total_data_points = sum(len(prices) for prices in self.market_data['stock_prices'].values())
            if total_data_points < self.config['min_historical_data_points']:
                raise RuntimeError(f"历史数据不足，需要至少 {self.config['min_historical_data_points']} 个数据点")
            else:
                self._build_baseline_models()
        
        try:
            results = {
                'return_anomaly': self._detect_return_anomaly(report_data),
                'volume_anomaly': self._detect_volume_anomaly(report_data),
                'prediction_accuracy_anomaly': self._detect_prediction_accuracy_anomaly(report_data),
                'price_trend_anomaly': self._detect_price_trend_anomaly(report_data),
                'overall_score': 0.0,
                'anomaly_level': 'NORMAL',
                'details': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 计算综合异常分数
            weights = self.config['anomaly_score_weights']
            anomaly_scores = [
                results['return_anomaly']['score'] * weights['return'],
                results['volume_anomaly']['score'] * weights['volume'],
                results['prediction_accuracy_anomaly']['score'] * weights['prediction_accuracy'],
                results['price_trend_anomaly']['score'] * weights['price_trend']
            ]
            
            results['overall_score'] = sum(anomaly_scores)
            
            # 确定异常等级
            results['anomaly_level'] = self._determine_anomaly_level(results['overall_score'])
            
            # 详细信息
            results['details'] = {
                'tracked_stocks': len(self.market_data['stock_prices']),
                'prediction_count': self.prediction_history['prediction_count'],
                'last_market_update': self.market_data['last_update'].isoformat() if self.market_data['last_update'] else None,
                'data_coverage_days': self._calculate_data_coverage()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"市场异常检测失败: {e}")
            raise
    
    def _detect_return_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测收益率异常"""
        stocks = report_data.get('stocks', [])
        analysis_date = report_data.get('analysis_date')
        
        if not stocks:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无股票信息"
            }
        
        anomaly_scores = []
        anomalous_stocks = []
        
        for stock_code in stocks:
            if stock_code not in self.market_baseline:
                continue
            
            baseline = self.market_baseline[stock_code]
            recent_returns = self.market_data['returns'][stock_code][-30:]  # 最近30天
            
            if len(recent_returns) < 10:
                continue
            
            # 计算最近收益率统计
            recent_mean = np.mean(recent_returns)
            recent_volatility = np.std(recent_returns)
            
            # 检查异常收益率
            return_threshold = self.config['return_anomaly']['return_threshold']
            volatility_multiplier = self.config['return_anomaly']['volatility_multiplier']
            
            # 收益率异常判断
            expected_range = volatility_multiplier * baseline['return_std']
            return_deviation = abs(recent_mean - baseline['return_mean'])
            
            if return_deviation > expected_range or abs(recent_mean) > return_threshold:
                # 计算异常分数
                volatility_score = return_deviation / max(expected_range, 1e-6)
                absolute_score = abs(recent_mean) / return_threshold
                stock_score = min(max(volatility_score, absolute_score), 1.0)
                
                anomaly_scores.append(stock_score)
                anomalous_stocks.append({
                    'stock_code': stock_code,
                    'recent_return': recent_mean,
                    'expected_return': baseline['return_mean'],
                    'deviation': return_deviation,
                    'score': stock_score
                })
        
        # 综合异常分数
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = overall_score > 0.5
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'anomalous_stocks_count': len(anomalous_stocks),
            'anomalous_stocks': anomalous_stocks[:5],  # 只显示前5个
            'description': f"发现{len(anomalous_stocks)}只股票收益率异常，最高分数{overall_score:.2f}"
        }
    
    def _detect_volume_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测成交量异常"""
        stocks = report_data.get('stocks', [])
        
        if not stocks:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无股票信息"
            }
        
        anomaly_scores = []
        anomalous_stocks = []
        
        volume_multiplier_threshold = self.config['volume_anomaly']['volume_multiplier_threshold']
        min_data_points = self.config['volume_anomaly']['min_volume_data_points']
        
        for stock_code in stocks:
            if stock_code not in self.market_baseline:
                continue
            
            volumes = self.market_data['volumes'][stock_code]
            if len(volumes) < min_data_points:
                continue
            
            baseline = self.market_baseline[stock_code]
            recent_volumes = volumes[-10:]  # 最近10天
            
            # 计算成交量统计
            recent_avg_volume = np.mean(recent_volumes)
            baseline_avg_volume = baseline['volume_mean']
            
            if baseline_avg_volume > 0:
                volume_ratio = recent_avg_volume / baseline_avg_volume
                
                # 检查成交量异常
                if volume_ratio > volume_multiplier_threshold or volume_ratio < (1.0 / volume_multiplier_threshold):
                    # 计算异常分数
                    if volume_ratio > volume_multiplier_threshold:
                        stock_score = min((volume_ratio - volume_multiplier_threshold) / volume_multiplier_threshold, 1.0)
                    else:
                        stock_score = min((1.0 / volume_multiplier_threshold - volume_ratio) / (1.0 / volume_multiplier_threshold), 1.0)
                    
                    anomaly_scores.append(stock_score)
                    anomalous_stocks.append({
                        'stock_code': stock_code,
                        'recent_volume': recent_avg_volume,
                        'baseline_volume': baseline_avg_volume,
                        'volume_ratio': volume_ratio,
                        'score': stock_score
                    })
        
        # 综合异常分数
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = overall_score > 0.5
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'anomalous_stocks_count': len(anomalous_stocks),
            'anomalous_stocks': anomalous_stocks[:5],
            'description': f"发现{len(anomalous_stocks)}只股票成交量异常，最高分数{overall_score:.2f}"
        }
    
    def _detect_prediction_accuracy_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测预测准确性异常"""
        if not self.prediction_baseline or len(self.prediction_history['accuracy_scores']) < 10:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "预测数据不足"
            }
        
        window_size = self.config['prediction_accuracy']['recent_predictions_window']
        accuracy_drop_threshold = self.config['prediction_accuracy']['accuracy_drop_threshold']
        
        # 计算最近准确率
        recent_accuracies = list(self.prediction_history['accuracy_scores'])[-window_size:]
        recent_avg_accuracy = np.mean(recent_accuracies)
        
        # 与基线比较
        baseline_accuracy = self.prediction_baseline['accuracy_mean']
        accuracy_drop = baseline_accuracy - recent_avg_accuracy
        
        # 检查准确率下降异常
        is_anomaly = accuracy_drop > accuracy_drop_threshold
        
        # 计算异常分数
        if is_anomaly:
            anomaly_score = min(accuracy_drop / accuracy_drop_threshold, 1.0)
        else:
            anomaly_score = max(0, accuracy_drop / accuracy_drop_threshold)
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'recent_accuracy': recent_avg_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_drop': accuracy_drop,
            'accuracy_trend': self.prediction_baseline.get('accuracy_trend', 0),
            'description': f"预测准确率{recent_avg_accuracy:.2%}，下降{accuracy_drop:.2%}"
        }
    
    def _detect_price_trend_anomaly(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """检测价格趋势异常"""
        stocks = report_data.get('stocks', [])
        predicted_direction = report_data.get('predicted_direction')  # 'up', 'down', 'neutral'
        
        if not stocks:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'description': "无股票信息"
            }
        
        trend_analysis_days = self.config['price_trend_anomaly']['trend_analysis_days']
        deviation_threshold = self.config['price_trend_anomaly']['trend_deviation_threshold']
        
        anomaly_scores = []
        trend_deviations = []
        
        for stock_code in stocks:
            if stock_code not in self.market_data['stock_prices']:
                continue
            
            prices = self.market_data['stock_prices'][stock_code]
            if len(prices) < trend_analysis_days:
                continue
            
            # 分析最近价格趋势
            recent_prices = prices[-trend_analysis_days:]
            price_changes = [p['close'] for p in recent_prices]
            
            # 计算趋势
            actual_trend = self._calculate_price_trend(price_changes)
            
            # 与预测方向比较
            if predicted_direction:
                trend_deviation = self._calculate_trend_deviation(actual_trend, predicted_direction)
                
                if abs(trend_deviation) > deviation_threshold:
                    stock_score = min(abs(trend_deviation) / deviation_threshold, 1.0)
                    anomaly_scores.append(stock_score)
                    
                    trend_deviations.append({
                        'stock_code': stock_code,
                        'predicted_direction': predicted_direction,
                        'actual_trend': actual_trend,
                        'deviation': trend_deviation,
                        'score': stock_score
                    })
        
        # 综合异常分数
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = overall_score > 0.5
        
        return {
            'is_anomaly': is_anomaly,
            'score': overall_score,
            'trend_deviations_count': len(trend_deviations),
            'trend_deviations': trend_deviations[:5],
            'description': f"发现{len(trend_deviations)}只股票趋势偏离预测，最高分数{overall_score:.2f}"
        }
    
    def _calculate_price_trend(self, prices: List[float]) -> float:
        """计算价格趋势（斜率）"""
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # 简单线性回归计算趋势
        n = len(x)
        if n == 0:
            return 0.0
        
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        return slope / np.mean(y) if np.mean(y) != 0 else 0.0  # 归一化斜率
    
    def _calculate_trend_deviation(self, actual_trend: float, predicted_direction: str) -> float:
        """计算趋势偏离度"""
        # 将预测方向转换为数值
        direction_mapping = {'up': 1.0, 'down': -1.0, 'neutral': 0.0}
        predicted_trend = direction_mapping.get(predicted_direction, 0.0)
        
        # 计算偏离度
        if predicted_trend == 0:  # 预测中性
            return abs(actual_trend)
        elif predicted_trend > 0:  # 预测上涨
            return -actual_trend if actual_trend < 0 else 0  # 实际下跌为偏离
        else:  # 预测下跌
            return actual_trend if actual_trend > 0 else 0   # 实际上涨为偏离
    
    def _calculate_data_coverage(self) -> int:
        """计算数据覆盖天数"""
        if not self.market_data['stock_prices']:
            return 0
        
        all_dates = []
        for stock_prices in self.market_data['stock_prices'].values():
            if stock_prices:
                all_dates.extend([p['date'] for p in stock_prices])
        
        if not all_dates:
            return 0
        
        return (max(all_dates) - min(all_dates)).days
    
    def _determine_anomaly_level(self, score: float) -> str:
        """
        确定异常等级
        
        Args:
            score: 异常分数 (0-1)
            
        Returns:
            str: 异常等级
        """
        if score >= 0.8:
            return 'CRITICAL'
        elif score >= 0.6:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        elif score >= 0.2:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        获取市场数据摘要
        
        Returns:
            Dict[str, Any]: 市场统计摘要
        """
        total_data_points = sum(len(prices) for prices in self.market_data['stock_prices'].values())
        
        return {
            'tracked_stocks': len(self.market_data['stock_prices']),
            'total_data_points': total_data_points,
            'prediction_count': self.prediction_history['prediction_count'],
            'data_coverage_days': self._calculate_data_coverage(),
            'market_baseline': {stock: baseline for stock, baseline in list(self.market_baseline.items())[:5]},
            'prediction_baseline': self.prediction_baseline,
            'recent_accuracy': (
                np.mean(list(self.prediction_history['accuracy_scores'])[-10:])
                if len(self.prediction_history['accuracy_scores']) >= 10 else None
            ),
            'is_trained': self.is_trained,
            'last_update': self.market_data['last_update'].isoformat() if self.market_data['last_update'] else None
        }


# 全局检测器实例
_global_market_detector = None


def get_market_detector() -> MarketAnomalyDetector:
    """
    获取全局市场异常检测器实例
    
    Returns:
        MarketAnomalyDetector: 检测器实例
    """
    global _global_market_detector
    
    if _global_market_detector is None:
        _global_market_detector = MarketAnomalyDetector()
    
    return _global_market_detector


if __name__ == "__main__":
    # 使用示例
    detector = MarketAnomalyDetector()
    
    # 模拟市场数据
    base_date = datetime.now() - timedelta(days=60)
    stock_codes = ["000001", "000002", "600000"]
    
    for i in range(60):
        for stock_code in stock_codes:
            # 模拟价格数据
            base_price = 10.0 + np.random.random() * 5
            price_change = np.random.normal(0, 0.02)  # 2%波动
            
            open_price = base_price
            close_price = base_price * (1 + price_change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.normal(1000000, 200000))
            
            detector.add_market_data(
                stock_code=stock_code,
                date=base_date + timedelta(days=i),
                open_price=open_price,
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                volume=max(volume, 100000)
            )
    
    # 模拟预测数据
    for i in range(30):
        detector.add_prediction_record({
            'stock_code': np.random.choice(stock_codes),
            'date': base_date + timedelta(days=30 + i),
            'predicted_value': 10.0 + np.random.random() * 2,
            'actual_value': 10.0 + np.random.random() * 2,
            'type': 'price'
        })
    
    # 检测市场异常
    test_report = {
        'stocks': ["000001", "000002"],
        'analysis_date': datetime.now(),
        'predicted_direction': 'up'
    }
    
    result = detector.detect_market_anomalies(test_report)
    
    print("市场异常检测结果:")
    print(f"整体异常分数: {result['overall_score']:.3f}")
    print(f"异常等级: {result['anomaly_level']}")
    print(f"收益率异常: {result['return_anomaly']['is_anomaly']}")
    print(f"成交量异常: {result['volume_anomaly']['is_anomaly']}")
    print(f"预测准确性异常: {result['prediction_accuracy_anomaly']['is_anomaly']}")
    print(f"价格趋势异常: {result['price_trend_anomaly']['is_anomaly']}") 
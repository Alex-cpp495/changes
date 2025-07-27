"""
市场数据收集器
收集和整理股票市场数据，包括价格、成交量、财务指标等，用于异常检测分析
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path

from ..utils.config_loader import load_config
from ..utils.logger import get_logger
from ..utils.file_utils import get_file_manager

logger = get_logger(__name__)


class MarketDataCollector:
    """
    市场数据收集器
    
    提供全面的市场数据收集功能：
    1. 股价数据 - 实时价格、历史价格、技术指标
    2. 交易数据 - 成交量、换手率、资金流向
    3. 财务数据 - 基本面指标、财务报表数据
    4. 市场数据 - 指数数据、行业数据、市场情绪
    5. 新闻数据 - 公告、新闻、研报数据
    6. 数据缓存 - 本地缓存、数据更新策略
    
    Args:
        config_path: 配置文件路径
        
    Attributes:
        config: 数据收集配置
        file_manager: 文件管理器
        data_cache: 数据缓存
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化市场数据收集器"""
        self.config_path = config_path or "configs/anomaly_thresholds.yaml"
        self.config = self._load_config()
        
        self.file_manager = get_file_manager()
        
        # 数据缓存
        self.data_cache = {
            'stock_prices': {},
            'financial_data': {},
            'market_indices': {},
            'news_data': {},
            'last_update': {}
        }
        
        # 数据源配置
        self.data_sources = self._setup_data_sources()
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 统计信息
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'data_points_collected': 0
        }
        
        logger.info("市场数据收集器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config = load_config(self.config_path)
            return config.get('market_data_collection', {})
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_sources': {
                'primary': 'akshare',  # 主要数据源
                'fallback': ['tushare', 'eastmoney'],  # 备用数据源
                'enable_cache': True,
                'cache_duration_hours': 1,
                'max_retries': 3,
                'retry_delay': 1.0
            },
            'collection_scope': {
                'collect_stock_prices': True,
                'collect_financial_data': True,
                'collect_market_indices': True,
                'collect_news_data': False,  # 需要额外配置
                'max_stocks_per_batch': 50,
                'default_period_days': 30
            },
            'data_quality': {
                'validate_data': True,
                'fill_missing_values': True,
                'remove_outliers': False,
                'outlier_threshold': 3.0
            },
            'performance': {
                'use_async': True,
                'max_concurrent_requests': 10,
                'request_timeout': 30,
                'rate_limit_delay': 0.1
            },
            'storage': {
                'save_raw_data': True,
                'save_processed_data': True,
                'data_directory': 'data/market_data',
                'compress_data': False
            }
        }
    
    def _setup_data_sources(self) -> Dict[str, Any]:
        """设置数据源配置"""
        sources = {
            'akshare': {
                'name': 'akshare',
                'available': self._check_akshare_availability(),
                'functions': {
                    'stock_price': 'stock_zh_a_hist',
                    'stock_info': 'stock_individual_info_em',
                    'financial_data': 'stock_financial_em'
                }
            },
            'tushare': {
                'name': 'tushare',
                'available': False,  # 需要token配置
                'functions': {}
            },
            'eastmoney': {
                'name': 'eastmoney',
                'available': True,  # API方式
                'base_url': 'http://push2.eastmoney.com/api/qt',
                'functions': {}
            }
        }
        
        return sources
    
    def _check_akshare_availability(self) -> bool:
        """检查akshare可用性"""
        try:
            import akshare as ak
            return True
        except ImportError:
            logger.warning("akshare未安装，将使用备用数据源")
            return False
    
    def collect_stock_data(self, stock_codes: List[str], 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        收集股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            Dict[str, Any]: 收集的股票数据
        """
        if not stock_codes:
            return {'error': '股票代码列表为空'}
        
        # 设置默认日期范围
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.config['collection_scope']['default_period_days'])).strftime('%Y-%m-%d')
        
        results = {
            'stock_prices': {},
            'stock_info': {},
            'collection_metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'requested_stocks': stock_codes,
                'successful_stocks': [],
                'failed_stocks': [],
                'collection_time': datetime.now().isoformat()
            }
        }
        
        logger.info(f"开始收集{len(stock_codes)}只股票的数据")
        
        for stock_code in stock_codes:
            try:
                # 规范化股票代码
                normalized_code = self._normalize_stock_code(stock_code)
                
                # 检查缓存
                cache_key = f"{normalized_code}_{start_date}_{end_date}"
                if self._check_cache(cache_key):
                    cached_data = self._get_from_cache(cache_key)
                    results['stock_prices'][stock_code] = cached_data
                    results['collection_metadata']['successful_stocks'].append(stock_code)
                    self.collection_stats['cache_hits'] += 1
                    continue
                
                # 收集价格数据
                price_data = self._collect_price_data(normalized_code, start_date, end_date)
                if price_data is not None:
                    results['stock_prices'][stock_code] = price_data
                    results['collection_metadata']['successful_stocks'].append(stock_code)
                    
                    # 缓存数据
                    self._save_to_cache(cache_key, price_data)
                    
                    self.collection_stats['data_points_collected'] += len(price_data) if isinstance(price_data, list) else 1
                else:
                    results['collection_metadata']['failed_stocks'].append(stock_code)
                
                # 收集基本信息
                info_data = self._collect_stock_info(normalized_code)
                if info_data:
                    results['stock_info'][stock_code] = info_data
                
                # 速率限制
                time.sleep(self.config['performance']['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"收集股票{stock_code}数据失败: {e}")
                results['collection_metadata']['failed_stocks'].append(stock_code)
        
        logger.info(f"数据收集完成: 成功{len(results['collection_metadata']['successful_stocks'])}只，失败{len(results['collection_metadata']['failed_stocks'])}只")
        
        return results
    
    def _normalize_stock_code(self, stock_code: str) -> str:
        """规范化股票代码"""
        # 移除空格
        code = stock_code.strip()
        
        # 如果没有交易所后缀，根据代码添加
        if '.' not in code and len(code) == 6:
            if code.startswith(('00', '30')):
                code += '.SZ'
            elif code.startswith('60'):
                code += '.SH'
            elif code.startswith(('43', '83', '87')):
                code += '.BJ'
        
        return code
    
    def _collect_price_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """收集股价数据"""
        self.collection_stats['total_requests'] += 1
        
        # 尝试使用akshare
        if self.data_sources['akshare']['available']:
            try:
                data = self._collect_with_akshare(stock_code, start_date, end_date)
                if data:
                    self.collection_stats['successful_requests'] += 1
                    return data
            except Exception as e:
                logger.debug(f"akshare收集失败: {e}")
        
        # 尝试使用备用数据源
        try:
            data = self._collect_with_eastmoney(stock_code, start_date, end_date)
            if data:
                self.collection_stats['successful_requests'] += 1
                return data
        except Exception as e:
            logger.debug(f"eastmoney收集失败: {e}")
        
        self.collection_stats['failed_requests'] += 1
        return None
    
    def _collect_with_akshare(self, stock_code: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """使用akshare收集数据"""
        try:
            import akshare as ak
            
            # 转换股票代码格式
            symbol = stock_code.replace('.', '')
            
            # 获取历史数据
            df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date.replace('-', ''), 
                                   end_date=end_date.replace('-', ''), adjust="qfq")
            
            if df is not None and not df.empty:
                # 转换为标准格式
                data = []
                for _, row in df.iterrows():
                    data.append({
                        'date': row['日期'].strftime('%Y-%m-%d') if hasattr(row['日期'], 'strftime') else str(row['日期']),
                        'open': float(row['开盘']),
                        'high': float(row['最高']),
                        'low': float(row['最低']),
                        'close': float(row['收盘']),
                        'volume': int(row['成交量']),
                        'turnover': float(row['成交额']),
                        'change_pct': float(row.get('涨跌幅', 0)),
                        'source': 'akshare'
                    })
                
                return data
            
        except Exception as e:
            logger.debug(f"akshare数据收集异常: {e}")
        
        return None
    
    def _collect_with_eastmoney(self, stock_code: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """使用东方财富API收集数据（简化版本）"""
        try:
            # 这里实现东方财富API调用
            # 实际应用中需要研究具体的API接口
            
            # 模拟数据结构（在实际应用中替换为真实API调用）
            logger.warning("东方财富API收集功能需要实际API接口配置")
            
            # 返回示例数据结构
            return None
            
        except Exception as e:
            logger.debug(f"eastmoney数据收集异常: {e}")
        
        return None
    
    def _collect_stock_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """收集股票基本信息"""
        try:
            if self.data_sources['akshare']['available']:
                import akshare as ak
                
                symbol = stock_code.replace('.', '')
                info = ak.stock_individual_info_em(symbol=symbol)
                
                if info is not None and not info.empty:
                    info_dict = {}
                    for _, row in info.iterrows():
                        key = str(row['item']).strip()
                        value = str(row['value']).strip()
                        info_dict[key] = value
                    
                    return {
                        'stock_code': stock_code,
                        'company_name': info_dict.get('公司名称', ''),
                        'industry': info_dict.get('所属行业', ''),
                        'market_cap': info_dict.get('总市值', ''),
                        'pe_ratio': info_dict.get('市盈率', ''),
                        'pb_ratio': info_dict.get('市净率', ''),
                        'source': 'akshare',
                        'update_time': datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.debug(f"股票信息收集失败: {e}")
        
        return None
    
    def collect_market_indices(self, indices: List[str] = None) -> Dict[str, Any]:
        """
        收集市场指数数据
        
        Args:
            indices: 指数代码列表，默认收集主要指数
            
        Returns:
            Dict[str, Any]: 指数数据
        """
        if indices is None:
            indices = ['000001', '399001', '399006']  # 上证指数、深证成指、创业板指
        
        results = {
            'indices_data': {},
            'collection_metadata': {
                'requested_indices': indices,
                'successful_indices': [],
                'failed_indices': [],
                'collection_time': datetime.now().isoformat()
            }
        }
        
        for index_code in indices:
            try:
                index_data = self._collect_index_data(index_code)
                if index_data:
                    results['indices_data'][index_code] = index_data
                    results['collection_metadata']['successful_indices'].append(index_code)
                else:
                    results['collection_metadata']['failed_indices'].append(index_code)
                    
            except Exception as e:
                logger.error(f"收集指数{index_code}数据失败: {e}")
                results['collection_metadata']['failed_indices'].append(index_code)
        
        return results
    
    def _collect_index_data(self, index_code: str) -> Optional[Dict[str, Any]]:
        """收集指数数据"""
        try:
            if self.data_sources['akshare']['available']:
                import akshare as ak
                
                # 获取指数实时数据
                df = ak.stock_zh_index_spot()
                
                if df is not None and not df.empty:
                    # 查找指定指数
                    index_row = df[df['代码'] == index_code]
                    if not index_row.empty:
                        row = index_row.iloc[0]
                        return {
                            'index_code': index_code,
                            'name': str(row['名称']),
                            'current_price': float(row['最新价']),
                            'change': float(row['涨跌额']),
                            'change_pct': float(row['涨跌幅']),
                            'volume': int(row['成交量']),
                            'turnover': float(row['成交额']),
                            'update_time': datetime.now().isoformat(),
                            'source': 'akshare'
                        }
        
        except Exception as e:
            logger.debug(f"指数数据收集失败: {e}")
        
        return None
    
    def collect_financial_data(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        收集财务数据
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, Any]: 财务数据
        """
        results = {
            'financial_data': {},
            'collection_metadata': {
                'requested_stocks': stock_codes,
                'successful_stocks': [],
                'failed_stocks': [],
                'collection_time': datetime.now().isoformat()
            }
        }
        
        for stock_code in stock_codes:
            try:
                financial_data = self._collect_stock_financial_data(stock_code)
                if financial_data:
                    results['financial_data'][stock_code] = financial_data
                    results['collection_metadata']['successful_stocks'].append(stock_code)
                else:
                    results['collection_metadata']['failed_stocks'].append(stock_code)
                    
            except Exception as e:
                logger.error(f"收集股票{stock_code}财务数据失败: {e}")
                results['collection_metadata']['failed_stocks'].append(stock_code)
        
        return results
    
    def _collect_stock_financial_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """收集股票财务数据"""
        try:
            if self.data_sources['akshare']['available']:
                import akshare as ak
                
                symbol = stock_code.replace('.', '')
                
                # 获取财务指标
                df = ak.stock_financial_em(symbol=symbol)
                
                if df is not None and not df.empty:
                    # 取最新的财务数据
                    latest = df.iloc[0]
                    
                    return {
                        'stock_code': stock_code,
                        'report_date': str(latest.get('报告期', '')),
                        'revenue': float(latest.get('营业收入', 0)) if latest.get('营业收入') else None,
                        'net_profit': float(latest.get('净利润', 0)) if latest.get('净利润') else None,
                        'total_assets': float(latest.get('总资产', 0)) if latest.get('总资产') else None,
                        'total_equity': float(latest.get('净资产', 0)) if latest.get('净资产') else None,
                        'roe': float(latest.get('净资产收益率', 0)) if latest.get('净资产收益率') else None,
                        'roa': float(latest.get('总资产收益率', 0)) if latest.get('总资产收益率') else None,
                        'debt_ratio': float(latest.get('资产负债率', 0)) if latest.get('资产负债率') else None,
                        'source': 'akshare',
                        'update_time': datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.debug(f"财务数据收集失败: {e}")
        
        return None
    
    def _check_cache(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self.config['data_sources']['enable_cache']:
            return False
        
        if cache_key not in self.data_cache['last_update']:
            return False
        
        last_update = self.data_cache['last_update'][cache_key]
        cache_duration = timedelta(hours=self.config['data_sources']['cache_duration_hours'])
        
        return datetime.now() - last_update < cache_duration
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """从缓存获取数据"""
        return self.data_cache['stock_prices'].get(cache_key)
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """保存数据到缓存"""
        if self.config['data_sources']['enable_cache']:
            self.data_cache['stock_prices'][cache_key] = data
            self.data_cache['last_update'][cache_key] = datetime.now()
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            data: 价格数据列表
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if not data:
            return {'valid': False, 'issues': ['数据为空']}
        
        issues = []
        
        # 检查数据完整性
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        for i, record in enumerate(data):
            for field in required_fields:
                if field not in record or record[field] is None:
                    issues.append(f"记录{i}缺少字段{field}")
        
        # 检查价格合理性
        for i, record in enumerate(data):
            try:
                open_price = float(record.get('open', 0))
                high_price = float(record.get('high', 0))
                low_price = float(record.get('low', 0))
                close_price = float(record.get('close', 0))
                
                # 价格逻辑检查
                if high_price < max(open_price, close_price):
                    issues.append(f"记录{i}最高价异常")
                if low_price > min(open_price, close_price):
                    issues.append(f"记录{i}最低价异常")
                if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    issues.append(f"记录{i}存在非正价格")
                    
            except (ValueError, TypeError):
                issues.append(f"记录{i}价格数据类型错误")
        
        # 检查时间序列
        dates = []
        for record in data:
            try:
                date_str = record.get('date', '')
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_obj)
            except:
                issues.append(f"日期格式错误: {record.get('date')}")
        
        if len(dates) > 1:
            # 检查日期顺序
            sorted_dates = sorted(dates)
            if dates != sorted_dates:
                issues.append("日期序列未排序")
            
            # 检查重复日期
            if len(set(dates)) != len(dates):
                issues.append("存在重复日期")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'record_count': len(data),
            'validation_time': datetime.now().isoformat()
        }
    
    def save_collected_data(self, data: Dict[str, Any], filename: str = None) -> bool:
        """
        保存收集的数据
        
        Args:
            data: 数据字典
            filename: 文件名，默认自动生成
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"market_data_{timestamp}.json"
            
            data_dir = Path(self.config['storage']['data_directory'])
            data_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = data_dir / filename
            
            # 添加保存元数据
            save_data = {
                'data': data,
                'metadata': {
                    'save_time': datetime.now().isoformat(),
                    'data_type': 'market_data',
                    'version': '1.0'
                }
            }
            
            success = self.file_manager.write_json_file(file_path, save_data)
            
            if success:
                logger.info(f"数据已保存到: {file_path}")
            else:
                logger.error(f"数据保存失败: {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"保存数据时发生错误: {e}")
            return False
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        获取数据收集统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_requests = self.collection_stats['total_requests']
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.collection_stats['successful_requests'],
            'failed_requests': self.collection_stats['failed_requests'],
            'success_rate': self.collection_stats['successful_requests'] / max(total_requests, 1),
            'cache_hits': self.collection_stats['cache_hits'],
            'cache_hit_rate': self.collection_stats['cache_hits'] / max(total_requests, 1),
            'data_points_collected': self.collection_stats['data_points_collected'],
            'data_sources_status': {
                source: info['available'] for source, info in self.data_sources.items()
            },
            'cache_size': len(self.data_cache['stock_prices']),
            'config': self.config
        }


# 全局市场数据收集器实例
_global_market_data_collector = None


def get_market_data_collector() -> MarketDataCollector:
    """
    获取全局市场数据收集器实例
    
    Returns:
        MarketDataCollector: 收集器实例
    """
    global _global_market_data_collector
    
    if _global_market_data_collector is None:
        _global_market_data_collector = MarketDataCollector()
    
    return _global_market_data_collector


if __name__ == "__main__":
    # 使用示例
    collector = MarketDataCollector()
    
    # 测试股票代码
    test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    print("开始收集市场数据...")
    
    # 收集股票数据
    stock_data = collector.collect_stock_data(test_stocks)
    print(f"\n股票数据收集结果:")
    print(f"成功: {len(stock_data['collection_metadata']['successful_stocks'])}")
    print(f"失败: {len(stock_data['collection_metadata']['failed_stocks'])}")
    
    # 收集指数数据
    index_data = collector.collect_market_indices()
    print(f"\n指数数据收集结果:")
    print(f"成功: {len(index_data['collection_metadata']['successful_indices'])}")
    print(f"失败: {len(index_data['collection_metadata']['failed_indices'])}")
    
    # 收集财务数据
    financial_data = collector.collect_financial_data(test_stocks[:2])  # 只测试前两只
    print(f"\n财务数据收集结果:")
    print(f"成功: {len(financial_data['collection_metadata']['successful_stocks'])}")
    print(f"失败: {len(financial_data['collection_metadata']['failed_stocks'])}")
    
    # 获取统计信息
    stats = collector.get_collection_statistics()
    print(f"\n收集统计信息:")
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
    print(f"数据点数: {stats['data_points_collected']}")
    
    # 保存数据（如果有数据的话）
    if stock_data['collection_metadata']['successful_stocks']:
        saved = collector.save_collected_data(stock_data, 'test_stock_data.json')
        print(f"\n数据保存: {'成功' if saved else '失败'}") 
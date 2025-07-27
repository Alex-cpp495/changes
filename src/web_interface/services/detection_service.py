"""
异常检测服务
负责处理异常检测相关的业务逻辑
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..models.request_models import DetectionRequest, BatchProcessRequest
from ..models.response_models import DetectionResponse, BatchProcessResponse
from ..models.data_models import AnomalyResult, AnomalyLevel, DetectionCategory
from ...anomaly_detection import get_ensemble_detector
from ...continuous_learning import get_continuous_learning_system, record_metrics
from ...utils.logger import get_logger

logger = get_logger(__name__)


class DetectionService:
    """
    异常检测服务
    
    提供完整的异常检测功能：
    1. 单个文档检测
    2. 批量文档检测
    3. 实时检测流处理
    4. 结果缓存和优化
    5. 性能监控集成
    6. 错误处理和重试
    """
    
    def __init__(self):
        """初始化检测服务"""
        self.ensemble_detector = get_ensemble_detector()
        self.continuous_learning = get_continuous_learning_system()
        
        # 缓存设置
        self.result_cache: Dict[str, Tuple[AnomalyResult, datetime]] = {}
        self.cache_ttl = 3600  # 1小时缓存时间
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 线程池用于并发处理
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("异常检测服务初始化完成")
    
    async def detect_anomaly(self, request: DetectionRequest) -> DetectionResponse:
        """
        执行异常检测
        
        Args:
            request: 检测请求
            
        Returns:
            DetectionResponse: 检测响应
        """
        start_time = time.time()
        request_id = self._generate_request_id(request)
        
        try:
            # 更新统计
            self.performance_stats['total_requests'] += 1
            
            # 生成报告ID
            if not request.report_id:
                request.report_id = self._generate_report_id(request.report_content)
            
            # 检查缓存
            cached_result = self._get_cached_result(request.report_id)
            if cached_result:
                logger.info(f"使用缓存结果: {request.report_id}")
                self.performance_stats['cache_hits'] += 1
                
                processing_time = time.time() - start_time
                return self._create_detection_response(
                    request.report_id, cached_result, processing_time, request
                )
            
            self.performance_stats['cache_misses'] += 1
            
            # 执行检测
            anomaly_result = await self._perform_detection(request)
            
            # 缓存结果
            self._cache_result(request.report_id, anomaly_result)
            
            # 记录性能指标
            processing_time = time.time() - start_time
            await self._record_performance_metrics(processing_time, anomaly_result)
            
            # 更新统计
            self.performance_stats['successful_requests'] += 1
            self._update_average_processing_time(processing_time)
            
            # 创建响应
            response = self._create_detection_response(
                request.report_id, anomaly_result, processing_time, request
            )
            
            logger.info(f"检测完成: {request.report_id}, 耗时: {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            # 错误处理
            self.performance_stats['failed_requests'] += 1
            processing_time = time.time() - start_time
            
            logger.error(f"检测失败: {request.report_id}, 错误: {e}")
            
            # 返回错误响应
            return DetectionResponse(
                status="error",
                message=f"检测失败: {str(e)}",
                report_id=request.report_id or "unknown",
                processing_time=processing_time,
                anomaly_result=self._create_default_anomaly_result(),
                request_id=request_id
            )
    
    async def batch_detect(self, request: BatchProcessRequest) -> BatchProcessResponse:
        """
        批量异常检测
        
        Args:
            request: 批量处理请求
            
        Returns:
            BatchProcessResponse: 批量处理响应
        """
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}_{len(request.reports)}"
        
        try:
            logger.info(f"开始批量检测: {batch_id}, 报告数: {len(request.reports)}")
            
            # 准备检测请求
            detection_requests = []
            for i, report in enumerate(request.reports):
                det_request = DetectionRequest(
                    report_content=report['content'],
                    report_id=report.get('id', f"{batch_id}_item_{i}"),
                    report_title=report.get('title'),
                    metadata=report.get('metadata', {})
                )
                detection_requests.append(det_request)
            
            # 执行批量检测
            if request.parallel_processing:
                results = await self._parallel_batch_detect(detection_requests, request.max_workers)
            else:
                results = await self._sequential_batch_detect(detection_requests)
            
            # 统计结果
            total_items = len(detection_requests)
            successful_items = len([r for r in results if r['status'] == 'success'])
            failed_items = total_items - successful_items
            processing_time = time.time() - start_time
            
            # 生成批次摘要
            batch_summary = self._generate_batch_summary(results)
            
            # 创建响应
            response = BatchProcessResponse(
                status="success" if failed_items == 0 else "partial_success",
                message=f"批量处理完成: 成功{successful_items}个, 失败{failed_items}个",
                batch_id=batch_id,
                total_items=total_items,
                processed_items=total_items,
                successful_items=successful_items,
                failed_items=failed_items,
                processing_time=processing_time,
                individual_results=results if request.include_individual_results else None,
                batch_summary=batch_summary
            )
            
            logger.info(f"批量检测完成: {batch_id}, 耗时: {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"批量检测失败: {batch_id}, 错误: {e}")
            
            return BatchProcessResponse(
                status="error",
                message=f"批量检测失败: {str(e)}",
                batch_id=batch_id,
                total_items=len(request.reports),
                processed_items=0,
                successful_items=0,
                failed_items=len(request.reports),
                processing_time=processing_time,
                batch_summary={'error': str(e)}
            )
    
    async def _perform_detection(self, request: DetectionRequest) -> AnomalyResult:
        """执行实际的检测逻辑"""
        try:
            # 准备检测配置
            detection_config = {
                'enable_statistical': request.enable_statistical,
                'enable_behavioral': request.enable_behavioral,
                'enable_market': request.enable_market,
                'enable_semantic': request.enable_semantic,
                'custom_thresholds': request.custom_thresholds or {}
            }
            
            # 在线程池中执行检测（避免阻塞异步循环）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._sync_detect,
                request.report_content,
                detection_config,
                request.metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"执行检测失败: {e}")
            raise
    
    def _sync_detect(self, content: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> AnomalyResult:
        """同步检测函数（在线程池中运行）"""
        try:
            # 调用集成检测器
            detection_result = self.ensemble_detector.detect_anomalies(
                text=content,
                config=config,
                metadata=metadata
            )
            
            # 转换为AnomalyResult格式
            anomaly_result = AnomalyResult(
                overall_anomaly_score=detection_result.get('overall_anomaly_score', 0.0),
                overall_anomaly_level=self._determine_anomaly_level(
                    detection_result.get('overall_anomaly_score', 0.0)
                ),
                is_anomalous=detection_result.get('is_anomalous', False),
                confidence=detection_result.get('confidence', 0.0),
                detection_scores=detection_result.get('detection_scores', {}),
                detection_categories=self._extract_detection_categories(detection_result),
                anomaly_details=detection_result.get('anomaly_details'),
                risk_factors=detection_result.get('risk_factors', []),
                processing_metadata=detection_result.get('metadata', {})
            )
            
            return anomaly_result
            
        except Exception as e:
            logger.error(f"同步检测失败: {e}")
            raise
    
    async def _parallel_batch_detect(self, requests: List[DetectionRequest], 
                                   max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """并行批量检测"""
        max_workers = max_workers or min(4, len(requests))
        
        # 创建任务
        tasks = []
        for request in requests:
            task = asyncio.create_task(self._detect_single_for_batch(request))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'report_id': requests[i].report_id,
                    'status': 'error',
                    'error': str(result),
                    'anomaly_result': None
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _sequential_batch_detect(self, requests: List[DetectionRequest]) -> List[Dict[str, Any]]:
        """顺序批量检测"""
        results = []
        
        for request in requests:
            try:
                result = await self._detect_single_for_batch(request)
                results.append(result)
            except Exception as e:
                results.append({
                    'report_id': request.report_id,
                    'status': 'error',
                    'error': str(e),
                    'anomaly_result': None
                })
        
        return results
    
    async def _detect_single_for_batch(self, request: DetectionRequest) -> Dict[str, Any]:
        """批处理中的单个检测"""
        try:
            anomaly_result = await self._perform_detection(request)
            
            return {
                'report_id': request.report_id,
                'status': 'success',
                'anomaly_result': anomaly_result.dict(),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"批处理单个检测失败: {request.report_id}, 错误: {e}")
            raise
    
    def _generate_request_id(self, request: DetectionRequest) -> str:
        """生成请求ID"""
        content_hash = hashlib.md5(request.report_content.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"req_{timestamp}_{content_hash}"
    
    def _generate_report_id(self, content: str) -> str:
        """生成报告ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = int(time.time())
        return f"report_{timestamp}_{content_hash}"
    
    def _get_cached_result(self, report_id: str) -> Optional[AnomalyResult]:
        """获取缓存结果"""
        if report_id in self.result_cache:
            result, cached_time = self.result_cache[report_id]
            
            # 检查是否过期
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return result
            else:
                # 清理过期缓存
                del self.result_cache[report_id]
        
        return None
    
    def _cache_result(self, report_id: str, result: AnomalyResult):
        """缓存结果"""
        self.result_cache[report_id] = (result, datetime.now())
        
        # 限制缓存大小
        if len(self.result_cache) > 1000:
            # 删除最旧的条目
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k][1])
            del self.result_cache[oldest_key]
    
    def _determine_anomaly_level(self, score: float) -> AnomalyLevel:
        """确定异常级别"""
        if score >= 0.8:
            return AnomalyLevel.CRITICAL
        elif score >= 0.6:
            return AnomalyLevel.HIGH
        elif score >= 0.4:
            return AnomalyLevel.MEDIUM
        else:
            return AnomalyLevel.LOW
    
    def _extract_detection_categories(self, result: Dict[str, Any]) -> List[DetectionCategory]:
        """提取检测类别"""
        categories = []
        detection_scores = result.get('detection_scores', {})
        
        # 阈值设定
        threshold = 0.5
        
        for category, score in detection_scores.items():
            if score > threshold:
                if category == 'statistical':
                    categories.append(DetectionCategory.STATISTICAL)
                elif category == 'behavioral':
                    categories.append(DetectionCategory.BEHAVIORAL)
                elif category == 'market':
                    categories.append(DetectionCategory.MARKET)
                elif category == 'semantic':
                    categories.append(DetectionCategory.SEMANTIC)
        
        return categories
    
    def _create_detection_response(self, report_id: str, anomaly_result: AnomalyResult,
                                 processing_time: float, request: DetectionRequest) -> DetectionResponse:
        """创建检测响应"""
        # 生成解释信息
        explanations = None
        if request.include_explanations:
            explanations = self._generate_explanations(anomaly_result)
        
        # 生成详细分数
        detailed_scores = None
        if request.include_raw_scores:
            detailed_scores = anomaly_result.detection_scores
        
        # 生成建议
        recommendations = self._generate_recommendations(anomaly_result)
        
        return DetectionResponse(
            status="success",
            message="检测完成",
            report_id=report_id,
            processing_time=processing_time,
            anomaly_result=anomaly_result,
            detailed_scores=detailed_scores,
            explanations=explanations,
            recommendations=recommendations
        )
    
    def _create_default_anomaly_result(self) -> AnomalyResult:
        """创建默认异常结果（用于错误情况）"""
        return AnomalyResult(
            overall_anomaly_score=0.0,
            overall_anomaly_level=AnomalyLevel.LOW,
            is_anomalous=False,
            confidence=0.0,
            detection_scores={},
            detection_categories=[]
        )
    
    def _generate_explanations(self, result: AnomalyResult) -> Dict[str, Any]:
        """生成解释信息"""
        explanations = {
            'overall_explanation': f"检测到{result.overall_anomaly_level.value}级别异常，置信度{result.confidence:.2%}",
            'category_explanations': {}
        }
        
        # 为每个触发的类别生成解释
        for category in result.detection_categories:
            score = result.detection_scores.get(category.value, 0.0)
            explanations['category_explanations'][category.value] = {
                'score': score,
                'description': self._get_category_description(category, score)
            }
        
        return explanations
    
    def _get_category_description(self, category: DetectionCategory, score: float) -> str:
        """获取类别描述"""
        descriptions = {
            DetectionCategory.STATISTICAL: f"统计指标异常 (分数: {score:.2f})",
            DetectionCategory.BEHAVIORAL: f"行为模式异常 (分数: {score:.2f})",
            DetectionCategory.MARKET: f"市场数据异常 (分数: {score:.2f})",
            DetectionCategory.SEMANTIC: f"语义内容异常 (分数: {score:.2f})"
        }
        return descriptions.get(category, f"未知类别异常 (分数: {score:.2f})")
    
    def _generate_recommendations(self, result: AnomalyResult) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if result.is_anomalous:
            recommendations.append("建议进一步核实报告内容的准确性")
            
            if DetectionCategory.STATISTICAL in result.detection_categories:
                recommendations.append("关注财务数据的统计一致性")
            
            if DetectionCategory.BEHAVIORAL in result.detection_categories:
                recommendations.append("检查业务行为模式的合理性")
            
            if DetectionCategory.MARKET in result.detection_categories:
                recommendations.append("验证市场数据和行业对比的准确性")
            
            if DetectionCategory.SEMANTIC in result.detection_categories:
                recommendations.append("审查文本内容的语义一致性和逻辑性")
        else:
            recommendations.append("报告内容未发现明显异常")
        
        return recommendations
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成批次摘要"""
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            return {'message': '没有成功的检测结果'}
        
        # 统计异常情况
        anomalous_count = 0
        total_score = 0.0
        category_counts = {}
        
        for result in successful_results:
            anomaly_result = result['anomaly_result']
            if anomaly_result and anomaly_result.get('is_anomalous'):
                anomalous_count += 1
            
            score = anomaly_result.get('overall_anomaly_score', 0.0) if anomaly_result else 0.0
            total_score += score
            
            # 统计类别
            categories = anomaly_result.get('detection_categories', []) if anomaly_result else []
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        summary = {
            'total_analyzed': len(successful_results),
            'anomalous_reports': anomalous_count,
            'anomaly_rate': anomalous_count / len(successful_results) if successful_results else 0.0,
            'average_anomaly_score': total_score / len(successful_results) if successful_results else 0.0,
            'category_breakdown': category_counts,
            'risk_level': 'HIGH' if anomalous_count / len(successful_results) > 0.3 else 'MEDIUM' if anomalous_count > 0 else 'LOW'
        }
        
        return summary
    
    async def _record_performance_metrics(self, processing_time: float, result: AnomalyResult):
        """记录性能指标"""
        try:
            # 记录到持续学习系统
            record_metrics(
                prediction_time=processing_time,
                accuracy=None,  # 准确率需要用户反馈后才能确定
                confidence=result.confidence
            )
        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
    
    def _update_average_processing_time(self, processing_time: float):
        """更新平均处理时间"""
        current_avg = self.performance_stats['average_processing_time']
        successful_count = self.performance_stats['successful_requests']
        
        if successful_count == 1:
            self.performance_stats['average_processing_time'] = processing_time
        else:
            # 计算移动平均
            new_avg = (current_avg * (successful_count - 1) + processing_time) / successful_count
            self.performance_stats['average_processing_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        stats['success_rate'] = (
            stats['successful_requests'] / stats['total_requests']
            if stats['total_requests'] > 0 else 0.0
        )
        stats['cached_results_count'] = len(self.result_cache)
        
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        self.result_cache.clear()
        logger.info("检测结果缓存已清理")


# 全局检测服务实例
_global_detection_service = None


def get_detection_service() -> DetectionService:
    """
    获取全局检测服务实例
    
    Returns:
        DetectionService: 检测服务实例
    """
    global _global_detection_service
    
    if _global_detection_service is None:
        _global_detection_service = DetectionService()
    
    return _global_detection_service 
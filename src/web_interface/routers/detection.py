"""
异常检测API路由
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional

from ..models.request_models import DetectionRequest, BatchProcessRequest
from ..models.response_models import DetectionResponse, SuccessResponse
from ..services.detection_service import get_detection_service
from ...data_processing.processors.report_processor import get_report_processor
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(
    request: DetectionRequest,
    detection_service = Depends(get_detection_service)
):
    """
    执行异常检测
    """
    try:
        logger.info(f"收到异常检测请求: {request.report_title}")
        
        result = await detection_service.detect_anomaly(request)
        
        logger.info(f"异常检测完成: {result.status}")
        return result
        
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")


@router.post("/batch", response_model=SuccessResponse)
async def batch_detect(
    request: BatchProcessRequest,
    detection_service = Depends(get_detection_service)
):
    """
    批量异常检测
    """
    try:
        logger.info(f"收到批量检测请求，数据条数: {len(request.reports_data)}")
        
        # 获取批量处理器
        processor = get_report_processor({
            'batch_size': request.batch_size or 10
        })
        
        # 批量处理
        results = await processor.process_batch_data(
            data_source=request.reports_data,
            data_format='list'
        )
        
        # 导出结果（如果需要）
        export_path = None
        if request.export_results:
            export_path = processor.export_results(
                results, 
                f"data/results/batch_results.{request.export_format or 'json'}",
                request.export_format or 'json'
            )
        
        return SuccessResponse(
            status="success",
            message=f"批量检测完成，成功处理 {results['summary']['successful']} 条",
            data={
                "summary": results['summary'],
                "export_path": export_path,
                "anomaly_distribution": results['anomaly_distribution']
            }
        )
        
    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量检测失败: {str(e)}")


@router.post("/upload", response_model=SuccessResponse)
async def upload_file(
    file: UploadFile = File(...),
    format: Optional[str] = None,
    detection_service = Depends(get_detection_service)
):
    """
    上传文件并检测
    """
    try:
        logger.info(f"收到文件上传: {file.filename}")
        
        # 检查文件格式
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 读取文件内容
        content = await file.read()
        
        # 根据文件类型处理
        if file.content_type == "text/plain":
            text_content = content.decode('utf-8')
            
            # 创建检测请求
            request = DetectionRequest(
                report_title=file.filename,
                report_content=text_content,
                include_explanations=True
            )
            
            # 执行检测
            result = await detection_service.detect_anomaly(request)
            
            return SuccessResponse(
                status="success",
                message="文件上传并检测完成",
                data={
                    "filename": file.filename,
                    "detection_result": result.dict()
                }
            )
        else:
            # 对于其他格式，返回文件内容供前端处理
            return SuccessResponse(
                status="success", 
                message="文件上传成功",
                data={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content),
                    "content": content.decode('utf-8', errors='ignore')[:1000]  # 只返回前1000个字符
                }
            )
            
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.get("/history")
async def get_detection_history(
    limit: int = 50,
    offset: int = 0,
    detection_service = Depends(get_detection_service)
):
    """
    获取检测历史
    """
    try:
        # 这里应该从数据库获取历史记录
        # 暂时返回模拟数据
        history = [
            {
                "report_id": "report_001",
                "title": "某公司Q3财报",
                "detection_time": "2024-01-22T14:30:00",
                "anomaly_score": 0.75,
                "anomaly_level": "HIGH",
                "is_anomalous": True,
                "confidence": 0.92
            },
            {
                "report_id": "report_002", 
                "title": "行业分析报告",
                "detection_time": "2024-01-22T13:15:00",
                "anomaly_score": 0.25,
                "anomaly_level": "LOW",
                "is_anomalous": False,
                "confidence": 0.88
            }
        ]
        
        return {
            "status": "success",
            "data": history[offset:offset+limit],
            "total": len(history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"获取检测历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取检测历史失败")


@router.get("/stats")
async def get_detection_stats(
    detection_service = Depends(get_detection_service)
):
    """
    获取检测统计信息
    """
    try:
        stats = detection_service.get_performance_stats()
        
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"获取检测统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取检测统计失败") 
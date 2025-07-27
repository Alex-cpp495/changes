"""
FastAPI主应用程序
东吴证券研报异常检测系统Web接口
"""

import time
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from .routers import detection, feedback, monitoring, reports, config, auth
from .services import get_detection_service, get_monitoring_service
from ..continuous_learning import initialize_continuous_learning
from ..optimization.performance_optimizer import get_performance_optimizer
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 全局变量
continuous_learning_system = None
performance_optimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时的操作
    logger.info("🚀 启动东吴证券研报异常检测系统")
    
    # 初始化持续学习系统
    global continuous_learning_system, performance_optimizer
    continuous_learning_system = initialize_continuous_learning(auto_start=True)
    logger.info("✅ 持续学习系统已启动")
    
    # 初始化性能优化器
    performance_optimizer = get_performance_optimizer({
        'max_concurrent': 50,
        'cpu_warning': 80.0,
        'memory_warning': 80.0
    })
    performance_optimizer.start_monitoring()
    logger.info("✅ 性能优化器已启动")
    
    # 优化批量处理
    performance_optimizer.optimize_for_batch_processing()
    
    yield
    
    # 关闭时的操作
    logger.info("🛑 关闭东吴证券研报异常检测系统")
    
    if continuous_learning_system:
        continuous_learning_system.stop_system()
        logger.info("✅ 持续学习系统已停止")
    
    if performance_optimizer:
        performance_optimizer.stop_monitoring()
        logger.info("✅ 性能优化器已停止")


# 创建FastAPI应用
app = FastAPI(
    title="东吴证券研报异常检测系统",
    description="基于AI的研报异常检测与分析系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 性能监控中间件
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """性能监控中间件"""
    start_time = time.time()
    
    # 记录请求
    logger.debug(f"收到请求: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录性能指标
        if performance_optimizer:
            await performance_optimizer.concurrency_limiter.acquire()
            try:
                # 这里可以添加更多的性能指标记录
                pass
            finally:
                performance_optimizer.concurrency_limiter.release()
        
        logger.debug(f"请求完成: {request.method} {request.url} - {process_time:.3f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"请求失败: {request.method} {request.url} - {process_time:.3f}s - {e}")
        raise


# 静态文件服务
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# 模板配置
templates = Jinja2Templates(directory="frontend/templates")

# 包含路由
app.include_router(detection.router, prefix="/api/detection", tags=["异常检测"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["用户反馈"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["系统监控"])
app.include_router(reports.router, prefix="/api/reports", tags=["报告管理"])
app.include_router(config.router, prefix="/api/config", tags=["系统配置"])
app.include_router(auth.router, prefix="/api/auth", tags=["用户认证"])


# 前端页面路由
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """仪表板页面"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/detection", response_class=HTMLResponse)
async def detection_page(request: Request):
    """异常检测页面"""
    return templates.TemplateResponse("detection.html", {"request": request})


@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    """反馈页面"""
    return templates.TemplateResponse("feedback.html", {"request": request})


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """监控页面"""
    return templates.TemplateResponse("monitoring.html", {"request": request})


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """配置页面"""
    return templates.TemplateResponse("config.html", {"request": request})


# API端点
@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {
            "continuous_learning": continuous_learning_system is not None,
            "performance_optimizer": performance_optimizer is not None
        }
    }


@app.get("/api/dashboard")
async def get_dashboard_data():
    """获取仪表板数据"""
    try:
        monitoring_service = get_monitoring_service()
        dashboard_data = await monitoring_service.get_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error(f"获取仪表板数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取仪表板数据失败")


@app.get("/api/performance")
async def get_performance_data():
    """获取性能数据"""
    try:
        if not performance_optimizer:
            raise HTTPException(status_code=503, detail="性能优化器未启动")
        
        performance_report = performance_optimizer.get_performance_report()
        recommendations = performance_optimizer.get_optimization_recommendations()
        
        return {
            "status": "success",
            "performance_report": performance_report,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取性能数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取性能数据失败")


@app.post("/api/optimize")
async def trigger_optimization():
    """触发系统优化"""
    try:
        if not performance_optimizer:
            raise HTTPException(status_code=503, detail="性能优化器未启动")
        
        # 执行批量处理优化
        optimization_result = performance_optimizer.optimize_for_batch_processing()
        
        return {
            "status": "success",
            "message": "系统优化完成",
            "optimization_result": optimization_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"系统优化失败: {e}")
        raise HTTPException(status_code=500, detail="系统优化失败")


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return {
        "status": "error",
        "error_code": exc.status_code,
        "message": exc.detail,
        "timestamp": time.time()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return {
        "status": "error",
        "error_code": 500,
        "message": "服务器内部错误",
        "timestamp": time.time()
    }


# 依赖项
def get_current_system_status() -> Dict[str, Any]:
    """获取当前系统状态"""
    return {
        "continuous_learning": continuous_learning_system is not None,
        "performance_optimizer": performance_optimizer is not None,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    # 开发环境启动
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
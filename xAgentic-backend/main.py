import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.chat import router as chat_router
from api.mcp import router as mcp_router
from cfg.setting import get_settings
from services.service_manager import service_manager
from utils.unified_logger import initialize_logging, get_logger

# 初始化统一日志系统
logging_config = initialize_logging(
    log_level=20,  # INFO
    log_dir="logs",
    main_log_filename="xagentic.log",
    enable_console=True,
    enable_file=True
)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    logger.info("正在初始化服务...")
    success = service_manager.initialize()
    if not success:
        logger.error("服务初始化失败，应用可能无法正常工作")
    else:
        logger.info("服务初始化完成")
    
    yield
    
    # 关闭时清理资源
    logger.info("服务清理完成")

app = FastAPI(
    title="xAgentic Chat API", 
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件以支持前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(mcp_router)

@app.get("/")
def read_root():
    return {"message": "xAgentic Chat API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "xagentic-backend"}

if __name__ == "__main__":
    uvicorn.run(app, port=get_settings().port, host=get_settings().host)
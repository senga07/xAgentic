"""
LangSmith工具类

提供LangSmith追踪和监控功能
"""
import os
import logging
from typing import Optional, Dict, Any
from functools import wraps
from cfg.setting import get_settings

logger = logging.getLogger(__name__)

class LangSmithManager:
    """LangSmith管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._enabled = False
        
    @property
    def client(self):
        """获取LangSmith客户端"""
        if self._client is None and self._enabled:
            try:
                import langsmith
                self._client = langsmith.Client()
            except ImportError:
                logger.warning("无法导入langsmith，追踪功能不可用")
                self._enabled = False
        return self._client
    
    def is_enabled(self) -> bool:
        """检查LangSmith是否启用"""
        # 检查环境变量或设置中的API密钥
        api_key = self.settings.langsmith_api_key
        tracing_enabled = self.settings.langsmith_tracing_v2
        
        return (tracing_enabled and 
                bool(api_key) and
                self._enabled)
    
    def enable_tracing(self):
        """启用追踪"""
        # 检查环境变量或设置
        api_key = self.settings.langsmith_api_key
        project = self.settings.langsmith_project
        endpoint = self.settings.langsmith_endpoint
        tracing_enabled = self.settings.langsmith_tracing_v2
        
        if tracing_enabled and api_key:
            # 确保环境变量已设置
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_PROJECT"] = project
            os.environ["LANGSMITH_ENDPOINT"] = endpoint
            self._enabled = True
            logger.info(f"LangSmith追踪已启用 - 项目: {project}")
    
    def create_run(self, name: str, run_type: str = "chain", inputs: dict = None, **kwargs) -> Optional[Any]:
        """创建追踪运行"""
        if not self.is_enabled():
            return None
            
        try:
            if self.client:
                return self.client.create_run(
                    name=name,
                    run_type=run_type,
                    inputs=inputs or {},
                    project_name=self.settings.langsmith_project,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"创建LangSmith运行失败: {e}")
        return None
    
    def log_feedback(self, run_id: str, key: str, value: Any, **kwargs):
        """记录反馈"""
        if not self.is_enabled() or not self.client:
            return
            
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                value=value,
                **kwargs
            )
        except Exception as e:
            logger.error(f"记录LangSmith反馈失败: {e}")


# 全局LangSmith管理器实例
langsmith_manager = LangSmithManager()


def trace_langsmith(name: Optional[str] = None, run_type: str = "chain"):
    """LangSmith追踪装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_manager.is_enabled():
                return func(*args, **kwargs)
            
            run_name = name or f"{func.__module__}.{func.__name__}"
            run = langsmith_manager.create_run(run_name, run_type)
            
            try:
                result = func(*args, **kwargs)
                if run:
                    run.end(outputs=result)
                return result
            except Exception as e:
                if run:
                    run.end(error=str(e))
                raise
                
        return wrapper
    return decorator


def get_langsmith_url(run_id: str) -> Optional[str]:
    """获取LangSmith运行URL"""
    if not langsmith_manager.is_enabled():
        return None
    
    try:
        base_url = langsmith_manager.settings.langsmith_endpoint.replace("api.", "")
        return f"{base_url}/o/{langsmith_manager.settings.langsmith_project}/r/{run_id}"
    except Exception as e:
        logger.error(f"生成LangSmith URL失败: {e}")
        return None

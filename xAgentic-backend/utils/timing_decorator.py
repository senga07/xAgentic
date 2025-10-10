"""
时间统计装饰器模块

为工具和模型调用添加时间统计功能
"""
import time
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict


def timing_decorator(func_name: str = None):
    """
    时间统计装饰器
    
    Args:
        func_name: 函数名称，用于日志记录
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                # 记录时间统计
                end_time = time.time()
                duration = end_time - start_time
                logging.info(f"{function_name} 执行耗时: {duration:.2f}秒")
                
                # 如果结果是字典，添加时间信息
                if isinstance(result, dict):
                    result["timing"] = {
                        "start_time": datetime.fromtimestamp(start_time).isoformat(),
                        "end_time": datetime.fromtimestamp(end_time).isoformat(),
                        "duration": round(duration, 2)
                    }
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logging.error(f"{function_name} 执行失败: {e}, 耗时: {duration:.2f}秒")
                raise
                
        return wrapper
    return decorator


def async_timing_decorator(func_name: str = None):
    """
    异步时间统计装饰器
    
    Args:
        func_name: 函数名称，用于日志记录
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录时间统计
                end_time = time.time()
                duration = end_time - start_time
                logging.info(f"{function_name} 执行耗时: {duration:.2f}秒")
                
                # 如果结果是字典，添加时间信息
                if isinstance(result, dict):
                    result["timing"] = {
                        "start_time": datetime.fromtimestamp(start_time).isoformat(),
                        "end_time": datetime.fromtimestamp(end_time).isoformat(),
                        "duration": round(duration, 2)
                    }
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logging.error(f"{function_name} 执行失败: {e}, 耗时: {duration:.2f}秒")
                raise
                
        return wrapper
    return decorator


def log_timing_info(operation: str, duration: float, details: Dict[str, Any] = None):
    """
    记录时间统计信息到日志
    
    Args:
        operation: 操作名称
        duration: 耗时（秒）
        details: 详细信息
    """
    timing_info = {
        "operation": operation,
        "duration": round(duration, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        timing_info.update(details)
    
    logging.info(f"⏱️ 时间统计: {operation} 耗时 {duration:.2f}秒")
    if details:
        logging.debug(f"详细信息: {details}")


def create_timing_context(operation: str):
    """
    创建时间统计上下文管理器
    
    Args:
        operation: 操作名称
    """
    class TimingContext:
        def __init__(self, operation: str):
            self.operation = operation
            self.start_time = None
            self.end_time = None
            self.duration = None
            
        def __enter__(self):
            self.start_time = time.time()
            logging.info(f"开始执行: {self.operation}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            
            if exc_type is None:
                logging.info(f"✅ {self.operation} 完成，耗时: {self.duration:.2f}秒")
            else:
                logging.error(f"❌ {self.operation} 失败，耗时: {self.duration:.2f}秒")
                
        def get_timing_info(self):
            """获取时间统计信息"""
            if self.start_time and self.end_time:
                return {
                    "operation": self.operation,
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                    "duration": round(self.duration, 2)
                }
            return None
    
    return TimingContext(operation)

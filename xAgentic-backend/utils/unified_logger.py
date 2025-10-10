"""
统一日志管理器

提供统一的日志配置和管理入口，支持多种日志类型和配置
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


class UnifiedLoggerManager:
    """统一日志管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._loggers: Dict[str, logging.Logger] = {}
            self._log_dir = Path("logs")
            self._log_dir.mkdir(exist_ok=True)
            self._initialized = True
    
    def initialize(
        self,
        log_level: int = logging.INFO,
        log_dir: str = "logs",
        main_log_filename: str = "xagentic.log",
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> Dict[str, Any]:
        """
        初始化统一日志系统
        
        Args:
            log_level: 日志级别
            log_dir: 日志文件目录
            main_log_filename: 主日志文件名
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            max_file_size: 单个日志文件最大大小（字节）
            backup_count: 保留的备份文件数量
            
        Returns:
            Dict: 初始化结果信息
        """
        try:
            # 更新日志目录
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(exist_ok=True)
            
            # 清除现有的根日志处理器
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # 设置根日志级别
            root_logger.setLevel(log_level)
            
            # 创建格式化器
            detailed_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            simple_formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 文件处理器
            if enable_file:
                main_log_path = self._log_dir / main_log_filename
                file_handler = RotatingFileHandler(
                    main_log_path,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(file_handler)
            
            # 控制台处理器
            if enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(simple_formatter)
                root_logger.addHandler(console_handler)
            
            # 设置第三方库的日志级别
            self._configure_third_party_loggers()
            
            # 记录初始化信息
            root_logger.info(f"统一日志系统已初始化 - 日志目录: {self._log_dir}")
            root_logger.info(f"日志级别: {logging.getLevelName(log_level)}")
            
            return {
                "status": "success",
                "log_dir": str(self._log_dir),
                "log_level": logging.getLevelName(log_level),
                "handlers_count": len(root_logger.handlers)
            }
            
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            return {"status": "error", "error": str(e)}
    
    
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]
    
    
    def log_error(self, logger: logging.Logger, error: Exception, context: str = ""):
        """记录错误信息的便捷方法"""
        error_msg = f"错误: {str(error)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        logger.error(error_msg, exc_info=True)
    
    def log_function_call(self, logger: logging.Logger, func_name: str, args: dict = None, result: any = None):
        """记录函数调用的便捷方法"""
        args_str = f"参数: {args}" if args else "无参数"
        result_str = f"返回值: {result}" if result is not None else "无返回值"
        logger.debug(f"函数调用: {func_name} - {args_str} - {result_str}")
    
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志系统统计信息"""
        return {
            "loggers_count": len(self._loggers),
            "log_dir": str(self._log_dir),
            "loggers": list(self._loggers.keys()),
            "log_files": [f.name for f in self._log_dir.glob("*.log")] if self._log_dir.exists() else []
        }
    
    def _configure_third_party_loggers(self):
        """配置第三方库的日志级别"""
        third_party_loggers = {
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'httpx': logging.WARNING,
            'asyncio': logging.WARNING,
            'uvicorn': logging.INFO,
            'fastapi': logging.INFO
        }
        
        for logger_name, level in third_party_loggers.items():
            logging.getLogger(logger_name).setLevel(level)


# 全局日志管理器实例
unified_logger_manager = UnifiedLoggerManager()


# 便捷函数
def initialize_logging(**kwargs) -> Dict[str, Any]:
    """初始化日志系统的便捷函数"""
    return unified_logger_manager.initialize(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return unified_logger_manager.get_logger(name)




def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """记录错误的便捷函数"""
    unified_logger_manager.log_error(logger, error, context)


def log_function_call(logger: logging.Logger, func_name: str, args: dict = None, result: any = None):
    """记录函数调用的便捷函数"""
    unified_logger_manager.log_function_call(logger, func_name, args, result)




def get_log_stats() -> Dict[str, Any]:
    """获取日志统计信息的便捷函数"""
    return unified_logger_manager.get_log_stats()

"""
服务管理器

提供基本的服务管理功能，避免循环依赖
"""

import os

from colorama import Fore, Style

from cfg.config import Config
from llm_provider.base import get_llm
from mcp_.client import MCPClientManager
from mcp_.manager import mcp_manager
from utils.langsmith_utils import langsmith_manager
from utils.unified_logger import get_logger
from langgraph.store.memory import InMemoryStore
from memory.embeddings import Embeddings
from langgraph.store.base import IndexConfig


class ServiceManager:
    """服务管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_logger(__name__)
            self.config = None
            self.strategic_llm = None
            self.fast_llm = None
            self.code_llm = None
            self._initialized = True
            self.mcp_tools = None
            self.store = None

    
    def initialize(self) -> bool:
        """初始化基本服务"""
        try:
            self.logger.info("开始初始化服务管理器...")
            
            # 1. 初始化配置
            self.config = Config()
            
            # 2. 启用LangSmith追踪
            langsmith_manager.enable_tracing()
            _setup_langsmith_tracing(self.config)
            
            # 3. 初始化LLM实例
            self._initialize_llms()

            # 4. 初始化mcp工具
            self._initialize_mcp_tools()

            embedding = Embeddings(self.config.embedding_provider, self.config.embedding_model).get_embeddings()
            self.store = InMemoryStore(index=IndexConfig(dims=1024,embed = embedding))
            self.logger.info("服务管理器初始化完成")
            return True
        except Exception as e:
            self.logger.error(f"服务初始化失败: {e}")
            return False
    
    def _initialize_llms(self):
        """初始化LLM实例"""
        try:
            # 战略LLM
            self.strategic_llm = get_llm(
                llm_provider=self.config.strategic_llm_provider,
                model=self.config.strategic_llm_model,
                **self.config.llm_kwargs
            ).llm
            
            # 快速LLM
            self.fast_llm = get_llm(
                llm_provider=self.config.fast_llm_provider,
                model=self.config.fast_llm_model,
                **self.config.llm_kwargs
            ).llm
            
            # 代码LLM
            self.code_llm = get_llm(
                llm_provider=self.config.coding_llm_provider,
                model=self.config.coding_llm_model,
                **self.config.llm_kwargs
            ).llm
            
            self.logger.info("LLM实例初始化完成")
        except Exception as e:
            self.logger.error(f"LLM初始化失败: {e}")
            raise
    
    def get_llms(self):
        """获取所有LLM实例"""
        return {
            'strategic_llm': self.strategic_llm,
            'fast_llm': self.fast_llm,
            'code_llm': self.code_llm
        }
    
    def get_config(self):
        """获取配置实例"""
        return self.config
    
    def get_mcp_tools(self):
        """获取MCP工具列表"""
        return self.mcp_tools or []
    

    def get_all_tools(self):
        """获取所有工具"""
        from tools.search_tools import web_search
        from tools.code_tools import execute_python_code
        from tools.time_tools import get_current_time, calculate_date_offset, get_time_info

        local_tools = [web_search, execute_python_code, get_current_time, calculate_date_offset, get_time_info]
        return local_tools + self.get_mcp_tools()



    def _initialize_mcp_tools(self):
        """初始化MCP工具"""
        try:
            mcp_configs = mcp_manager.load_config()
            self.mcp_tools = []
            
            if len(mcp_configs) > 0:
                self.mcp_client_manager = MCPClientManager(mcp_configs)

                import concurrent.futures
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._run_mcp_tools_async)
                        tools = future.result()
                    
                    self.mcp_tools.extend(tools)
                except Exception as e:
                    self.logger.warning(f"MCP工具加载过程中出现错误: {e}")
                    # 继续执行，不中断服务初始化
                    
            self.logger.info(f"成功加载 {len(self.mcp_tools)} 个MCP工具")
            
        except Exception as e:
            self.logger.error(f"MCP工具加载失败: {e}")
            # 设置空列表，确保服务可以继续运行
            self.mcp_tools = []
    
 

    def _run_mcp_tools_async(self):
        """在单独的事件循环中运行MCP工具加载"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.mcp_client_manager.get_all_tools()
            )
        finally:
            loop.close()


def _setup_langsmith_tracing(config) -> None:
    """配置LangSmith追踪"""
    if config.langsmith_tracing_v2 and config.langsmith_api_key:
        # 设置环境变量
        os.environ["LANGSMITH_TRACING_V2"] = str(config.langsmith_tracing_v2)
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint

        # 导入并配置LangSmith
        try:
            import langsmith
            print(f"{Fore.GREEN}LangSmith追踪已启用 - 项目: {config.langsmith_project}{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.YELLOW}警告: 无法导入langsmith，追踪功能可能不可用{Style.RESET_ALL}")


# 全局服务管理器实例
service_manager = ServiceManager()

"""
基础图类 - 提取公共方法和功能

包含所有图类的公共方法，避免代码重复
"""
import time
from datetime import datetime
from typing import List, TypedDict, AsyncIterator

from langchain_core.messages.base import BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from services.service_manager import service_manager
from tools.code_tools import *
from tools.search_tools import *
from tools.time_tools import *
from utils.custom_serializer import CustomSerializer
from utils.unified_logger import get_logger


class BaseGraphState(TypedDict):
    """基础图状态 - 包含所有图类的公共状态字段"""
    # 消息和通信
    messages: List[BaseMessage]
    streaming_chunks: List[Dict[str, Any]]
    
    # 状态管理
    status: str
    error: str
    
    # 上下文信息
    thread_id: str
    
    # 元数据
    timing_info: Dict[str, Any]


class BaseGraph:
    """基础图类 - 提供所有图类的公共功能"""
    
    def __init__(self, graph_name: str):
        """初始化基础图类"""
        self.thread_id = None
        self.logger = get_logger(__name__)
        self.graph_name = graph_name
        
        # 使用全局服务管理器中的配置和LLM实例
        self.config = service_manager.get_config()
        llms = service_manager.get_llms()
        self.strategic_llm = llms.get('strategic_llm')
        self.fast_llm = llms.get('fast_llm')
        self.vision_llm = llms.get('vision_llm')
        
        # 获取工具
        self.mcp_tools = service_manager.get_mcp_tools()
        self.local_tools = [
            web_search, execute_python_code, get_current_time,
            calculate_date_offset, get_time_info
        ]
        self.all_tools = self.local_tools + self.mcp_tools
        
        self.store = service_manager.store
    
    def _initialize_checkpointer(self, graph_class):
        """初始化检查点存储 - 公共方法"""
        if not hasattr(graph_class, '_initialized') or not graph_class._initialized:
            # 初始化LangGraph检查点存储
            custom_serializer = CustomSerializer()
            graph_class._shared_checkpointer = MemorySaver(serde=custom_serializer)
            graph_class._initialized = True
            self.logger.info(f"{self.graph_name} 首次初始化完成")
    
    def _add_streaming_chunk(self, state: Dict[str, Any], step: str, message: str,
                             data: Dict[str, Any] = None) -> None:
        """添加流式输出块的辅助方法 - 公共方法"""
        if "streaming_chunks" not in state:
            state["streaming_chunks"] = []
        
        chunk = {
            "step": step,
            "message": message
        }
        if data:
            chunk["data"] = data
        state["streaming_chunks"].append(chunk)
    
    def _extract_execution_result(self, result, node_description: str = ""):
        """提取执行结果 - 公共方法"""
        execution_result = ""
        try:
            if result and isinstance(result, dict):
                if "messages" in result:
                    messages = result["messages"]
                    if messages:
                        # 查找最后一个AI消息
                        for msg in reversed(messages):
                            if hasattr(msg, 'content'):
                                msg_type = str(type(msg)).lower()
                                if 'ai' in msg_type or 'assistant' in msg_type:
                                    execution_result = msg.content
                                    break
                        else:
                            # 如果没有找到AI消息，使用最后一个消息
                            last_message = messages[-1]
                            if hasattr(last_message, 'content'):
                                execution_result = last_message.content
                            else:
                                execution_result = str(last_message)
                elif "output" in result:
                    execution_result = str(result["output"])
                else:
                    execution_result = str(result)
            elif result and hasattr(result, 'content'):
                # 如果result直接是消息对象（直接LLM调用）
                execution_result = result.content
            elif result:
                # 其他情况，转换为字符串
                execution_result = str(result)
            
            # 如果结果太短或包含错误信息，提供更详细的信息
            if len(execution_result) < 10 or "sorry" in execution_result.lower():
                execution_result = f"任务描述: {node_description}\n执行状态: 已完成\n结果: {execution_result}"
                
        except Exception as e:
            self.logger.error(f"提取执行结果时出错: {str(e)}")
            execution_result = f"执行完成，但结果提取时出现错误: {str(e)}"
        
        return execution_result
    
    def _get_timing_info(self, start_time: float, operation_name: str) -> Dict[str, Any]:
        """获取时间信息 - 公共方法"""
        duration = time.time() - start_time
        return {
            f"{operation_name}_duration": round(duration, 2),
            f"{operation_name}_timestamp": datetime.now().isoformat()
        }
    
    
    def _format_tools_list(self, tools: List) -> List[str]:
        """格式化工具列表 - 公共方法"""
        tools_list = []
        for t in tools:
            if hasattr(t, 'name') and hasattr(t, 'description'):
                tools_list.append(f"- {t.name}: {t.description}")
            else:
                tools_list.append(f"- {t.__name__ if hasattr(t, '__name__') else str(t)}")
        return tools_list

"""
主管多智能体系统 - 基于 langgraph-supervisor 的命理分析系统

使用 create_supervisor 方法创建主管智能体，管理三个专业智能体：
- 八字智能体：基于生辰八字进行命理分析
- 手相智能体：基于手相图片进行分析
- 面相智能体：基于面部照片进行分析
"""
from typing import List, TypedDict, AsyncIterator, Dict, Any
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from prompt.supervisor_prompts import *
from services.service_manager import service_manager
from tools.code_tools import *
from tools.search_tools import *
from tools.time_tools import *
from utils import json_utils
from utils.custom_serializer import CustomSerializer
from utils.unified_logger import get_logger, log_error
from graph.base_graph import BaseGraph, BaseGraphState
import time
from datetime import datetime


class SupervisorState(BaseGraphState):
    """主管多智能体系统状态"""
    # 任务分析
    user_input: str
    analysis_type: str  # "bazi", "hand", "facial", "comprehensive"
    analysis_data: Dict[str, Any]  # 用户提供的分析数据

    # 智能体执行
    agent_results: Dict[str, Any]  # 各智能体的执行结果
    current_agent: str  # 当前执行的智能体


class SupervisorGraph(BaseGraph):
    """主管多智能体系统 - 使用 langgraph-supervisor 管理三个专业智能体"""

    # 类级别的图实例和检查点存储
    _shared_checkpointer = None
    _shared_graph = None
    _initialized = False

    def __init__(self):
        super().__init__("SupervisorGraph")

        # 使用共享的图实例和检查点存储
        if not SupervisorGraph._initialized:
            # 初始化检查点存储
            self._initialize_checkpointer(SupervisorGraph)
            
            # 构建图
            SupervisorGraph._shared_graph = self._build_graph()
            SupervisorGraph._initialized = True

        # 使用共享的图实例
        self.graph = SupervisorGraph._shared_graph
        self.checkpointer = SupervisorGraph._shared_checkpointer
        self.logger.info("SupervisorGraph 实例创建完成")

    def _build_graph(self) -> CompiledStateGraph:
        """构建主管多智能体图 - 使用 create_supervisor 方法"""

        # 创建三个专业智能体
        bazi_agent = self._create_bazi_agent()
        hand_agent = self._create_hand_agent()
        facial_agent = self._create_facial_agent()

        # 创建主管智能体，管理三个专业智能体
        supervisor_workflow = create_supervisor(
            agents=[bazi_agent, hand_agent, facial_agent],
            model=self.strategic_llm,
            prompt=supervisor_prompt.template,
            tools=[web_search],
            add_handoff_back_messages=True,
            supervisor_name="命理分析主管")

        return supervisor_workflow.compile(checkpointer=SupervisorGraph._shared_checkpointer)

    def _create_bazi_agent(self):
        """创建八字分析智能体"""
        return create_react_agent(
            model=self.fast_llm,
            tools=[web_search, tian_gan_di_zhi],
            name="八字专家",
            prompt=bazi_prompt.template,
        )

    def _create_hand_agent(self):
        """创建手相分析智能体"""
        return create_react_agent(
            model=self.vision_llm,
            tools=[web_search],
            name="手相专家",
            prompt=hand_prompt.template
        )

    def _create_facial_agent(self):
        """创建面相分析智能体"""
        return create_react_agent(
            model=self.vision_llm,
            tools=[web_search],
            name="面相专家",
            prompt=facial_prompt.template
        )

    async def chat_with_supervisor_stream(self, thread_id: str, messages: List = None) -> AsyncIterator[Dict[str, Any]]:
        """流式聊天接口 - 支持主管多智能体模式"""
        try:
            # 初始化状态
            self.thread_id = thread_id
            self.logger.info(f"开始主管多智能体分析，线程ID: {thread_id}")

            # 准备配置
            config = RunnableConfig(configurable={"thread_id": self.thread_id})

            # 转换消息格式为字典
            if messages and isinstance(messages, list):
                # 将消息对象转换为字典格式
                message_dicts = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        message_dicts.append({
                            "role": "user" if hasattr(msg, '__class__') and 'Human' in str(
                                msg.__class__) else "assistant",
                            "content": msg.content
                        })
                    else:
                        message_dicts.append(msg)

                input_data = {"messages": message_dicts}
            else:
                input_data = {"messages": messages or []}

            # 获取流式事件
            events = self.graph.astream_events(input_data, config=config)

            # 处理流式事件
            async for chunk in self.process_streaming_events(events):
                yield chunk

        except Exception as e:
            self.logger.error(f"主管多智能体执行失败: {str(e)}")
            yield {
                "step": "error",
                "message": f"执行失败: {str(e)}",
                "data": {"error": str(e)},
                "node": "supervisor_error"
            }
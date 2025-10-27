"""
主管多智能体系统 - 基于 langgraph-supervisor 的命理分析系统

使用 create_supervisor 方法创建主管智能体，管理三个专业智能体：
- 八字智能体：基于生辰八字进行命理分析
- 手相智能体：基于手相图片进行分析
- 面相智能体：基于面部照片进行分析
"""
from typing import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage

from graph.base_graph import BaseGraph
from prompt.supervisor_prompts import *
from tools.code_tools import *


class SupervisorGraph(BaseGraph):
    """主管多智能体系统 - 使用 langgraph-supervisor 管理三个专业智能体"""

    # 类级别的图实例和检查点存储
    _shared_checkpointer = None
    _shared_graph = None
    _initialized = False
    _bazi_info = None
    _hand_images = None
    _facial_images = None

    def __init__(self, analysis_data: Dict[str, Any] = None):
        super().__init__("SupervisorGraph")

        self._bazi_info = analysis_data.get("bazi")
        self._hand_images = analysis_data.get("hand_images")
        self._facial_images = analysis_data.get("facial_image")

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

        export_agents = []
        if self._bazi_info:
            bazi_agent = create_react_agent(
                model=self.fast_llm,
                tools=[],
                name="八字专家",
                prompt=bazi_prompt.template,
                pre_model_hook=self._append_bazi_info
            )
            export_agents.append(bazi_agent)

        if self._hand_images:
            hand_agent = create_react_agent(
                model=self.vision_llm,
                tools=[],
                name="手相专家",
                prompt=hand_prompt.template,
                pre_model_hook=self._append_hand_image
            )
            export_agents.append(hand_agent)

        if self._facial_images:
            facial_agent = create_react_agent(
                model=self.vision_llm,
                tools=[],
                name="面相专家",
                prompt=facial_prompt.template,
                pre_model_hook=self._append_facial_image
            )
            export_agents.append(facial_agent)

        # 创建主管智能体，管理三个专业智能体
        supervisor_workflow = create_supervisor(
            agents=export_agents,
            model=self.strategic_llm,
            prompt=supervisor_prompt.template,
            add_handoff_back_messages=False,
            supervisor_name="命理分析主管",
        )
        return supervisor_workflow.compile(checkpointer=SupervisorGraph._shared_checkpointer)

    async def chat_with_supervisor_stream(self, thread_id: str, user_msg: str) -> AsyncIterator[
        Dict[str, Any]]:
        """流式聊天接口 - 支持主管多智能体模式，处理包含图片数据的分析"""
        self.thread_id = thread_id
        config = RunnableConfig(configurable={"thread_id": self.thread_id})

        initial_input = {
            "messages": [HumanMessage(content=f"用户提供了{user_msg}，开始分析")]
        }
        events = self.graph.astream_events(initial_input, config=config)
        async for chunk in self.process_streaming_events(events):
            yield chunk

    async def process_streaming_events(self, events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[
        Dict[str, Any]]:
        """处理流式事件"""
        try:
            async for event in events:
                event_name = event.get('name', 'unknown')
                event_type = event['event']
                event_data = event.get('data', {})
                self.logger.info(f"收到event: {event_type} - {event_name}")

                # 处理链式流输出事件
                if event_type == "on_chain_stream":
                    if event_name == '八字专家':
                        chunk = event.get("data", {}).get("chunk", {})
                        if chunk:
                            if "chunk" in event_data and "messages" in event_data["chunk"]:
                                content = event_data["chunk"]["messages"][-1].content
                                if len(content) > 0:
                                    yield {
                                        "analysis": content,
                                    }
                            elif "output" in event_data:
                                content = event_data["output"].content
                                yield {
                                    "analysis": content,
                                }
                    elif event_name in ['手相专家', '面相专家']:
                        # 处理手相专家和面相专家的输出
                        chunk = event.get("data", {}).get("chunk", {})
                        if chunk:
                            if "chunk" in event_data and "messages" in event_data["chunk"]:
                                content = event_data["chunk"]["messages"][-1].content
                                # 处理文本片段格式
                                processed_content = self._process_text_fragments(content)
                                if processed_content:
                                    yield {
                                        "analysis": processed_content,
                                    }
                            elif "output" in event_data:
                                content = event_data["output"].content
                                # 处理文本片段格式
                                processed_content = self._process_text_fragments(content)
                                if processed_content:
                                    yield {
                                        "analysis": processed_content,
                                    }
        except Exception as e:
            self.logger.error(f"流式处理失败: {str(e)}")
            yield {
                "analysis": f"执行失败: {str(e)}",
            }

    def _process_text_fragments(self, content):
        """处理文本片段并整合为完整句子"""
        if isinstance(content, list):
            # 如果是列表格式，提取所有文本片段
            text_fragments = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_fragments.append(item['text'])
            
            if text_fragments:
                # 整合所有文本片段为完整句子
                full_text = ''.join(text_fragments)
                self.logger.info(f"文本片段整合: {len(text_fragments)}个片段 -> {full_text[:50]}...")
                return full_text
        elif isinstance(content, str):
            # 如果是字符串，直接返回
            return content
        
        return None

    def _append_bazi_info(self, state):
        """在模型调用前添加八字信息到消息中"""
        current_messages = state.get("messages", [])
        if self._bazi_info:
            message = HumanMessage(content=f"用户八字为{self._bazi_info}")
            current_messages.append(message)
            state["messages"] = current_messages


    def _append_hand_image(self, state):
        """在模型调用前添加手相图片到消息中"""
        current_messages = state.get("messages", [])
        if self._hand_images:
            # 处理多张手相照片（左手和右手）
            for hand_data in self._hand_images:
                hand_type = hand_data.get("type", "unknown")
                hand_image = hand_data.get("image", "")
                
                if hand_image:
                    # 添加图片消息
                    image_message = HumanMessage(
                        content=[
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{hand_image}"
                            }
                        ]
                    )
                    current_messages.append(image_message)
                    
                    # 添加手型说明
                    type_message = HumanMessage(content=f"这是用户的{hand_type}照片，请进行手相分析")
                    current_messages.append(type_message)
            
            state["messages"] = current_messages

    def _append_facial_image(self, state):
        """在模型调用前添加面相图片到消息中"""
        current_messages = state.get("messages", [])
        if self._facial_images:
            image_message = HumanMessage(
                content=[
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{self._facial_images}"
                    }
                ]
            )
            current_messages.append(image_message)
            state["messages"] = current_messages


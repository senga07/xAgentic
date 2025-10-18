"""
循环式规划执行器 - 每个节点先检查不确定性，确认后执行

执行流程：
1. 制定计划
2. 循环执行：检查节点不确定性 → 确认（如需要）→ 执行节点
3. 生成最终回复
"""
from typing import List, TypedDict, AsyncIterator

from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from prompt.plan_executor_prompts import *
from services.service_manager import service_manager
from tools.code_tools import *
from tools.search_tools import *
from tools.time_tools import *
from utils import json_utils
from utils.custom_serializer import CustomSerializer
from utils.unified_logger import get_logger, log_error
from typing import Dict, Any
import time
from datetime import datetime
from langmem import create_manage_memory_tool, create_search_memory_tool


class PlanExecutorState(TypedDict):
    """循环式规划执行状态"""
    # 消息和通信

    messages: List[BaseMessage]
    streaming_chunks: List[Dict[str, Any]]

    # 任务执行
    user_task: str
    task_analysis: str
    execution_plan: List[Dict[str, Any]]
    current_step: int
    step_results: List[Dict[str, Any]]

    # 状态管理
    status: str
    error: str

    # 上下文信息
    thread_id: str

    # 元数据
    timing_info: Dict[str, Any]


class PlanExecutorGraph:
    """循环式规划执行器 - 每个节点先检查不确定性，确认后执行"""

    # 类级别的图实例和检查点存储，确保所有实例共享相同的状态
    _shared_checkpointer = None
    _shared_graph = None
    _initialized = False

    def __init__(self):
        self.thread_id = None
        self.logger = get_logger(__name__)

        # 使用全局服务管理器中的配置和LLM实例
        self.config = service_manager.get_config()
        llms = service_manager.get_llms()
        self.plan_llm = llms.get('strategic_llm')
        self.worker_llm = llms.get('fast_llm')

        # 获取工具
        self.mcp_tools = service_manager.get_mcp_tools()
        self.local_tools = [
            web_search, execute_python_code, get_current_time,
            calculate_date_offset, get_time_info
        ]
        self.all_tools = self.local_tools + self.mcp_tools

        self.store = service_manager.store

        # 使用共享的图实例和检查点存储
        if not PlanExecutorGraph._initialized:
            # 初始化LangGraph检查点存储
            custom_serializer = CustomSerializer()
            PlanExecutorGraph._shared_checkpointer = MemorySaver(serde=custom_serializer)

            # 构建图
            PlanExecutorGraph._shared_graph = self._build_graph()
            PlanExecutorGraph._initialized = True
            self.logger.info("PlanExecutorGraph 首次初始化完成")

        # 使用共享的图实例
        self.graph = PlanExecutorGraph._shared_graph
        self.checkpointer = PlanExecutorGraph._shared_checkpointer
        self.logger.info("PlanExecutorGraph 实例创建完成")

    def _build_graph(self) -> CompiledStateGraph:
        """构建循环式执行图 - 每个节点执行前都检查不确定性"""
        workflow = StateGraph(PlanExecutorState)

        # 添加节点
        workflow.add_node("analyze_and_plan", self._analyze_and_plan)
        workflow.add_node("check_and_execute_node", self._check_and_execute_node)
        workflow.add_node("generate_response", self._generate_response)

        # 设置流程
        workflow.set_entry_point("analyze_and_plan")
        workflow.add_edge("analyze_and_plan", "check_and_execute_node")

        # 检查并执行节点后的条件边
        workflow.add_conditional_edges(
            "check_and_execute_node",
            self._after_check_and_execute,
            {
                "next_node": "check_and_execute_node",  # 继续检查下一个节点
                "complete": "generate_response"  # 所有节点执行完成
            }
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile(checkpointer=PlanExecutorGraph._shared_checkpointer)


    def _analyze_and_plan(self, state: PlanExecutorState) -> PlanExecutorState:
        """任务分析和计划创建节点"""
        try:
            self.logger.info("开始任务分析和计划创建")
            if "streaming_chunks" not in state:
                state["streaming_chunks"] = []

            start_time = time.time()
            user_input = state["messages"][0].content
            if not user_input:
                raise Exception("无法获取用户输入")
            state["user_task"] = user_input
            messages = [HumanMessage(content=planning_prompt.format(user_task=user_input))]
            response = self.plan_llm.invoke(messages)

            plan_data = json_utils.json_match(response.content)
            if not plan_data or not plan_data.get("execution_plan"):
                raise Exception("计划解析失败")

            state["task_analysis"] = plan_data.get("task_analysis", "")
            state["execution_plan"] = plan_data.get("execution_plan", [])
            state["current_step"] = 0

            total_duration = time.time() - start_time
            state["timing_info"] = {
                "plan_creation_duration": round(total_duration, 2),
                "plan_creation_timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"任务分析和计划创建完成，总耗时: {total_duration:.2f}秒")
            self._add_streaming_chunk(state, "plan", "📋 制定执行计划", {
                "task_analysis": state["task_analysis"],
                "execution_plan": state["execution_plan"]
            })
            return state
        except Exception as e:
            log_error(self.logger, e)
            state["error"] = str(e)
            state["status"] = "failed"
            return state


    def _check_and_execute_node(self, state: PlanExecutorState) -> PlanExecutorState:
        """执行当前节点"""

        execution_plan = state.get("execution_plan")
        current_step = state.get("current_step")

        if current_step >= len(execution_plan):
            self.logger.info("所有节点执行完成")
            state["status"] = "completed"
            return state

        current_node = execution_plan[current_step]
        self.check_node(current_node, current_step, execution_plan, state)

        node_result = self._do_execute(current_node, current_step)

        state["step_results"].append(node_result)
        state["current_step"] += 1

        return self.process_result(current_step + 1, node_result, state)

    def process_result(self, current_step, node_result, state):
        """处理执行结果"""
        if node_result.get('status') == 'failed':
            state["status"] = "failed"
            state["error"] = f"节点 {current_step} 执行失败: {node_result.get('execution_result', '')}"

            self._add_streaming_chunk(
                state,
                "execution_failed",
                f"❌ 节点 {current_step} 执行失败，停止执行",
                {"error": state["error"]}
            )
            return state

        timing = node_result.get("timing")
        self._add_streaming_chunk(
            state,
            "node_completed",
            f"✅ 节点 {current_step} 执行完成！",
            {
                "status": "completed",
                "result": node_result,
                "step_number": current_step,
                "execution_result": node_result.get("execution_result"),
                "timing": timing
            }
        )
        return state

    def check_node(self, current_node, current_step, execution_plan, state):
        """检查节点是否需要补充信息"""
        if current_node.get("requires_confirmation"):
            self.logger.info(f"节点 {current_step + 1} 需要用户确认")

            # 准备确认信息并中断
            confirmation_info = {
                "type": "confirmation_required",
                "current_step": current_step + 1,
                "total_steps": len(execution_plan),
                "step_info": {
                    "step": current_node.get("step"),
                    "description": current_node.get("description"),
                    "uncertainty_reason": current_node.get("uncertainty_reason"),
                    "expected_result": current_node.get("expected_result")
                }
            }
            # 中断等待用户输入
            current_node["user_feedback"] = interrupt(confirmation_info)

    def _after_check_and_execute(self, state: PlanExecutorState) -> str:
        """检查并执行节点后的条件判断"""
        status = state.get("status")
        if status == "completed":
            return "complete"  # 所有节点完成，转到生成回复
        elif status == "failed":
            return "complete"  # 失败也结束流程
        else:
            return "next_node"  # 继续执行下一个节点

    def _do_execute(self, node: Dict[str, Any], node_index: int) -> Dict[str, Any]:
        """执行单个节点"""
        start_time = time.time()

        memory_tools = [
            create_manage_memory_tool(namespace=("execute_memories",)),
            create_search_memory_tool(namespace=("execute_memories",)), ]

        try:
            # 创建ReAct Agent
            agent = create_react_agent(
                model=self.worker_llm,
                tools=self.all_tools + memory_tools,
                checkpointer=self.checkpointer,
                store=self.store
            )

            # 格式化工具列表
            tools_list = []
            for t in self.all_tools:
                if hasattr(t, 'name') and hasattr(t, 'description'):
                    tools_list.append(f"- {t.name}: {t.description}")
                else:
                    tools_list.append(f"- {t.__name__ if hasattr(t, '__name__') else str(t)}")

            prompt = react_prompt.format(
                description=node.get("description"),
                expected_result=node.get("expected_result"),
                user_feedback=node.get("user_feedback"),
                tools="\n".join(tools_list)
            )
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=RunnableConfig(configurable={"thread_id": self.thread_id})
                )
            except Exception as agent_error:
                self.logger.warning(f"ReAct Agent 执行失败，使用直接 LLM 调用: {str(agent_error)}")
                result = self.worker_llm.invoke([HumanMessage(content=prompt)])

            duration = time.time() - start_time
            return {
                "step": node_index + 1,
                "execution_result": self.get_execution_result(node, result),
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "timing": {
                    "duration": round(duration, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return {
                "step": node_index + 1,
                "execution_result": f"{str(e)}",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "timing": {
                    "duration": round(duration, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def get_execution_result(self, node, result):
        execution_result = ""
        try:
            if result and isinstance(result, dict):
                if "messages" in result:
                    messages = result["messages"]
                    if messages:
                        # 查找最后一个AI消息
                        for msg in reversed(messages):
                            if hasattr(msg, 'content'):
                                # 检查是否是AI消息
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
                execution_result = f"任务描述: {node.get('description', '')}\n执行状态: 已完成\n结果: {execution_result}"

        except Exception as e:
            self.logger.error(f"提取执行结果时出错: {str(e)}")
            execution_result = f"执行完成，但结果提取时出现错误: {str(e)}"
        return execution_result

    def _generate_response(self, state: PlanExecutorState) -> PlanExecutorState:
        """生成最终回复"""
        try:
            task_analysis = state.get("task_analysis")
            execution_plan = state.get("execution_plan")
            step_results = state.get("step_results")
            start_time = time.time()

            # 格式化执行计划
            plan_text = ""
            if execution_plan:
                for i, step in enumerate(execution_plan, 1):
                    plan_text += f"步骤{i}: {step.get('description')}\n"

            # 格式化执行结果
            results_text = ""
            if step_results:
                for i, result in enumerate(step_results, 1):
                    results_text += f"步骤{i}:{result.get('execution_result')}\n"
            try:
                # 构建总结提示
                summary_messages = [
                    HumanMessage(content=summary_response_prompt.format(
                        user_task=state.get("user_task"),
                        task_analysis=task_analysis,
                        execution_plan=plan_text,
                        step_results=results_text
                    ))
                ]

                response = self.plan_llm.invoke(summary_messages)
                final_result = response.content.strip()

                state["status"] = "completed"
            except Exception as e:
                self.logger.error(f"LLM总结生成失败: {e}")
                raise e

            duration = time.time() - start_time
            if "timing_info" not in state:
                state["timing_info"] = {}
            state["timing_info"]["response_generation_duration"] = round(duration, 2)
            state["timing_info"]["response_generation_timestamp"] = datetime.now().isoformat()

            # 计算总时间
            total_duration = (
                    state["timing_info"].get("plan_creation_duration", 0) +
                    state["timing_info"].get("response_generation_duration", 0)
            )

            state["streaming_chunks"].append({
                "step": "completed",
                "message": f"🎉 任务完成！总耗时: {total_duration:.2f}秒",
                "data": {
                    "response": final_result,
                    "timing_info": state["timing_info"],
                    "total_nodes": len(state.get("execution_plan", [])),
                    "completed_nodes": len(state.get("step_results", [])),
                    "step_results": state.get("step_results", []),
                    "execution_plan": state.get("execution_plan", [])
                }
            })

            self.logger.info(f"回复生成完成，耗时: {duration:.2f}秒")
            return state
        except Exception as e:
            log_error(self.logger, e, "回复生成失败")
            state["error"] = str(e)
            state["status"] = "failed"
            return state

    def _add_streaming_chunk(self, state: PlanExecutorState, step: str, message: str,
                             data: Dict[str, Any] = None) -> None:
        """添加流式输出块的辅助方法"""
        chunk = {
            "step": step,
            "message": message
        }
        if data:
            chunk["data"] = data
        state["streaming_chunks"].append(chunk)

    async def process_streaming_events(self, events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """处理流式事件的公共方法"""
        try:
            async for event in events:
                self.logger.info(f"收到event: {event['event']} - {event.get('name', 'unknown')}")
                if event["event"] == "on_chain_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if isinstance(chunk, dict) and "streaming_chunks" in chunk:
                        for streaming_chunk in chunk["streaming_chunks"]:
                            yield {
                                "step": streaming_chunk.get("step"),
                                "message": streaming_chunk.get("message"),
                                "data": streaming_chunk.get("data"),
                                "node": event.get("name", "unknown")
                            }
                    if isinstance(chunk, dict) and "__interrupt__" in chunk:
                        chunk_interrupt = chunk["__interrupt__"][0]
                        yield {
                            "step": "interrupt",
                            "message": "需要用户确认",
                            "data": chunk_interrupt.value,
                            "node": "check_node_uncertainty"
                        }
                        return
        except Exception as e:
            self.logger.error(f"流式处理失败: {str(e)}")
            yield {
                "step": "error",
                "message": f"执行失败: {str(e)}",
                "data": {},
                "node": "error"
            }

    async def chat_with_planning_stream(self, thread_id: str, messages: List = None) -> AsyncIterator[Dict[str, Any]]:
        """流式聊天接口 - 支持规划执行模式"""
        # 初始化状态
        self.thread_id = thread_id
        initial_state = {
            "messages": messages,
            "task_analysis": "",
            "execution_plan": [],
            "current_step": 0,
            "step_results": [],
            "status": "running",
            "error": "",
            "streaming_chunks": [],
            "timing_info": {},
        }

        config = RunnableConfig(configurable={"thread_id": self.thread_id})
        events = self.graph.astream_events(initial_state, config=config)

        async for chunk in self.process_streaming_events(events):
            yield chunk

    def get_current_state(self, config, current_state):
        """从检查点获取最新状态"""
        if not current_state:
            try:
                checkpoint_state = self.graph.get_state(config)
                if checkpoint_state and checkpoint_state.values:
                    return checkpoint_state.values
            except Exception as e:
                self.logger.warning(f"无法从检查点获取状态: {e}")
        return None

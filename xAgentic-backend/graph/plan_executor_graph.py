"""
循环式规划执行器 - 每个节点先检查不确定性，确认后执行

执行流程：
1. 制定计划
2. 循环执行：检查节点不确定性 → 确认（如需要）→ 执行节点
3. 生成最终回复
"""
from typing import List, TypedDict

from langchain_core.messages import AIMessage
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


class PlanExecutorState(TypedDict):
    """循环式规划执行状态"""
    # 消息和通信
    messages: List[Any]
    streaming_chunks: List[Dict[str, Any]]

    # 任务执行
    task_analysis: str
    execution_plan: List[Dict[str, Any]]
    current_step: int
    step_results: List[Dict[str, Any]]
    final_response: str

    # 状态管理
    status: str
    error: str

    # 上下文信息
    thread_id: str

    # 元数据
    timing_info: Dict[str, Any]
    
    # Human-in-the-loop 相关字段
    pending_confirmation: Dict[str, Any]  # 当前等待确认的步骤信息
    user_feedback: str  # 用户的反馈信息
    is_replanning: bool  # 是否正在重新规划



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
        workflow.add_node("check_node_uncertainty", self._check_node_uncertainty)
        workflow.add_node("execute_node", self._execute_node)
        workflow.add_node("generate_response", self._generate_response)

        # 设置流程
        workflow.set_entry_point("analyze_and_plan")
        workflow.add_edge("analyze_and_plan", "check_node_uncertainty")
        
        # 不确定性检查后的条件边
        workflow.add_conditional_edges(
            "check_node_uncertainty",
            self._after_uncertainty_check,
            {
                "execute_node": "execute_node",  # 执行当前节点
                "complete": "generate_response"  # 所有节点执行完成
            }
        )

        # 执行节点后的条件边
        workflow.add_conditional_edges(
            "execute_node",
            self._after_execute_node,
            {
                "next_node": "check_node_uncertainty",  # 检查下一个节点的不确定性
                "complete": "generate_response"  # 所有节点执行完成
            }
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile(checkpointer=PlanExecutorGraph._shared_checkpointer)

    # @trace_langsmith(name="analyze_and_plan", run_type="llm")
    def _analyze_and_plan(self, state: PlanExecutorState) -> PlanExecutorState:
        """任务分析和计划创建节点"""
        try:
            self.logger.info("开始任务分析和计划创建")
            if "streaming_chunks" not in state:
                state["streaming_chunks"] = []

            start_time = time.time()
            user_input = self._get_user_input(state)
            if not user_input:
                raise Exception("无法获取用户输入")
            messages = [HumanMessage(content=planning_prompt.format(user_task=user_input))]
            response = self.plan_llm.invoke(messages)

            plan_data = json_utils.json_match(response.content)
            if not plan_data or not plan_data.get("execution_plan"):
                raise Exception("计划解析失败")

            state["task_analysis"] = plan_data.get("task_analysis", "")
            state["execution_plan"] = plan_data.get("execution_plan", [])
            state["current_step"] = 0
            state["step_results"] = []
            state["status"] = "planned"

            # 添加流式输出
            self._add_streaming_chunk(state, "planning", "📋 制定执行计划", {
                "task_analysis": state["task_analysis"],
                "execution_plan": state["execution_plan"]
            })

            total_duration = time.time() - start_time
            state["timing_info"] = {"plan_creation": {
                "duration": round(total_duration, 2),
                "timestamp": datetime.now().isoformat()
            }}

            self.logger.info(f"任务分析和计划创建完成，总耗时: {total_duration:.2f}秒")
            return state
        except Exception as e:
            log_error(self.logger, e)
            state["error"] = str(e)
            state["status"] = "failed"
            return state

    # @trace_langsmith(name="check_node_uncertainty", run_type="chain")
    def _check_node_uncertainty(self, state: PlanExecutorState) -> PlanExecutorState:
        """检查当前节点是否需要用户确认"""

        self.logger.info("检查当前节点的不确定性")
        execution_plan = state.get("execution_plan")
        current_step = state.get("current_step")

        # 检查是否所有节点都已处理完成
        if current_step >= len(execution_plan):
            self.logger.info("所有节点执行完成")
            state["status"] = "all_nodes_completed"
            return state

        # 获取当前要检查的节点
        current_node = execution_plan[current_step]

        # 如果节点需要确认且没有用户反馈，则中断等待确认
        if current_node.get("requires_confirmation") and not state.get("user_feedback"):
            self.logger.info(f"节点 {current_step + 1} 需要用户确认")

            # 准备确认信息
            confirmation_info = {
                "type": "confirmation_required",
                "current_step": current_step + 1,
                "total_steps": len(execution_plan),
                "step_info": {
                    "step": current_node.get("step"),
                    "description": current_node.get("description"),
                    "uncertainty_reason": current_node.get("uncertainty_reason", ""),
                    "expected_result": current_node.get("expected_result")
                }
            }

            # 保存待确认信息到状态
            state["pending_confirmation"] = confirmation_info
            state["is_replanning"] = True

            # 添加流式输出
            self._add_streaming_chunk(
                state,
                "uncertainty_detected",
                f"⚠️ 步骤 {current_step + 1} 需要确认: {current_node.get('uncertainty_reason', '')}",
                confirmation_info
            )
            interrupt(confirmation_info)

        else:
            self.logger.info(f"节点 {current_step + 1} 无需确认，可以直接执行")
            state["pending_confirmation"] = {}

        return state

    def _after_uncertainty_check(self, state: PlanExecutorState) -> str:
        """不确定性检查后的条件判断"""
        if state.get("status") == "all_nodes_completed":
            return "complete"
        else:
            return "execute_node"


    # @trace_langsmith(name="execute_node", run_type="chain")
    def _execute_node(self, state: PlanExecutorState) -> PlanExecutorState:
        """执行当前节点"""

        start_time = time.time()
        execution_plan = state.get("execution_plan")
        current_step = state.get("current_step")

        # 检查是否所有节点都已处理完成
        if current_step >= len(execution_plan):
            self.logger.info("所有节点执行完成")
            state["status"] = "all_nodes_completed"
            return state

        # 获取当前要执行的节点
        current_node = execution_plan[current_step]
        current_node["user_feedback"] = state.get("user_feedback", "")

        # 添加流式输出
        self._add_streaming_chunk(
            state,
            "executing_node",
            f"🔄 正在执行节点 {current_step + 1}/{len(execution_plan)}: {current_node.get('description', '')}",
            {"node_index": current_step, "total_nodes": len(execution_plan)}
        )

        # 执行当前节点
        node_result = self._do_execute(current_node, current_step)

        # 更新状态
        state["step_results"].append(node_result)
        state["current_step"] += 1

        duration = time.time() - start_time
        if "timing_info" not in state:
            state["timing_info"] = {}
        state["timing_info"][f"node_{current_step + 1}"] = {
            "duration": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }

        # 添加完成状态
        self._add_streaming_chunk(
            state,
            "node_completed",
            f"✅ 节点 {current_step + 1} 执行完成 (耗时: {duration:.2f}秒)",
            {
                "status": node_result.get("status", "completed"),
                "result": node_result,
                "step_number": current_step + 1,
                "execution_result": node_result.get("execution_result", ""),
                "timing": node_result.get("timing", {})
            }
        )

        # 检查是否有异常
        if node_result.get('status') == 'failed':
            self.logger.error(f"节点 {current_step + 1} 执行失败，停止执行")
            state["status"] = "node_failed"
            state["error"] = f"节点 {current_step + 1} 执行失败: {node_result.get('execution_result', '')}"

            self._add_streaming_chunk(
                state,
                "execution_failed",
                f"❌ 节点 {current_step + 1} 执行失败，停止执行",
                {"error": state["error"], "failed_node": current_step + 1}
            )
            return state

        state["status"] = "node_completed"
        state["is_replanning"] = False
        state["pending_confirmation"] = {}
        self.logger.info(f"节点 {current_step + 1} 执行完成，耗时: {duration:.2f}秒")
        return state


    def _after_execute_node(self, state: PlanExecutorState) -> str:
        """执行节点后的条件判断"""
        if state.get("status") == "all_nodes_completed":
            return "complete"
        elif state.get("status") == "node_failed":
            return "complete"  # 失败也结束流程
        else:
            return "next_node"  # 继续检查下一个节点的不确定性


    def _do_execute(self, node: Dict[str, Any], node_index: int) -> Dict[str, Any]:
        """执行单个节点"""
        start_time = time.time()

        try:
            # 创建ReAct Agent
            agent = create_react_agent(
                model=self.worker_llm,
                tools=self.all_tools,
                checkpointer=self.checkpointer,
                store=self.store
            )

            # 格式化工具列表
            tools_list = []
            for tool in self.all_tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tools_list.append(f"- {tool.name}: {tool.description}")
                else:
                    tools_list.append(f"- {tool.__name__ if hasattr(tool, '__name__') else str(tool)}")
            
            prompt = react_prompt.format(
                description=node.get("description"),
                expected_result=node.get("expected_result"),
                user_feedback=node.get("user_feedback"),
                tools="\n".join(tools_list)
            )

            # 执行节点
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": self.thread_id},
                        "recursion_limit": 15,  # 增加递归限制
                        "max_execution_time": 60,  # 增加执行时间限制
                    }
                )
                self.logger.info(f"节点 {node_index + 1} 执行结果: {type(result)} - {str(result)[:200]}...")
            except Exception as agent_error:
                self.logger.warning(f"ReAct Agent 执行失败，使用直接 LLM 调用: {str(agent_error)}")
                # 回退到直接 LLM 调用
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=prompt)]
                result = self.worker_llm.invoke(messages)
                self.logger.info(f"直接 LLM 调用结果: {type(result)} - {str(result)[:200]}...")

            # 提取结果 - 改进返回值处理
            execution_result = "节点执行完成"
            
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

            duration = time.time() - start_time

            return {
                "step": node_index + 1,
                "execution_result": execution_result,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "timing": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": round(duration, 2)
                }
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            return {
                "step": node_index + 1,
                "execution_result": f"节点执行失败: {str(e)}",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error_details": str(e),
                "timing": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": round(duration, 2)
                }
            }

    def _generate_response(self, state: PlanExecutorState) -> PlanExecutorState:
        """生成最终回复"""
        try:
            start_time = time.time()
            self.logger.info("生成最终回复")

            # 生成回复
            if state["step_results"]:
                final_result = state["step_results"][-1].get("execution_result", "")
            else:
                final_result = state["task_analysis"]

            state["final_response"] = final_result
            state["status"] = "completed"

            # 添加到消息历史
            if "messages" in state:
                state["messages"].append(AIMessage(content=final_result))
            else:
                state["messages"] = [AIMessage(content=final_result)]

            duration = time.time() - start_time
            if "timing_info" not in state:
                state["timing_info"] = {}
            state["timing_info"]["response_generation"] = {
                "duration": round(duration, 2),
                "timestamp": datetime.now().isoformat()
            }

            # 计算总时间
            total_duration = sum(
                info.get("duration", 0) for info in state["timing_info"].values()
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

    def _get_user_input(self, state: PlanExecutorState):
        if "messages" in state and state["messages"]:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    return msg.content
        return None

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


    async def process_streaming_events(self, events, error_message_prefix="执行失败"):
        """处理流式事件的公共方法"""
        try:
            async for event in events:
                self.logger.info(f"收到event: {event['event']} - {event.get('name', 'unknown')}")
                if event["event"] == "on_chain_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if isinstance(chunk, dict) and "streaming_chunks" in chunk:
                        # 输出流式块
                        for streaming_chunk in chunk["streaming_chunks"]:
                            step_type = streaming_chunk.get("step", "")
                            yield {
                                "step": step_type,
                                "message": streaming_chunk.get("message", ""),
                                "data": streaming_chunk.get("data", {}),
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
                "message": f"{error_message_prefix}: {str(e)}",
                "data": {
                    "error": str(e),
                    "status": "failed"
                },
                "node": "error"
            }

    async def chat_with_planning_stream(self, thread_id: str, conversation_history: List = None, ):
        """流式聊天接口 - 支持规划执行模式"""
        # 初始化状态
        self.thread_id = thread_id
        initial_state = {
            "messages": conversation_history,
            "task_analysis": "",
            "execution_plan": [],
            "current_step": 0,
            "step_results": [],
            "final_response": "",
            "status": "initializing",
            "error": "",
            "streaming_chunks": [],
            "timing_info": {},
            "pending_confirmation": {},
            "user_feedback": {},
            "is_replanning": False
        }

        config = {"configurable": {"thread_id": self.thread_id}}
        events = self.graph.astream_events(initial_state, config=config, version="v1")
        
        async for chunk in self.process_streaming_events(events, error_message_prefix="执行失败"):
            yield chunk

    def get_current_state(self, config, current_state):

        if not current_state:
            try:
                # 从检查点获取最新状态
                checkpoint_state = self.graph.get_state(config)
                if checkpoint_state and checkpoint_state.values:
                    return checkpoint_state.values
            except Exception as e:
                self.logger.warning(f"无法从检查点获取状态: {e}")
        return None

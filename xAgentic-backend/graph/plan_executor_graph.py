"""
循环式规划执行器 - 每个节点先检查不确定性，确认后执行

执行流程：
1. 制定计划
2. 循环执行：检查节点不确定性 → 确认（如需要）→ 执行节点
3. 生成最终回复
"""
from typing import List, TypedDict, Literal, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command
import time
from datetime import datetime

from prompt.plan_executor_prompts import *
from services.service_manager import service_manager
from tools.code_tools import *
from tools.search_tools import *
from tools.time_tools import *
from utils import json_utils
from utils.custom_serializer import CustomSerializer
from utils.langsmith_utils import trace_langsmith
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
        """构建循环式执行图"""
        workflow = StateGraph(PlanExecutorState)

        # 添加节点
        workflow.add_node("analyze_and_plan", self._analyze_and_plan)
        workflow.add_node("execute_node", self._execute_node)
        workflow.add_node("generate_response", self._generate_response)

        # 设置流程
        workflow.set_entry_point("analyze_and_plan")
        workflow.add_edge("analyze_and_plan", "execute_node")

        # 执行节点后的条件边
        workflow.add_conditional_edges(
            "execute_node",
            self._after_execute_node,
            {
                "next_node": "execute_node",  # 继续下一个节点
                "complete": "generate_response"  # 所有节点执行完成
            }
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile(checkpointer=PlanExecutorGraph._shared_checkpointer)

    @trace_langsmith(name="analyze_and_plan", run_type="llm")
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



    @trace_langsmith(name="execute_node", run_type="chain")
    def _execute_node(self, state: PlanExecutorState) -> PlanExecutorState:
        """执行当前节点"""
        try:
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
                    "node_index": current_step, 
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
            self.logger.info(f"节点 {current_step + 1} 执行完成，耗时: {duration:.2f}秒")
            return state

        except Exception as e:
            log_error(self.logger, e, f"节点执行失败")
            state["error"] = str(e)
            state["status"] = "node_failed"
            return state


    def _after_execute_node(self, state: PlanExecutorState) -> str:
        """执行节点后的条件判断"""
        if state.get("status") == "all_nodes_completed":
            return "complete"
        elif state.get("status") == "node_failed":
            return "complete"  # 失败也结束流程
        else:
            return "next_node"  # 继续下一个节点


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

            prompt = react_prompt.format(
                description=node.get("description"),
                expected_result=node.get("expected_result"),
                tools="\n".join([f"- {t.name}: {t.description}" for t in self.all_tools])
            )

            # 执行节点
            result = agent.invoke({"messages":[{"role": "user", "content": prompt}]},
                                  config={
                                      "configurable": {"thread_id": self.thread_id},
                                      "recursion_limit": 10,
                                      "max_execution_time": 30,
                                  }
                                  )

            # 提取结果 - 修复返回值处理
            execution_result = "节点执行完成"
            if result and isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    # 查找最后一个AI消息
                    for msg in reversed(messages):
                        if hasattr(msg, 'content') and hasattr(msg, '__class__'):
                            if 'AI' in msg.__class__.__name__ or 'AIMessage' in str(type(msg)):
                                execution_result = msg.content
                                break
                    else:
                        # 如果没有找到AI消息，使用最后一个消息
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            execution_result = last_message.content
                        else:
                            execution_result = str(last_message)
            elif result and hasattr(result, 'content'):
                # 如果result直接是消息对象
                execution_result = result.content
            elif result:
                # 其他情况，转换为字符串
                execution_result = str(result)

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

            # 准备总结回复的数据
            user_task = ""
            if state.get("messages"):
                # 从消息历史中获取用户任务
                for msg in state["messages"]:
                    if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage':
                        user_task = msg.content
                        break
            
            task_analysis = state.get("task_analysis", "")
            execution_plan = state.get("execution_plan", [])
            step_results = state.get("step_results", [])

            # 格式化执行计划
            plan_text = ""
            if execution_plan:
                for i, step in enumerate(execution_plan, 1):
                    plan_text += f"步骤{i}: {step.get('description', '')}\n"
                    plan_text += f"预期结果: {step.get('expected_result', '')}\n\n"

            # 格式化执行结果
            results_text = ""
            if step_results:
                for i, result in enumerate(step_results, 1):
                    results_text += f"步骤{i}执行结果:\n"
                    results_text += f"状态: {result.get('status', 'unknown')}\n"
                    results_text += f"结果: {result.get('execution_result', '')}\n"
                    if result.get('error_details'):
                        results_text += f"错误详情: {result.get('error_details')}\n"
                    results_text += "\n"

            # 使用LLM生成总结回复
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                
                # 构建总结提示
                summary_messages = [
                    HumanMessage(content=summary_response_prompt.format(
                        user_task=user_task,
                        task_analysis=task_analysis,
                        execution_plan=plan_text,
                        step_results=results_text
                    ))
                ]
                
                # 使用战略LLM生成总结
                llm_start_time = time.time()
                response = self.plan_llm.invoke(summary_messages)
                llm_duration = time.time() - llm_start_time
                
                final_result = response.content.strip()
                self.logger.info(f"LLM总结生成耗时: {llm_duration:.2f}秒")
                state["final_response"] = final_result
                state["status"] = "completed"
            except Exception as e:
                self.logger.error(f"LLM总结生成失败: {e}")
                raise e

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

    async def chat_with_planning_stream(self, thread_id: str, conversation_history: List = None, ):
        """流式聊天接口 - 支持规划执行模式"""
        try:
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
                "timing_info": {}
            }

            # 开始执行图
            config = {"configurable": {"thread_id": self.thread_id}}

            # 流式执行图
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    self.logger.info(f"收到event: {event}")
                    
                    # 处理流式数据事件 - 这是主要的输出来源
                    if event["event"] == "on_chain_stream":
                        chunk = event.get("data", {}).get("chunk", {})
                        if isinstance(chunk, dict) and "streaming_chunks" in chunk:
                            # 输出流式块
                            for streaming_chunk in chunk["streaming_chunks"]:
                                step_type = streaming_chunk.get("step", "")
                                if step_type == "node_completed":
                                    yield {
                                        "type": "node_result",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": event.get("name", "unknown")
                                    }
                                elif step_type == "completed":
                                    yield {
                                        "type": "final_result",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": event.get("name", "unknown")
                                    }
                                else:
                                    yield {
                                        "type": "streaming",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": event.get("name", "unknown")
                                    }
                    
                    # 处理节点完成事件
                    elif event["event"] == "on_chain_end" and event.get("name") in ["analyze_and_plan", "execute_node", "generate_response"]:
                        node_name = event["name"]
                        node_output = event.get("output", {})
                        self.logger.info(f"处理节点 {node_name} 的输出: {type(node_output)} - {node_output}")
                        
                        if isinstance(node_output, dict) and "streaming_chunks" in node_output:
                            # 输出流式块
                            for streaming_chunk in node_output["streaming_chunks"]:
                                # 根据步骤类型设置不同的type
                                step_type = streaming_chunk.get("step", "")
                                if step_type == "node_completed":
                                    yield {
                                        "type": "node_result",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": node_name
                                    }
                                elif step_type == "completed":
                                    yield {
                                        "type": "final_result",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": node_name
                                    }
                                else:
                                    yield {
                                        "type": "streaming",
                                        "step": step_type,
                                        "message": streaming_chunk.get("message", ""),
                                        "data": streaming_chunk.get("data", {}),
                                        "node": node_name
                                    }

                        # 检查执行完成
                        if isinstance(node_output, dict) and node_output.get("status") == "completed":
                            yield {
                                "type": "completed",
                                "message": "任务执行完成",
                                "data": {
                                    "final_response": node_output.get("final_response", ""),
                                    "step_results": node_output.get("step_results", []),
                                    "timing_info": node_output.get("timing_info", {})
                                },
                                "node": node_name
                            }
                            return

                        # 检查执行失败
                        if isinstance(node_output, dict) and node_output.get("status") == "failed":
                            yield {
                                "type": "error",
                                "message": f"执行失败: {node_output.get('error', '未知错误')}",
                                "data": {
                                    "error": node_output.get("error", ""),
                                    "status": node_output.get("status", "")
                                },
                                "node": node_name
                            }
                            return
                            
            except Exception as e:
                # 处理异常
                self.logger.error(f"流式执行异常: {str(e)}")
                yield {
                    "type": "error",
                    "message": f"执行异常: {str(e)}",
                    "data": {
                        "error": str(e),
                        "status": "failed"
                    },
                    "node": "error"
                }
                return

            # 如果没有明确的完成或失败状态，输出最终结果
            yield {
                "type": "completed",
                "message": "任务执行完成",
                "data": {
                    "final_response": "任务已执行完成",
                    "status": "completed"
                },
                "node": "final"
            }

        except Exception as e:
            self.logger.error(f"流式聊天执行失败: {str(e)}")
            yield {
                "type": "error",
                "message": f"执行失败: {str(e)}",
                "data": {
                    "error": str(e),
                    "status": "failed"
                },
                "node": "error"
            }


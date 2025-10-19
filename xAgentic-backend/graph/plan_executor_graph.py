"""
循环式规划执行器 - 每个节点先检查不确定性，确认后执行

执行流程：
1. 制定计划
2. 循环执行：检查节点不确定性 → 确认（如需要）→ 执行节点
3. 生成最终回复
"""
import time
from datetime import datetime
from typing import List, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from langmem import create_manage_memory_tool, create_search_memory_tool

from graph.base_graph import BaseGraph, BaseGraphState
from prompt.plan_executor_prompts import *
from tools.code_tools import *
from utils import json_utils
from utils.unified_logger import log_error


class PlanExecutorState(BaseGraphState):
    """循环式规划执行状态"""
    # 任务执行
    user_task: str
    task_analysis: str
    execution_plan: List[Dict[str, Any]]
    current_step: int
    step_results: List[Dict[str, Any]]


class PlanExecutorGraph(BaseGraph):
    """循环式规划执行器 - 每个节点先检查不确定性，确认后执行"""

    # 类级别的图实例和检查点存储，确保所有实例共享相同的状态
    _shared_checkpointer = None
    _shared_graph = None
    _initialized = False

    def __init__(self):
        super().__init__("PlanExecutorGraph")
        
        # 使用共享的图实例和检查点存储
        if not PlanExecutorGraph._initialized:
            # 初始化检查点存储
            self._initialize_checkpointer(PlanExecutorGraph)
            
            # 构建图
            PlanExecutorGraph._shared_graph = self._build_graph()
            PlanExecutorGraph._initialized = True

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
            response = self.strategic_llm.invoke(messages)

            plan_data = json_utils.json_match(response.content)
            if not plan_data or not plan_data.get("execution_plan"):
                raise Exception("计划解析失败")

            state["task_analysis"] = plan_data.get("task_analysis", "")
            state["execution_plan"] = plan_data.get("execution_plan", [])
            state["current_step"] = 0

            state["timing_info"] = self._get_timing_info(start_time, "plan_creation")

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
        self.check_node(current_node, current_step, execution_plan)

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

    def check_node(self, current_node, current_step, execution_plan):
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
                model=self.fast_llm,
                tools=self.all_tools + memory_tools,
                checkpointer=self.checkpointer,
                store=self.store
            )

            # 格式化工具列表
            tools_list = self._format_tools_list(self.all_tools)

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
                result = self.fast_llm.invoke([HumanMessage(content=prompt)])

            timing_info = self._get_timing_info(start_time, "node_execution")
            return {
                "step": node_index + 1,
                "execution_result": self._extract_execution_result(result, node.get("description")),
                "status": "completed",
                "timing": timing_info
            }
        except Exception as e:
            timing_info = self._get_timing_info(start_time, "node_execution")
            return {
                "step": node_index + 1,
                "execution_result": f"{str(e)}",
                "status": "failed",
                "timing": timing_info
            }

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

                response = self.strategic_llm.invoke(summary_messages)
                final_result = response.content.strip()

                state["status"] = "completed"
            except Exception as e:
                self.logger.error(f"LLM总结生成失败: {e}")
                raise e

            if "timing_info" not in state:
                state["timing_info"] = {}
            response_timing = self._get_timing_info(start_time, "response_generation")
            state["timing_info"].update(response_timing)

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
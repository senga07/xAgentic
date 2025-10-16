import json
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[str]] = None
    mcp_configs: Optional[List[Dict[str, Any]]] = None  # MCP服务器配置

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[str]
    planning_details: Optional[dict] = None  # 规划详情（仅在planning模式下返回）

class FeedbackRequest(BaseModel):
    thread_id: str
    action: str  # "continue", "cancel", "replan"
    feedback: Optional[str] = None  # 用户补充信息



# 临时作为thread_id
user_id = "user01"

# 全局图实例管理器
_graph_instances = {}

@router.post("/stream")
async def chat_with_planning_stream(request: ChatRequest):
    """使用plan-executor模式的流式聊天请求"""
    try:
        # 转换对话历史为消息对象
        from langchain_core.messages import HumanMessage, AIMessage
        conversation_history = []
        if request.conversation_history:
            for i, msg in enumerate(request.conversation_history):
                if i % 2 == 0:  # 用户消息
                    conversation_history.append(HumanMessage(content=msg))
                else:  # AI消息
                    conversation_history.append(AIMessage(content=msg))
        else:
            conversation_history.append(HumanMessage(content=request.message))

        # 使用plan-executor模式
        from graph.plan_executor_graph import PlanExecutorGraph
        plan_executor_graph = PlanExecutorGraph()
        plan_executor_graph.thread_id = user_id
        
        # 保存图实例到全局管理器
        _graph_instances[user_id] = plan_executor_graph
        
        async def generate_stream():
            try:
                async for chunk in plan_executor_graph.chat_with_planning_stream(user_id, conversation_history):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_chunk = {
                    "step": "error",
                    "message": f"流式处理失败: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式规划聊天处理失败: {str(e)}")


@router.post("/feedback-stream")
async def handle_user_feedback_stream(request: FeedbackRequest):
    """处理用户反馈并恢复流式执行"""
    try:
        from graph.plan_executor_graph import PlanExecutorGraph
        
        # 尝试获取已存在的图实例
        plan_executor_graph = _graph_instances.get(request.thread_id)
        
        if not plan_executor_graph:
            raise HTTPException(status_code=404, detail="未找到对应的图实例")
        
        # 更新图状态
        config = {"configurable": {"thread_id": request.thread_id}}
        plan_executor_graph.graph.update_state(
            config=config,
            values={"user_feedback": request.feedback, "is_replanning" : False}
        )
        
        # 恢复流式执行
        async def generate_continuation_stream():
            try:
                events = plan_executor_graph.graph.astream_events(None, config=config, version="v1")
                async for chunk in plan_executor_graph.process_streaming_events(
                    events, error_message_prefix="恢复执行失败"
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_chunk = {
                    "step": "error",
                    "message": f"恢复执行失败: {str(e)}",
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_continuation_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理用户反馈失败: {str(e)}")

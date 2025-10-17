import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel
from graph.plan_executor_graph import PlanExecutorGraph

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    
    class Config:
        extra = "forbid"  # 禁止额外字段

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[str]
    planning_details: Optional[dict] = None  # 规划详情（仅在planning模式下返回）

class FeedbackRequest(BaseModel):
    action: str  # "continue", "cancel", "replan"
    feedback: Optional[str] = None  # 用户补充信息



# 临时作为thread_id
user_id = "user01"

# 全局图实例管理器
_graph_instances:dict[str, PlanExecutorGraph] = {}


def return_response(func):
    return StreamingResponse(
        func,
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.post("/stream")
async def chat_with_planning_stream(request: ChatRequest):
    """使用plan-executor模式的流式聊天请求"""
    try:
        plan_executor_graph = PlanExecutorGraph()
        plan_executor_graph.thread_id = user_id

        _graph_instances[user_id] = plan_executor_graph
        
        async def generate_stream():
            """流式输出"""
            try:
                async for chunk in plan_executor_graph.chat_with_planning_stream(
                        user_id, [HumanMessage(content=request.message)]):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_chunk = {
                    "step": "error",
                    "message": f"流式处理失败: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        return return_response(generate_stream())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式规划聊天处理失败: {str(e)}")


@router.post("/feedback-stream")
async def handle_user_feedback_stream(request: FeedbackRequest):
    """处理用户反馈并恢复流式执行"""
    try:
        plan_executor_graph = _graph_instances.get(user_id)
        if not plan_executor_graph:
            raise HTTPException(status_code=404, detail="未找到对应的图实例")

        async def generate_continuation_stream():
            """恢复流式执行"""
            try:
                events = plan_executor_graph.graph.astream_events(
                    Command(resume=request.feedback),
                    config={"configurable": {"thread_id": user_id}})
                async for chunk in plan_executor_graph.process_streaming_events(events):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_chunk = {
                    "step": "error",
                    "message": f"恢复执行失败: {str(e)}",
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        return return_response(generate_continuation_stream())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理用户反馈失败: {str(e)}")

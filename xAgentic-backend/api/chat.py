from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from mcp_.manager import mcp_manager
from services.service_manager import service_manager


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[str]] = None
    mcp_configs: Optional[List[Dict[str, Any]]] = None  # MCP服务器配置

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[str]
    planning_details: Optional[dict] = None  # 规划详情（仅在planning模式下返回）



# 临时作为thread_id
user_id = "user01"

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
        
        async def generate_stream():
            try:
                async for chunk in plan_executor_graph.chat_with_planning_stream(user_id, conversation_history):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_chunk = {
                    "type": "error",
                    "message": f"流式处理失败: {str(e)}",
                    "step": "error"
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




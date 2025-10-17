from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcp_.manager import mcp_manager

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

class MCPConfigRequest(BaseModel):
    configs: Dict[str, Dict[str, Any]]

@router.get("/configs")
async def get_mcp_configs():
    """获取MCP配置列表"""
    try:
        configs = mcp_manager.load_config()
        return {"configs": configs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取MCP配置失败: {str(e)}")

@router.post("/configs")
async def save_mcp_configs(request: MCPConfigRequest):
    """保存MCP配置"""
    try:
        success = mcp_manager.save_config(request.configs)
        if success:
            return {"message": "MCP配置保存成功"}
        else:
            raise HTTPException(status_code=500, detail="保存MCP配置失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存MCP配置失败: {str(e)}")
"""
面相手相分析系统后端接口
提供文件上传和相学分析功能
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# 创建路由器
router = APIRouter(prefix="/api/fortune", tags=["fortune"])


# 允许的图片格式
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

class FortuneAnalysisRequest(BaseModel):
    """相学分析请求模型"""
    birthDateTime: str
    palmPhoto: str  # 文件路径
    facePhoto: str  # 文件路径

class FortuneAnalysisResponse(BaseModel):
    """相学分析响应模型"""
    success: bool
    message: str
    analysis: Optional[str] = None
    recommendations: Optional[str] = None
    timestamp: str

def validate_file(file: UploadFile) -> bool:
    """验证上传文件"""
    if not file:
        return False
    
    # 检查文件大小
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        return False
    
    # 检查文件扩展名
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False
    
    return True

def read_uploaded_file_stream(file: UploadFile) -> bytes:
    """读取上传文件的流数据"""
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="无效的文件格式或大小")
    
    # 读取文件内容
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制")
    
    return content



@router.post("/analyze", response_model=FortuneAnalysisResponse)
async def analyze_fortune(
    birthDateTime: str = Form(None),
    leftHandPhoto: UploadFile = File(None),
    rightHandPhoto: UploadFile = File(None),
    facePhoto: UploadFile = File(None)
):
    """
    面相手相分析接口
    
    参数:
    - birthDateTime: 出生日期时间 (YYYY-MM-DD HH:MM) - 可选
    - leftHandPhoto: 左手照片文件 - 可选
    - rightHandPhoto: 右手照片文件 - 可选
    - facePhoto: 面相照片文件 - 可选
    
    返回:
    - 相学分析结果
    """
    try:
        user_data = {}
        user_msg = []
        
        # 处理出生日期（如果提供）
        if birthDateTime:
            birth_dt = datetime.strptime(birthDateTime, "%Y-%m-%d %H:%M")
            if birth_dt:
                from tools.code_tools import tian_gan_di_zhi
                tian, gan, di, zhi = tian_gan_di_zhi(int(birth_dt.year),
                                       int(birth_dt.month),
                                       int(birth_dt.day),
                                       int(birth_dt.hour))
                tmp = f"年柱{tian}，月柱{gan}，日柱{di}，时柱{zhi}"
                user_data.update({"bazi": tmp})
                user_msg.append(f"八字信息:{tmp}")

        import base64
        hand_images = []
        
        # 处理左手照片
        if leftHandPhoto and leftHandPhoto.filename:
            left_hand_image_data = read_uploaded_file_stream(leftHandPhoto)
            left_hand_base64 = base64.b64encode(left_hand_image_data).decode('utf-8')
            hand_images.append({"type": "left", "image": left_hand_base64})
            user_msg.append("左手照片")

        # 处理右手照片
        if rightHandPhoto and rightHandPhoto.filename:
            right_hand_image_data = read_uploaded_file_stream(rightHandPhoto)
            right_hand_base64 = base64.b64encode(right_hand_image_data).decode('utf-8')
            hand_images.append({"type": "right", "image": right_hand_base64})
            user_msg.append("右手照片")

        # 如果有手相照片，添加到用户数据中
        if hand_images:
            user_data.update({"hand_images": hand_images})
            logging.info(f"添加手相照片到用户数据: {len(hand_images)}张照片")

        if facePhoto and facePhoto.filename:
            facial_image_data = read_uploaded_file_stream(facePhoto)
            user_data.update({"facial_image": base64.b64encode(facial_image_data).decode('utf-8')})
            user_msg.append("面相图片")

        # 验证至少有一个输入
        if not user_data:
            raise HTTPException(status_code=400, detail="请至少提供出生日期、手相照片或面相照片中的一项")

        # 调用supervisor_graph进行分析
        from graph.supervisor_graph import SupervisorGraph
        supervisor = SupervisorGraph(user_data)
        
        # 执行分析
        analysis_parts = []
        try:
            import uuid
            uuid4 = uuid.uuid4()
            async for chunk in supervisor.chat_with_supervisor_stream(str(uuid4), "、".join(user_msg)):
                chunk_analysis = chunk.get("analysis", "")
                if chunk_analysis:
                    analysis_parts.append(chunk_analysis)
            
            # 合并所有分析内容
            analysis = "\n\n".join(analysis_parts) if analysis_parts else "分析完成"
        except Exception as e:
            logging.error(f"分析过程中发生错误: {str(e)}")
            analysis = f"分析过程中发生错误: {str(e)}"

        return FortuneAnalysisResponse(
            success=True,
            message="分析完成",
            analysis=analysis,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析过程中发生错误: {str(e)}")


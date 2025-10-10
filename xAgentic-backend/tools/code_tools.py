"""
代码执行工具模块

使用langchain-sandbox实现安全的代码执行功能
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any
from langchain_sandbox import SyncPyodideSandbox
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from prompt.code_prompts import *


# 全局沙箱实例
_sandbox = None

def get_sandbox():
    """获取沙箱实例（单例模式）"""
    global _sandbox
    if _sandbox is None:
        _sandbox = SyncPyodideSandbox(allow_net=True)
    return _sandbox


def generate_code(task_description: str, context: str = "") -> str:
    """
    根据任务描述生成Python代码
    
    Args:
        task_description: 任务描述
        context: 上下文信息
        
    Returns:
        生成的Python代码
    """
    start_time = time.time()
    try:
        # 使用服务管理器避免循环依赖
        from services.service_manager import service_manager
        code_llm = service_manager.get_llms()['code_llm']
        
        messages = [
            SystemMessage(content=system_prompt.template),
            HumanMessage(content=user_prompt.format(
                task_description=task_description,
                context=context)),
        ]
        
        llm_start_time = time.time()
        response = code_llm.invoke(messages)
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time
        logging.info(f"代码生成LLM调用耗时: {llm_duration:.2f}秒")
        code = response.content.strip()
        
        # 清理代码，移除可能的markdown标记
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        # 记录时间统计
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"代码生成耗时: {duration:.2f}秒")
        
        # 使用时间统计日志记录器
        # 时间统计信息已通过标准日志记录
        
        return code.strip()
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"代码生成失败: {e}, 耗时: {duration:.2f}秒")
        return f"# 代码生成失败: {str(e)}"


def execute_code(code: str) -> Dict[str, Any]:
    """
    执行Python代码
    
    Args:
        code: 要执行的Python代码
        
    Returns:
        执行结果字典
    """
    start_time = time.time()
    try:
        sandbox = get_sandbox()
        result = sandbox.execute(code)
        
        # 记录时间统计
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"代码执行耗时: {duration:.2f}秒")
        
        # 使用时间统计日志记录器
        # 时间统计信息已通过标准日志记录
        
        return {
            "status": "success",
            "result": result,
            "code": code,
            "error": None,
            "timing": {
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration": round(duration, 2)
            }
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"代码执行失败: {e}, 耗时: {duration:.2f}秒")
        return {
            "status": "error",
            "result": None,
            "code": code,
            "error": str(e),
            "timing": {
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration": round(duration, 2)
            }
        }


@tool
def execute_python_code(task_description: str, context: str = "") -> str:
    """
    根据任务描述生成并执行Python代码
    
    Args:
        task_description: 任务描述，说明需要完成什么任务
        context: 上下文信息，提供额外的背景信息
        
    Returns:
        执行结果的字符串表示
    """
    start_time = time.time()
    try:
        # 生成代码
        code = generate_code(task_description, context)
        
        # 执行代码
        execution_result = execute_code(code)
        
        # 记录总时间统计
        end_time = time.time()
        total_duration = end_time - start_time
        logging.info(f"Python代码工具总耗时: {total_duration:.2f}秒")
        
        if execution_result["status"] == "success":
            return f"代码执行成功！\n\n生成的代码：\n```python\n{code}\n```\n\n执行结果：\n{execution_result['result']}\n\n⏱️ 总耗时: {total_duration:.2f}秒"
        else:
            return f"代码执行失败：\n\n生成的代码：\n```python\n{code}\n```\n\n错误信息：\n{execution_result['error']}\n\n⏱️ 总耗时: {total_duration:.2f}秒"
            
    except Exception as e:
        end_time = time.time()
        total_duration = end_time - start_time
        logging.error(f"代码执行工具调用失败：{str(e)}, 耗时: {total_duration:.2f}秒")
        return f"代码执行工具调用失败：{str(e)}\n\n⏱️ 总耗时: {total_duration:.2f}秒"
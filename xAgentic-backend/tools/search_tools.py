"""
搜索相关工具

提供联网搜索功能，使用 Tavily API 进行实时网络搜索
"""
import logging
import time
from datetime import datetime
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from cfg.setting import get_settings


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    使用 Tavily 进行联网搜索，获取最新的网络信息
    
    Args:
        query (str): 搜索查询字符串
        max_results (int): 最大返回结果数量，默认为3
        
    Returns:
        str: 搜索结果摘要，包含相关网页的标题、内容和链接
    """
    start_time = time.time()
    try:
        settings = get_settings()
        
        # 创建 Tavily 搜索工具实例
        search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=max_results
        )
        
        # 执行搜索
        results = search.run(query)
        
        # 记录时间统计
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"网络搜索耗时: {duration:.2f}秒")
        
        # 使用时间统计日志记录器
        # 时间统计信息已通过标准日志记录
        
        if not results:
            return f"未找到与 '{query}' 相关的搜索结果。\n\n⏱️ 搜索耗时: {duration:.2f}秒"
        
        # 格式化搜索结果
        formatted_results = []
        results = results.get('results')
        for i, result in enumerate(results, 0):
            title = result.get('title', '无标题')
            content = result.get('content', '无内容')
            url = result.get('url', '无链接')
            
            formatted_result = f"""
结果 {i}:
标题: {title}
内容: {content}
链接: {url}
"""
            formatted_results.append(formatted_result.strip())
        
        formatted_output = "\n\n".join(formatted_results)
        return f"{formatted_output}\n\n⏱️ 搜索耗时: {duration:.2f}秒"
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"搜索失败：{str(e)}, 耗时: {duration:.2f}秒")
        return f"搜索失败：{str(e)}。请检查 Tavily API 密钥是否正确配置。\n\n⏱️ 搜索耗时: {duration:.2f}秒"


# 导出所有工具
__all__ = ["web_search"]

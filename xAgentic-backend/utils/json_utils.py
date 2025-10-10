import json
import logging
import re


def json_match(content: str):
    """更健壮的JSON解析函数"""
    if not content:
        return {}
    
    try:
        # 首先尝试直接解析整个内容
        result_data = json.loads(content)
        return result_data
    except json.JSONDecodeError:
        pass
    
    try:
        # 尝试查找JSON块
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
            return result_data
    except json.JSONDecodeError:
        pass
    
    try:
        # 尝试查找多个JSON块，取第一个
        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        for match in json_matches:
            try:
                result_data = json.loads(match)
                return result_data
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    try:
        # 尝试更宽松的JSON匹配，包括嵌套结构
        json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        for match in json_matches:
            try:
                result_data = json.loads(match)
                return result_data
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    logging.error(f"JSON解析失败，原始内容: {content[:200]}...")
    return {}
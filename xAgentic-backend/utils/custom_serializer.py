"""
自定义序列化器 - 解决 ToolMessage 序列化问题

支持 LangGraph 检查点功能，同时处理 ToolMessage 等特殊类型的序列化
"""
import pickle
import json
from typing import Any, Tuple, Dict
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.serde.base import SerializerProtocol


class CustomSerializer(SerializerProtocol):
    """自定义序列化器，专门处理 LangChain 消息类型"""
    
    def dumps(self, obj: Any) -> bytes:
        """序列化对象"""
        if isinstance(obj, ToolMessage):
            # 将 ToolMessage 转换为可序列化的字典
            obj_dict = {
                'type': 'ToolMessage',
                'content': obj.content,
                'tool_call_id': getattr(obj, 'tool_call_id', None),
                'name': getattr(obj, 'name', None),
                'additional_kwargs': getattr(obj, 'additional_kwargs', {}),
                'id': getattr(obj, 'id', None)
            }
            return pickle.dumps(obj_dict)
        elif isinstance(obj, (AIMessage, HumanMessage, SystemMessage)):
            # 处理其他消息类型
            obj_dict = {
                'type': obj.__class__.__name__,
                'content': obj.content,
                'additional_kwargs': getattr(obj, 'additional_kwargs', {}),
                'id': getattr(obj, 'id', None)
            }
            return pickle.dumps(obj_dict)
        else:
            # 其他类型使用 pickle 序列化
            return pickle.dumps(obj)
    
    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        """带类型的序列化"""
        if isinstance(obj, ToolMessage):
            obj_dict = {
                'type': 'ToolMessage',
                'content': obj.content,
                'tool_call_id': getattr(obj, 'tool_call_id', None),
                'name': getattr(obj, 'name', None),
                'additional_kwargs': getattr(obj, 'additional_kwargs', {}),
                'id': getattr(obj, 'id', None)
            }
            return "ToolMessage", pickle.dumps(obj_dict)
        elif isinstance(obj, (AIMessage, HumanMessage, SystemMessage)):
            obj_dict = {
                'type': obj.__class__.__name__,
                'content': obj.content,
                'additional_kwargs': getattr(obj, 'additional_kwargs', {}),
                'id': getattr(obj, 'id', None)
            }
            return obj.__class__.__name__, pickle.dumps(obj_dict)
        else:
            return "pickle", pickle.dumps(obj)
    
    def loads(self, data: bytes) -> Any:
        """反序列化对象"""
        try:
            obj = pickle.loads(data)
            if isinstance(obj, dict) and 'type' in obj:
                return self._reconstruct_message(obj)
            return obj
        except Exception as e:
            # 如果反序列化失败，返回原始数据
            return data
    
    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        """带类型的反序列化"""
        type_str, bytes_data = data
        try:
            obj = pickle.loads(bytes_data)
            if isinstance(obj, dict) and 'type' in obj:
                return self._reconstruct_message(obj)
            return obj
        except Exception as e:
            # 如果反序列化失败，返回原始数据
            return bytes_data
    
    def _reconstruct_message(self, obj_dict: Dict[str, Any]) -> Any:
        """重建消息对象"""
        message_type = obj_dict.get('type')
        
        if message_type == 'ToolMessage':
            return ToolMessage(
                content=obj_dict.get('content', ''),
                tool_call_id=obj_dict.get('tool_call_id'),
                name=obj_dict.get('name'),
                additional_kwargs=obj_dict.get('additional_kwargs', {}),
                id=obj_dict.get('id')
            )
        elif message_type == 'AIMessage':
            return AIMessage(
                content=obj_dict.get('content', ''),
                additional_kwargs=obj_dict.get('additional_kwargs', {}),
                id=obj_dict.get('id')
            )
        elif message_type == 'HumanMessage':
            return HumanMessage(
                content=obj_dict.get('content', ''),
                additional_kwargs=obj_dict.get('additional_kwargs', {}),
                id=obj_dict.get('id')
            )
        elif message_type == 'SystemMessage':
            return SystemMessage(
                content=obj_dict.get('content', ''),
                additional_kwargs=obj_dict.get('additional_kwargs', {}),
                id=obj_dict.get('id')
            )
        else:
            return obj_dict

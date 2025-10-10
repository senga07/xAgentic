import json
import os
from typing import List, Dict, Any


class MCPConfigManager:
    """MCP配置管理器 - 单例模式"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config_file = "mcp_config.json"
            self._initialized = True

    def load_config(self) -> List[Dict[str, Any]]:
        """从mcp_config.json文件加载MCP配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('mcp_server', [])
            return []
        except Exception as e:
            print(f"加载MCP配置失败: {e}")
            return []

    def save_config(self, configs: List[Dict[str, Any]]) -> bool:
        """保存MCP配置到JSON文件"""
        try:
            data = {'mcp_server': configs}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存MCP配置失败: {e}")
            return False


# 全局MCP配置管理器实例
mcp_manager = MCPConfigManager()

"""
MCP Client Management Module

Handles MCP client creation, configuration conversion, and connection management.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages MCP client lifecycle and configuration.
    
    Responsible for:
    - Converting xAgentic MCP configs to langchain format
    - Creating and managing MultiServerMCPClient instances
    - Handling client cleanup and resource management
    """

    def __init__(self, mcp_configs: List[Dict[str, Any]]):
        """
        Initialize the MCP client manager.
        
        Args:
            mcp_configs: List of MCP server configurations from xAgentic
        """
        self.mcp_configs = mcp_configs or []
        self._client = None
        self._client_lock = asyncio.Lock()

    def convert_configs_to_langchain_format(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert xAgentic MCP configs to langchain-mcp-adapters format.
        
        Returns:
            Dict[str, Dict[str, Any]]: Server configurations for MultiServerMCPClient
        """
        server_configs = {}
        
        for i, config in enumerate(self.mcp_configs):
            # Generate server name
            server_name = config.get("name", f"mcp_server_{i+1}")
            config = config.get("config")
            # Build the server config
            server_config = {}
            
            # Auto-detect transport type from URL if provided
            connection_url = config.get("connection_url")
            if connection_url:
                if connection_url.startswith(("wss://", "ws://")):
                    server_config["transport"] = "websocket"
                    server_config["url"] = connection_url
                elif connection_url.startswith(("https://", "http://")):
                    server_config["transport"] = "streamable_http"
                    server_config["url"] = connection_url
                else:
                    # Fallback to specified connection_type or stdio
                    connection_type = config.get("connection_type", "stdio")
                    server_config["transport"] = connection_type
                    if connection_type in ["websocket", "streamable_http", "http"]:
                        server_config["url"] = connection_url
            else:
                # No URL provided, use stdio (default) or specified connection_type
                connection_type = config.get("connection_type", "stdio")
                server_config["transport"] = connection_type
            
            # Handle stdio transport configuration
            if server_config.get("transport") == "stdio":
                if config.get("command"):
                    server_config["command"] = config["command"]
                    
                    # Handle server_args
                    server_args = config.get("args", [])
                    if isinstance(server_args, str):
                        server_args = server_args.split()
                    server_config["args"] = server_args
                    
                    # Handle environment variables
                    server_env = config.get("env", {})
                    if server_env:
                        server_config["env"] = server_env
                        
            # Add authentication if provided
            if config.get("connection_token"):
                server_config["token"] = config["connection_token"]
                
            server_configs[server_name] = server_config
            
        return server_configs

    async def get_or_create_client(self) -> Optional[object]:
        """
        Get or create a MultiServerMCPClient with proper lifecycle management.
        
        Returns:
            MultiServerMCPClient: The client instance or None if creation fails
        """
        async with self._client_lock:
            if self._client is not None:
                return self._client
                
            if not self.mcp_configs:
                logger.error("No MCP server configurations found")
                return None
                
            try:
                # Convert configs to langchain format
                server_configs = self.convert_configs_to_langchain_format()
                logger.info(f"Creating MCP client for {len(server_configs)} server(s)")
                
                # 过滤掉可能有问题的服务器配置
                filtered_configs = {}
                for server_name, config in server_configs.items():
                    try:
                        # 检查Excel MCP配置，添加错误处理
                        if "excel" in server_name.lower():
                            # 为Excel MCP添加特殊的错误处理配置
                            if "env" not in config:
                                config["env"] = {}
                            config["env"]["PYTHONUNBUFFERED"] = "1"
                            config["env"]["PYTHONIOENCODING"] = "utf-8"
                            logger.info(f"为Excel MCP添加特殊配置: {server_name}")
                        
                        # 验证配置的完整性
                        if self._validate_server_config(config):
                            filtered_configs[server_name] = config
                        else:
                            logger.warning(f"服务器配置 {server_name} 验证失败，跳过")
                    except Exception as e:
                        logger.warning(f"跳过有问题的服务器配置 {server_name}: {e}")
                        continue
                
                if not filtered_configs:
                    logger.error("没有可用的服务器配置")
                    return None
                
                # Initialize the MultiServerMCPClient
                self._client = MultiServerMCPClient(filtered_configs)
                
                return self._client
                
            except Exception as e:
                logger.error(f"Error creating MCP client: {e}")
                return None

    async def close_client(self):
        """
        Properly close the MCP client and clean up resources.
        """
        async with self._client_lock:
            if self._client is not None:
                try:
                    # Since MultiServerMCPClient doesn't support context manager
                    # or explicit close methods in langchain-mcp-adapters 0.1.0,
                    # we just clear the reference and let garbage collection handle it
                    logger.debug("Releasing MCP client reference")
                except Exception as e:
                    logger.error(f"Error during MCP client cleanup: {e}")
                finally:
                    # Always clear the reference
                    self._client = None

    async def get_all_tools(self) -> List:
        """
        Get all available tools from MCP servers.
        
        Returns:
            List: All available MCP tools
        """
        client = await self.get_or_create_client()
        if not client:
            return []
            
        try:
            # Get tools from all servers with timeout
            import asyncio
            all_tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)
            
            if all_tools:
                logger.info(f"Loaded {len(all_tools)} total tools from MCP servers")
                return all_tools
            else:
                logger.warning("No tools available from MCP servers")
                return []
                
        except asyncio.TimeoutError:
            logger.error("MCP工具获取超时")
            return []
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            # 尝试关闭客户端并重新创建
            try:
                await self.close_client()
            except:
                pass
            return []
    
    def _validate_server_config(self, config: Dict[str, Any]) -> bool:
        """
        验证服务器配置的完整性
        
        Args:
            config: 服务器配置字典
            
        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查必需的配置项
            transport = config.get("transport")
            if not transport:
                return False
            
            # 根据传输类型验证配置
            if transport == "stdio":
                # stdio传输需要command
                if not config.get("command"):
                    return False
            elif transport in ["websocket", "streamable_http", "http"]:
                # 网络传输需要URL
                if not config.get("url"):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置类 - 使用Pydantic Settings管理配置"""
    
    # Azure OpenAI配置
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    # 百炼apikey
    dashscope_api_key: str
    
    # Tavily 搜索 API 配置
    tavily_api_key: str

    fast_llm: str
    strategic_llm: str
    coding_llm: str
    
    # 服务器配置
    host: str
    port: int

    embedding: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings():
    return Settings()

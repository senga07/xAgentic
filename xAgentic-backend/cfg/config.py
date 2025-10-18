from cfg.setting import get_settings
from typing import Dict, Any

class Config:
    """配置类 - 单例模式，避免重复加载配置"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.llm_kwargs: Dict[str, Any] = {}

            self._set_llm_attributes()
            self._initialized = True

    def _set_llm_attributes(self) -> None:
        settings = get_settings()
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(settings.fast_llm)
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(settings.strategic_llm)
        self.coding_llm_provider, self.coding_llm_model = self.parse_llm(settings.coding_llm)
        self.embedding_provider, self.embedding_model = self.parse_llm(settings.embedding)


    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """Parse llm string into (llm_provider, llm_model)."""
        from llm_provider.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, (
                    f"Unsupported {llm_provider}.\nSupported llm providers are: "
                    + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return llm_provider, llm_model
        except ValueError:
            raise ValueError(
                "Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' "
                "Eg 'azure_openai:gpt-4o-mini'"
            )

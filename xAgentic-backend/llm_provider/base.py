import importlib
import subprocess
import sys
import os
from typing import Any
from colorama import Fore, Style, init
from cfg.setting import get_settings

_SUPPORTED_PROVIDERS = {
    "azure_openai",
    "dashscope",
}

SUPPORT_REASONING_EFFORT_MODELS = [
    "o4-mini",
]


class GenericLLMProvider:
    """The generic LLM provider."""

    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def from_provider(cls, provider: str, **kwargs: Any):
        settings = get_settings()
        
        if provider == "azure_openai":
            _check_pkg("langchain_openai")
            from langchain_openai import AzureChatOpenAI

            kwargs = {"azure_endpoint": settings.azure_openai_endpoint,
                      "api_key": settings.azure_openai_api_key,
                      "api_version": settings.azure_openai_api_version,
                      **kwargs}

            llm = AzureChatOpenAI(**kwargs)
        elif provider == "dashscope":
            _check_pkg("dashscope")
            from langchain_community.chat_models.tongyi import ChatTongyi

            kwargs = {"api_key": settings.dashscope_api_key, **kwargs}
            llm = ChatTongyi(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: {', '.join(_SUPPORTED_PROVIDERS)}")
        
        return cls(llm)


def _check_pkg(pkg: str) -> None:
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        # Import colorama and initialize it
        init(autoreset=True)

        try:
            print(f"{Fore.YELLOW}Installing {pkg_kebab}...{Style.RESET_ALL}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg_kebab])
            print(f"{Fore.GREEN}Successfully installed {pkg_kebab}{Style.RESET_ALL}")

            # Try importing again after install
            importlib.import_module(pkg)

        except subprocess.CalledProcessError:
            raise ImportError(
                Fore.RED + f"Failed to install {pkg_kebab}. Please install manually with "
                           f"`pip install -U {pkg_kebab}`"
            )


def get_llm(llm_provider, **kwargs):
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)
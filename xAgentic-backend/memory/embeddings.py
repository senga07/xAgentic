from typing import Any

from cfg.setting import get_settings

_SUPPORTED_PROVIDERS = {
    "azure_openai",
    "dashscope",
}


class Embeddings:
    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
        settings = get_settings()
        _embeddings = None
        match embedding_provider:
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings
                _embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=settings.azure_endpoint,
                    api_key=settings.azure_openai_api_key,
                    openai_api_version=settings.azure_openai_api_version,
                    **embedding_kwargs,
                )
            case "dashscope":
                from langchain_community.embeddings import DashScopeEmbeddings
                _embeddings = DashScopeEmbeddings(
                    model=model,
                    dashscope_api_key=settings.dashscope_api_key,
                    **embedding_kwargs)
            case _:
                raise Exception("Embedding not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings

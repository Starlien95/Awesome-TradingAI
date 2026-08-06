import os
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union

import httpx
from loguru import logger
from pydantic import BaseModel


class EmbeddingObject(BaseModel):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingSuccessResponse(BaseModel):
    object: Literal["list"]
    data: List[EmbeddingObject]
    model: Literal[
        "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"
    ]
    usage: Dict[str, int]


class ErrorObject(BaseModel):
    message: str
    type: str
    param: Union[None, str]
    code: Union[None, str]


class EmbeddingErrorResponse(BaseModel):
    error: ErrorObject


class OpenAIEmbeddingError(Exception):
    def __init__(self, message: str, error_type: str) -> None:
        self.message = (
            f"OpenAI Embedding failed, with error type {error_type}, "
            f"error message: *[{message}]*"
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class EmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, emb_config: Dict) -> None:
        self.config = emb_config
        logger.trace(f"EMB-Initializing OpenAIEmbedding with config: {self.config}")

        openai_api_key = (
            self.config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )

        if not openai_api_key:
            logger.error("Can not find OPENAI_API_KEY / DASHSCOPE_API_KEY")
            raise ValueError("Can not find OPENAI_API_KEY / DASHSCOPE_API_KEY")

        self.header = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        self.batch_size = int(self.config.get("embedding_batch_size", 10))
        self.max_retries = int(self.config.get("embedding_max_retries", 8))
        self.retry_base_sleep = float(self.config.get("embedding_retry_base_sleep", 2.0))
        self.retry_max_sleep = float(self.config.get("embedding_retry_max_sleep", 60.0))

        # httpx 默认 trust_env=True，会读取 http_proxy/https_proxy/all_proxy。
        # 你之前的报错显示请求走了 127.0.0.1:7890 代理并在 TLS 阶段 reset。
        # 如果直连 DashScope 更稳定，可以 export EMBEDDING_DISABLE_PROXY=1。
        self.disable_proxy = str(
            self.config.get("disable_proxy", os.environ.get("EMBEDDING_DISABLE_PROXY", "0"))
        ).lower() in {"1", "true", "yes", "y"}

    def _build_client(self) -> httpx.Client:
        timeout_seconds = float(self.config.get("embedding_timeout", 600))

        timeout = httpx.Timeout(
            connect=timeout_seconds,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )

        # max_keepalive_connections=0：避免复用被代理/服务器重置的旧连接。
        limits = httpx.Limits(
            max_connections=10,
            max_keepalive_connections=0,
            keepalive_expiry=0,
        )

        trust_env = not self.disable_proxy

        proxy_env = {
            "http_proxy": os.environ.get("http_proxy"),
            "https_proxy": os.environ.get("https_proxy"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            "all_proxy": os.environ.get("all_proxy"),
            "ALL_PROXY": os.environ.get("ALL_PROXY"),
        }

        logger.info(
            "EMB-httpx client settings: "
            f"trust_env={trust_env}, disable_proxy={self.disable_proxy}, "
            f"timeout={timeout_seconds}, proxy_env={proxy_env}"
        )

        return httpx.Client(
            timeout=timeout,
            limits=limits,
            trust_env=trust_env,
            http2=False,
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        base = min(self.retry_max_sleep, self.retry_base_sleep * (2 ** attempt))
        jitter = random.uniform(0, min(1.0, base * 0.1))
        sleep_s = base + jitter

        logger.warning(
            f"EMB-Retry sleeping {sleep_s:.2f}s "
            f"(attempt={attempt + 1}/{self.max_retries})"
        )
        time.sleep(sleep_s)

    def _post_embedding_with_retry(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        last_error = None

        retryable_exceptions = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
            httpx.NetworkError,
            httpx.TransportError,
        )

        for attempt in range(self.max_retries):
            try:
                # 每次 retry 都新建 client，避免复用已被 reset 的坏连接。
                with self._build_client() as client:
                    response = client.post(
                        url=self.config["request_endpoint"],
                        headers=self.header,
                        json=request_data,
                    )

                if response.status_code == 200:
                    return response.json()

                # 429 / 5xx 通常可以重试。
                if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    last_error = RuntimeError(
                        f"Retryable API status {response.status_code}: {response.text[:500]}"
                    )
                    logger.warning(f"EMB-Retryable API Error: {last_error}")

                    if attempt < self.max_retries - 1:
                        self._sleep_before_retry(attempt)
                        continue

                    raise last_error

                # 其他 4xx 多半是 key / 参数 / 权限问题，不建议重试。
                logger.error(f"EMB-Embedding API Error: {response.text}")
                raise RuntimeError(
                    f"API Request failed with status {response.status_code}: {response.text}"
                )

            except retryable_exceptions as e:
                last_error = e
                logger.warning(
                    f"EMB-Network error on embedding request: "
                    f"{type(e).__name__}: {e}"
                )

                if attempt < self.max_retries - 1:
                    self._sleep_before_retry(attempt)
                    continue

                raise RuntimeError(
                    "Embedding request failed after retries. "
                    "This is usually a network/proxy issue. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e

        raise RuntimeError(f"Embedding request failed after retries: {last_error}")

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        logger.trace(
            f"EMB-Calling OpenAIEmbedding with model: {self.config['emb_model_name']}, "
            f"endpoint: {self.config['request_endpoint']}"
        )

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            request_data = {
                "input": batch_texts,
                "model": self.config["emb_model_name"],
                "parameters": {
                    "text_type": "document"
                }
            }

            res_json = self._post_embedding_with_retry(request_data)

            data_list = res_json.get("data", [])
            sorted_data = sorted(data_list, key=lambda x: x.get("index", 0))

            batch_embeddings = []
            for item in sorted_data:
                emb = item.get("embedding")
                if emb is None:
                    raise RuntimeError(f"Embedding response item missing embedding: {item}")
                batch_embeddings.append(emb)

            if len(batch_embeddings) != len(batch_texts):
                raise RuntimeError(
                    "Embedding response length mismatch: "
                    f"expected={len(batch_texts)}, got={len(batch_embeddings)}, "
                    f"response={str(res_json)[:500]}"
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

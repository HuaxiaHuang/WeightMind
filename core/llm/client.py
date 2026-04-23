"""
LLM 客户端统一封装
支持 OpenAI / DeepSeek / Ollama 等兼容接口
"""
import json
import re
from typing import AsyncIterator, Optional

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings


class LLMClient:
    """统一 LLM 调用接口"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                timeout=180,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """
        单次完成调用
        json_mode=True 时使用 JSON 输出格式
        """
        model = model or settings.LLM_SUMMARY_MODEL
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        
        # 如果期望 JSON 但没有启用 json_mode，尝试提取 JSON 块
        if json_mode:
            return self._extract_json(content)
        
        return content

    async def stream_complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        流式输出
        Yields: 文本片段
        """
        model = model or settings.LLM_MODEL
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取 JSON 块"""
        # 尝试直接解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        
        # 提取 ```json ... ``` 块
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # 提取裸 JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        
        logger.warning(f"无法从 LLM 输出中提取 JSON: {text[:200]}")
        return text


# 全局单例
llm_client = LLMClient()

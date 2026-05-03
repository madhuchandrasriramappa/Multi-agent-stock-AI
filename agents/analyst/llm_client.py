"""
Thin wrapper around Azure OpenAI Chat Completions.

Falls back to a deterministic mock response when Azure credentials are absent
so the full pipeline runs locally without any paid cloud service.
"""
from __future__ import annotations

from config.settings import settings

_MOCK_REPORT = (
    "[Mock report — set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for real analysis.] "
    "Technical indicators show mixed signals with no strong directional bias at this time. "
    "The recorded anomaly alerts warrant attention but do not yet form a conclusive pattern. "
    "Prudent risk management and close monitoring of upcoming candles is recommended. "
    "OUTLOOK: NEUTRAL"
)


class LLMClient:
    """
    Azure OpenAI client with automatic mock fallback.

    Mock mode is active when AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY
    are not set — which is the default for local Docker-based development.
    """

    def __init__(self) -> None:
        self._mock = not (settings.azure_openai_endpoint and settings.azure_openai_api_key)
        if not self._mock:
            from openai import AzureOpenAI  # lazy import — not needed in mock mode
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            self._deployment = settings.azure_openai_deployment

    @property
    def is_mock(self) -> bool:
        return self._mock

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> tuple[str, int, int]:
        """
        Call the LLM and return (report_text, prompt_tokens, completion_tokens).
        In mock mode returns a fixed placeholder and (0, 0) token counts.
        """
        if self._mock:
            return _MOCK_REPORT, 0, 0

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return text, response.usage.prompt_tokens, response.usage.completion_tokens

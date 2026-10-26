import httpx
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from app.routing.circuit_breaker import CircuitBreaker, CircuitState
from app.core.config import settings

logger = logging.getLogger("synapse.routing")

class Provider:
    def __init__(self, name: str, base_url: str, api_key: Optional[str] = None, priority: int = 1):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.priority = priority
        self.breaker = CircuitBreaker(
            name=name,
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_time_sec=settings.CIRCUIT_BREAKER_RESET_TIMEOUT_SEC
        )

class ProviderPool:
    """
    Manages upstream LLM providers, health state, circuit breaking,
    and automatic fallback routing.
    """
    def __init__(self):
        self.providers: Dict[str, Provider] = {}
        self._init_default_providers()
        self.client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SEC)

    def _init_default_providers(self):
        # 1. Groq (Ultra-fast Llama-3 inference)
        if settings.GROQ_API_KEY:
            self.providers["groq"] = Provider(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
                priority=1
            )
        
        # 2. OpenAI
        if settings.OPENAI_API_KEY:
            self.providers["openai"] = Provider(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key=settings.OPENAI_API_KEY,
                priority=2
            )

        # 3. Local Ollama (Always available fallback if local daemon is running)
        self.providers["ollama"] = Provider(
            name="ollama",
            base_url=f"{settings.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            priority=3
        )

        # 4. Built-in Local Simulator (for development, testing, and offline demos)
        self.providers["mock"] = Provider(
            name="mock-simulator",
            base_url="mock://internal",
            api_key=None,
            priority=99
        )

    def get_healthy_providers(self) -> List[Provider]:
        """Returns sorted list of providers that are eligible for execution."""
        eligible = [p for p in self.providers.values() if p.breaker.can_execute()]
        return sorted(eligible, key=lambda p: p.priority)

    async def forward_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Tuple[Any, str, float]:
        """
        Attempts execution across healthy providers sequentially until one succeeds.
        If all fail or trip circuit breakers, raises an aggregated exception.
        """
        eligible = self.get_healthy_providers()
        if not eligible:
            raise RuntimeError("All upstream providers are currently UNHEALTHY (Circuit breakers OPEN).")

        errors = []
        for provider in eligible:
            logger.info(f"Attempting upstream request via [{provider.name}]...")
            start_time = time.time()
            try:
                if provider.name == "mock-simulator":
                    # Realistic simulated LLM delay (e.g. 150ms) and OpenAI-compliant response
                    await asyncio.sleep(0.15)
                    last_msg = payload.get("messages", [{}])[-1].get("content", "")
                    mock_resp = {
                        "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": payload.get("model", "mock-llama-3"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"[Synapse Sim] Generated response to: '{last_msg[:60]}...'"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(last_msg.split()),
                            "completion_tokens": 42,
                            "total_tokens": len(last_msg.split()) + 42
                        }
                    }
                    provider.breaker.record_success()
                    elapsed = time.time() - start_time
                    return mock_resp, provider.name, elapsed

                # Prepare headers
                req_headers = {"Content-Type": "application/json"}
                if provider.api_key:
                    req_headers["Authorization"] = f"Bearer {provider.api_key}"

                target_url = f"{provider.base_url}/{endpoint.lstrip('/')}"
                response = await self.client.post(target_url, json=payload, headers=req_headers)

                if response.status_code >= 400:
                    provider.breaker.record_failure()
                    logger.warning(
                        f"Provider [{provider.name}] returned HTTP {response.status_code}: {response.text[:120]}"
                    )
                    errors.append(f"{provider.name}: HTTP {response.status_code}")
                    continue  # Fall through to next provider

                # Success!
                provider.breaker.record_success()
                elapsed = time.time() - start_time
                logger.info(f"Provider [{provider.name}] succeeded in {elapsed*1000:.1f}ms")
                return response.json(), provider.name, elapsed

            except Exception as exc:
                provider.breaker.record_failure()
                logger.error(f"Provider [{provider.name}] failed with error: {exc}")
                errors.append(f"{provider.name}: {str(exc)}")

        raise RuntimeError(f"All available providers failed: {'; '.join(errors)}")

    def pool_status(self) -> List[dict]:
        return [p.breaker.status() for p in self.providers.values()]

    async def aclose(self) -> None:
        """Cleanly releases underlying HTTP connection pools."""
        await self.client.aclose()

provider_pool = ProviderPool()

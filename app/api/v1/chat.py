import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status, Response
from fastapi.responses import StreamingResponse
import json
import asyncio
from pydantic import BaseModel, Field, ConfigDict

from app.cache.semantic_cache import semantic_cache
from app.guardrails.pii_filter import PIIFilter
from app.guardrails.injection_detector import PromptInjectionDetector
from app.routing.provider_pool import provider_pool
from app.telemetry.metrics import telemetry
from app.core.config import settings

# BPE Tokenizer for accurate production token accounting
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

def count_tokens(text: str) -> int:
    """Exact BPE token count using OpenAI cl100k_base tokenizer."""
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))

from app.core.security import verify_gateway_api_key
from fastapi import Depends

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str = Field(..., max_length=settings.MAX_PROMPT_LENGTH)

class ChatCompletionRequest(BaseModel):
    model: str = "llama-3-8b-instant"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

    model_config = ConfigDict(extra="allow")

@router.post("/chat/completions", dependencies=[Depends(verify_gateway_api_key)])
async def chat_completions(
    req: ChatCompletionRequest,
    raw_request: Request,
    response: Response
):
    req_id = f"syn-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    # 1. Extract tenant, user prompt, and system prompt context
    tenant_id = raw_request.headers.get("X-Tenant-ID", "default")
    system_messages = [m for m in req.messages if m.role == "system"]
    system_prompt = system_messages[0].content if system_messages else ""
    temperature = req.temperature if req.temperature is not None else 0.7

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat completion requires at least one user message."
        )
    raw_prompt = user_messages[-1].content

    # 2. Ingress Guardrails Check
    pii_types_found: List[str] = []
    sanitized_prompt = raw_prompt

    if settings.GUARDRAILS_ENABLED:
        # A. Prompt Injection & Adversarial Jailbreak Check
        if settings.BLOCK_PROMPT_INJECTION:
            is_injection, triggers, risk_score = PromptInjectionDetector.evaluate(raw_prompt)
            if is_injection:
                telemetry.record_request(
                    request_id=req_id,
                    model=req.model,
                    prompt=raw_prompt,
                    status="blocked_injection",
                    latency_ms=(time.time() - start_time) * 1000,
                    source="guardrail_blocked",
                    is_injection=True
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "SecurityViolation",
                        "message": "Potential prompt injection or jailbreak pattern detected.",
                        "triggers": triggers,
                        "risk_score": risk_score
                    }
                )

        # B. Sensitive Data / PII Masking with Luhn Verification
        if settings.MASK_PII:
            sanitized_prompt, pii_types_found = PIIFilter.sanitize(raw_prompt)
            if pii_types_found:
                user_messages[-1].content = sanitized_prompt

    # 3. Two-Tier Multi-Tenant Semantic Vector Cache Lookup
    if settings.CACHE_ENABLED:
        cached_result, similarity, hit_type = await semantic_cache.lookup(
            prompt=sanitized_prompt,
            model=req.model,
            tenant_id=tenant_id,
            system_prompt=system_prompt,
            temperature=temperature
        )
        if cached_result is not None:
            latency_ms = (time.time() - start_time) * 1000
            
            # Accurate token accounting: prompt tokens + cached completion tokens
            cached_completion = ""
            try:
                cached_completion = cached_result["choices"][0]["message"]["content"]
            except Exception:
                pass
            tokens_saved = count_tokens(sanitized_prompt) + count_tokens(cached_completion)
            
            telemetry.record_request(
                request_id=req_id,
                model=req.model,
                prompt=sanitized_prompt,
                status="success_cached",
                latency_ms=latency_ms,
                source=f"cache_{hit_type}",
                tokens_saved=tokens_saved,
                pii_detected=pii_types_found
            )
            
            # Set high-visibility telemetry headers
            response_headers = {
                "X-Synapse-Cache": "HIT",
                "X-Synapse-Hit-Type": hit_type.upper(),
                "X-Synapse-Similarity": f"{similarity:.4f}",
                "X-Synapse-Tokens-Saved": str(tokens_saved),
                "X-Synapse-Tenant-ID": tenant_id,
                "X-Synapse-Latency-Ms": f"{latency_ms:.2f}"
            }

            if req.stream:
                async def stream_cached_sse():
                    content = ""
                    try:
                        content = cached_result["choices"][0]["message"]["content"]
                    except Exception:
                        content = str(cached_result)

                    words = content.split(" ")
                    for i, word in enumerate(words):
                        chunk = {
                            "id": f"chatcmpl-{req_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.005)

                    done_chunk = {
                        "id": f"chatcmpl-{req_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_cached_sse(),
                    media_type="text/event-stream",
                    headers=response_headers
                )

            for k, v in response_headers.items():
                response.headers[k] = v
            return cached_result

    # 4. Cache MISS: Forward through Resilient Provider Router
    try:
        payload = req.model_dump()
        result, provider_name, provider_latency = await provider_pool.forward_request(
            endpoint="/chat/completions",
            payload=payload,
            headers={}
        )
        total_latency_ms = (time.time() - start_time) * 1000

        # Extract tokens from provider usage metadata
        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens", count_tokens(sanitized_prompt) + 50)

        # 5. Persist into Isolated Multi-Tenant Semantic Cache
        if settings.CACHE_ENABLED:
            await semantic_cache.store(
                prompt=sanitized_prompt,
                model=req.model,
                response_payload=result,
                tokens_saved=total_tokens,
                tenant_id=tenant_id,
                system_prompt=system_prompt,
                temperature=temperature
            )

        telemetry.record_request(
            request_id=req_id,
            model=req.model,
            prompt=sanitized_prompt,
            status="success_upstream",
            latency_ms=total_latency_ms,
            source=f"upstream_{provider_name}",
            tokens_saved=0,
            pii_detected=pii_types_found
        )

        response.headers["X-Synapse-Cache"] = "MISS"
        response.headers["X-Synapse-Provider"] = provider_name
        response.headers["X-Synapse-Tenant-ID"] = tenant_id
        response.headers["X-Synapse-Latency-Ms"] = f"{total_latency_ms:.2f}"
        return result

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        telemetry.record_request(
            request_id=req_id,
            model=req.model,
            prompt=sanitized_prompt,
            status="failed",
            latency_ms=latency_ms,
            source="none",
            pii_detected=pii_types_found
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SynapseGateway error across upstream providers: {str(e)}"
        )

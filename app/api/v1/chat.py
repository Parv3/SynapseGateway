import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status, Response
from pydantic import BaseModel, Field

from app.cache.semantic_cache import semantic_cache
from app.guardrails.pii_filter import PIIFilter
from app.guardrails.injection_detector import PromptInjectionDetector
from app.routing.provider_pool import provider_pool
from app.telemetry.metrics import telemetry
from app.core.config import settings

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llama-3-8b-instant"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

    class Config:
        extra = "allow"

@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    raw_request: Request,
    response: Response
):
    req_id = f"syn-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    # 1. Extract the latest user prompt for caching and evaluation
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat completion requires at least one user message."
        )
    raw_prompt = user_messages[-1].content

    # 2. Input Guardrails Check
    pii_types_found: List[str] = []
    sanitized_prompt = raw_prompt

    if settings.GUARDRAILS_ENABLED:
        # A. Prompt Injection Detection
        if settings.BLOCK_PROMPT_INJECTION:
            is_injection, triggers = PromptInjectionDetector.evaluate(raw_prompt)
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
                        "triggers": triggers
                    }
                )

        # B. Sensitive Data / PII Masking
        if settings.MASK_PII:
            sanitized_prompt, pii_types_found = PIIFilter.sanitize(raw_prompt)
            if pii_types_found:
                # Update prompt in place for upstream request
                user_messages[-1].content = sanitized_prompt

    # 3. Semantic Vector Cache Lookup
    if settings.CACHE_ENABLED:
        cached_result, similarity, hit_type = await semantic_cache.lookup(
            prompt=sanitized_prompt,
            model=req.model
        )
        if cached_result is not None:
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute tokens saved approximation
            tokens_saved = int(len(sanitized_prompt.split()) * 1.3) + 150
            
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
            
            # Decorate response headers
            response.headers["X-Synapse-Cache"] = "HIT"
            response.headers["X-Synapse-Hit-Type"] = hit_type.upper()
            response.headers["X-Synapse-Similarity"] = f"{similarity:.4f}"
            response.headers["X-Synapse-Latency-Ms"] = f"{latency_ms:.2f}"
            
            return cached_result

    # 4. Cache MISS: Forward through resilient Provider Router
    try:
        # Prepare JSON payload matching OpenAI specs
        payload = req.model_dump()
        result, provider_name, provider_latency = await provider_pool.forward_request(
            endpoint="/chat/completions",
            payload=payload,
            headers={}
        )
        total_latency_ms = (time.time() - start_time) * 1000

        # Estimate tokens produced
        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens", len(sanitized_prompt.split()) + 100)

        # 5. Store into Semantic Cache for future identical / semantically similar queries
        if settings.CACHE_ENABLED:
            await semantic_cache.store(
                prompt=sanitized_prompt,
                model=req.model,
                response_payload=result,
                tokens_saved=total_tokens
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

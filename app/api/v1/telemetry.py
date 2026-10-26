from fastapi import APIRouter
from app.telemetry.metrics import telemetry
from app.cache.semantic_cache import semantic_cache
from app.routing.provider_pool import provider_pool

router = APIRouter()

@router.get("/telemetry/metrics")
async def get_metrics():
    """
    Returns live aggregated gateway telemetry:
    - Total requests & cache hit rate
    - P99 latency savings
    - Financial cost savings ($)
    - Security events (PII masked, Injections blocked)
    - Recent event stream
    """
    return telemetry.summary()

@router.get("/telemetry/cache")
async def get_cache_stats():
    """
    Returns internal vector cache statistics.
    """
    return semantic_cache.stats()

@router.get("/telemetry/providers")
async def get_provider_health():
    """
    Returns circuit breaker state and health metrics for all upstream providers.
    """
    return {
        "providers": provider_pool.pool_status()
    }

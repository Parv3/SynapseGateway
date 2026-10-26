import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class RequestLog:
    id: str
    timestamp: float
    model: str
    prompt_snippet: str
    status: str
    latency_ms: float
    source: str  # "cache_exact", "cache_semantic", "upstream_<provider>"
    tokens_saved: int
    pii_detected: List[str]
    is_injection: bool

class TelemetryCollector:
    """
    Central in-memory metrics and telemetry aggregation.
    Provides live data feeds for the UI dashboard with O(1) ring buffers.
    """
    def __init__(self, max_recent_logs: int = 100):
        self.max_recent_logs = max_recent_logs
        self.recent_logs: deque = deque(maxlen=max_recent_logs)
        
        self.total_requests: int = 0
        self.total_cache_hits: int = 0
        self.total_tokens_saved: int = 0
        self.total_blocked_injections: int = 0
        self.total_pii_masked: int = 0
        
        # O(1) rolling latency tracking
        self.cache_latencies: deque = deque(maxlen=500)
        self.provider_latencies: deque = deque(maxlen=500)

    def record_request(
        self,
        request_id: str,
        model: str,
        prompt: str,
        status: str,
        latency_ms: float,
        source: str,
        tokens_saved: int = 0,
        pii_detected: Optional[List[str]] = None,
        is_injection: bool = False
    ) -> None:
        self.total_requests += 1
        if "cache" in source:
            self.total_cache_hits += 1
            self.cache_latencies.append(latency_ms)
        else:
            self.provider_latencies.append(latency_ms)

        self.total_tokens_saved += tokens_saved
        if is_injection:
            self.total_blocked_injections += 1
        if pii_detected:
            self.total_pii_masked += len(pii_detected)

        log = RequestLog(
            id=request_id,
            timestamp=time.time(),
            model=model,
            prompt_snippet=prompt[:80],
            status=status,
            latency_ms=round(latency_ms, 2),
            source=source,
            tokens_saved=tokens_saved,
            pii_detected=pii_detected or [],
            is_injection=is_injection
        )
        self.recent_logs.append(log)

    def summary(self) -> Dict[str, Any]:
        avg_cache_lat = round(sum(self.cache_latencies) / len(self.cache_latencies), 2) if self.cache_latencies else 0.0
        avg_prov_lat = round(sum(self.provider_latencies) / len(self.provider_latencies), 2) if self.provider_latencies else 0.0
        
        # Estimated cost saved ($0.01 per 1k tokens standard blended pricing)
        cost_saved_usd = round((self.total_tokens_saved / 1000) * 0.01, 4)

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.total_cache_hits,
            "cache_hit_rate_pct": round((self.total_cache_hits / self.total_requests * 100), 2) if self.total_requests else 0.0,
            "tokens_saved": self.total_tokens_saved,
            "cost_saved_usd": cost_saved_usd,
            "avg_cache_latency_ms": avg_cache_lat,
            "avg_provider_latency_ms": avg_prov_lat,
            "blocked_injections": self.total_blocked_injections,
            "pii_masked_count": self.total_pii_masked,
            "recent_events": [
                {
                    "id": l.id,
                    "model": l.model,
                    "prompt": l.prompt_snippet,
                    "source": l.source,
                    "latency_ms": l.latency_ms,
                    "status": l.status,
                    "time": time.strftime("%H:%M:%S", time.localtime(l.timestamp))
                }
                for l in reversed(list(self.recent_logs))
            ]
        }

telemetry = TelemetryCollector()

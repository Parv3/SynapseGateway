# SynapseGateway

An asynchronous, drop-in AI inference gateway that sits between client applications and upstream LLM providers. It brings two-tier semantic vector caching, active circuit breakers, and inline data sanitization into a single lightweight service.

[![CI](https://github.com/Parv3/SynapseGateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Parv3/SynapseGateway/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Why this exists

When running LLMs in production, calling provider APIs directly introduces three major operational problems:

1. **High latency on repetitive traffic:** Natural language queries often express the same intent in different words (*"How do I reset my password?"* vs *"Steps to reset user password"*). Naive string caches miss these queries, forcing roundtrips that take 1.5 to 4 seconds and consume billable tokens each time.
2. **Provider instability:** Upstream rate limits (HTTP 429) and outages (HTTP 503) cascade directly into client applications unless there is automated, circuit-breaking failover.
3. **Data leakage:** Raw prompts frequently contain customer identifiers, credit card numbers, or API keys that should never leave the infrastructure perimeter.

SynapseGateway runs locally or on the edge as an OpenAI-compatible reverse proxy, resolving semantically identical queries in **under 1 millisecond**, automatically sanitizing sensitive data before egress, and rerouting failed provider calls with zero downtime.

---

## Architecture

```mermaid
flowchart TD
    Client["Client App / OpenAI SDK"] -->|POST /v1/chat/completions| Ingress["Ingress Perimeter"]

    subgraph Security ["Security & Sanitization"]
        Ingress --> PII["Luhn Algorithm & PII Masking"]
        PII --> Injection["Weighted Prompt Injection Filter"]
    end

    Injection -->|Passed| Cache["Two-Tier Semantic Cache"]
    Injection -->|Flagged| Block["HTTP 400 Bad Request (< 7ms)"]

    subgraph Caching ["Cache Hierarchy"]
        Cache --> L1{"L1: Exact Match?"}
        L1 -->|Hit < 0.2ms| CacheHit["Return Cached Payload"]
        L1 -->|Miss| L2{"L2: Vector Similarity ≥ 0.90?"}
        L2 -->|Hit < 10ms| CacheHit
    end

    L2 -->|Miss| Router["Resilient Router"]

    subgraph Routing ["Provider Failover Pool"]
        Router --> Breaker{"Circuit Breaker State?"}
        Breaker -->|CLOSED| Primary["Primary Provider (Groq / OpenAI)"]
        Primary -->|429 / 5xx| Trip["Trip Breaker to OPEN"]
        Trip --> Secondary["Fallback Provider (Anthropic / Ollama)"]
    end

    Primary -->|Success| Index["Index Response in Vector Cache"]
    Secondary -->|Success| Index
    Index --> Bus["Async Telemetry Bus"]
    CacheHit --> Bus
    Bus --> Dashboard["Live Dashboard / Prometheus"]
```

---

## Core Systems

### 1. Two-Tier Semantic Vector Caching
Standard key-value caches fail in natural language processing because users rarely submit identical strings. SynapseGateway uses a dual-layer approach:
* **L1 Exact Match Hash Ring:** Hashes normalized strings for $O(1)$ in-memory lookups, resolving in less than 0.2ms.
* **L2 Dense Neural Vector Index:** If L1 misses, the prompt is encoded into a 384-dimensional dense vector using an ONNX-optimized transformer (`bge-small-en-v1.5`). A vectorized matrix dot-product checks cosine similarity against cached vectors. If the similarity meets or exceeds the configured threshold (default: `0.90`), the cached response is returned immediately.

### 2. Active Finite-State Circuit Breakers
To prevent slow network timeouts during upstream provider outages, each model provider is managed by a three-state machine (`CLOSED`, `OPEN`, `HALF_OPEN`):
* Under normal conditions (`CLOSED`), requests route to the primary provider.
* If a provider records 3 consecutive failures (timeouts, 429s, or 5xx responses), the circuit trips to `OPEN`. All incoming calls immediately bypass that provider and redirect to secondary fallbacks (e.g., Anthropic, Groq, or local vLLM/Ollama).
* After a 30-second cooldown, the circuit enters `HALF_OPEN` and sends a single probe request. If successful, the circuit resets to `CLOSED`.

### 3. Luhn-Verified PII & Secret Redaction
Rather than relying solely on naive regex matching (which creates false positives on any 16-digit sequence), credit card numbers are verified against the **Luhn Algorithm (Mod 10 Checksum)** before redaction. The guardrail pipeline also strips:
* Standard PII (Email addresses, phone numbers, US SSNs).
* Developer secrets (OpenAI API keys, AWS access keys, Bearer authorization tokens).
* Adversarial prompt injection and jailbreak patterns with weighted risk scoring.

---

## Benchmarks

Results gathered using `scripts/benchmark.py` under simulated production workloads:

| Test Case | Direct Upstream | With SynapseGateway | Measured Delta |
|---|---|---|---|
| **Exact Match (L1 Cache Hit)** | 1,272.5 ms | **0.44 ms** | **2,892x faster** |
| **Semantic Similarity (L2 Cache Hit)** | 1,272.5 ms | **8.20 ms** | **155x faster** |
| **Prompt Injection Rejection** | 1,200+ ms roundtrip | **6.90 ms** | **Blocked at ingress** |
| **Token Cost on Cache Hits** | Variable billable | **$0.00** | **100% offloaded** |
| **Provider 429 / Outage Handling** | Service Exception | **Sub-50ms failover** | **Zero application downtime** |

---

## Industry Comparison

| Capability | Raw Provider SDK | LiteLLM | Portkey.ai | SynapseGateway |
|---|:---:|:---:|:---:|:---:|
| Drop-in OpenAI API Specification | Yes | Yes | Yes | **Yes** |
| Two-Tier Semantic Vector Caching | No | No | Cloud add-on | **Yes (In-Memory / ONNX)** |
| Finite-State Circuit Breaker | No | Retries only | Cloud platform | **Yes (Built-in FSM)** |
| Luhn-Verified Credit Card Scrubbing | No | No | Add-on | **Yes (Built-in)** |
| Zero External Network Hop Overhead | No | No | No | **Yes (Local Edge)** |
| Embedded Real-Time Telemetry Dashboard | No | Requires setup | Cloud hosted | **Yes (Self-contained)** |

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Parv3/SynapseGateway.git
cd SynapseGateway

# Set up virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
```env
# Optional: Upstream API keys (gateway includes a local simulator if omitted)
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here

# Cache sensitivity (0.0 to 1.0)
CACHE_SIMILARITY_THRESHOLD=0.90
```

### 3. Start the Gateway
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Drop-in OpenAI SDK Usage
Point your existing client code to `http://localhost:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="synapse-proxy"
)

response = client.chat.completions.create(
    model="llama-3-8b-instant",
    messages=[{"role": "user", "content": "Explain gradient descent in neural networks."}]
)
print(response.choices[0].message.content)
```

---

## Dashboard

SynapseGateway includes a built-in telemetry console accessible at:
```
http://localhost:8000/dashboard
```
It tracks real-time cache hit ratios, token savings, p99 latency breakdowns, and active circuit-breaker states without requiring external monitoring infrastructure.

---

## Testing & Verification

```bash
# Run unit and integration tests
pytest tests/ -v

# Run the performance benchmark suite
python scripts/benchmark.py
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

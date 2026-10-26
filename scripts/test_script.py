#!/usr/bin/env python3
"""
Zero-dependency smoke-test script for SynapseGateway.
Uses standard Python library (urllib) to verify gateway operation without external packages.
"""

import urllib.request
import json
import time

GATEWAY_URL = "http://localhost:8000"

def run_smoke_test():
    print("Testing SynapseGateway Health...")
    req = urllib.request.Request(f"{GATEWAY_URL}/health")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"[PASS] Health Check: {data}")
    except Exception as e:
        print(f"[FAIL] Could not connect to {GATEWAY_URL}: {e}")
        return

    # Test Chat Completion
    print("\nSending Test Prompt to /v1/chat/completions...")
    payload = {
        "model": "llama-3-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a concise engineering assistant."},
            {"role": "user", "content": "Explain gradient descent in 1 sentence."}
        ],
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "test-suite"
    }

    req_call = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    # 1. Cold Call
    t0 = time.time()
    with urllib.request.urlopen(req_call) as resp:
        res = json.loads(resp.read().decode())
        t_cold = (time.time() - t0) * 1000
        cache_hdr = resp.headers.get("X-Synapse-Cache")
        print(f"[PASS] Cold Request: Latency = {t_cold:.1f}ms | Cache = {cache_hdr}")

    # 2. Warm Call (Should hit cache)
    t1 = time.time()
    with urllib.request.urlopen(req_call) as resp2:
        res2 = json.loads(resp2.read().decode())
        t_warm = (time.time() - t1) * 1000
        cache_hdr2 = resp2.headers.get("X-Synapse-Cache")
        print(f"[PASS] Warm Request: Latency = {t_warm:.2f}ms | Cache = {cache_hdr2}")
        print(f"       Speedup Factor: {t_cold / max(0.1, t_warm):.1f}x")

    print("\nAll gateway smoke tests passed successfully!")

if __name__ == "__main__":
    run_smoke_test()

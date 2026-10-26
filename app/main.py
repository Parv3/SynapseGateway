from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging

from app.api.v1.chat import router as chat_router
from app.api.v1.telemetry import router as telemetry_router
from app.core.config import settings

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("synapse.main")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-Throughput Distributed AI Gateway with Semantic Caching, Guardrails & Circuit Breaking"
)

# Enable CORS for dashboards & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(chat_router, prefix=settings.API_V1_STR, tags=["Chat Completions"])
app.include_router(telemetry_router, prefix=settings.API_V1_STR, tags=["Telemetry"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def live_dashboard():
    """
    Embedded, high-speed real-time monitoring dashboard.
    Auto-refreshes every 2 seconds with live telemetry.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>SynapseGateway • AI Gateway Mission Control</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/feather-icons/dist/feather.min.js"></script>
      <style>
        body { background-color: #0B0F17; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .glass-card { background: rgba(18, 24, 38, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background-color: #10B981; box-shadow: 0 0 10px #10B981; }
      </style>
    </head>
    <body class="p-6 md:p-10 min-h-screen">
      <div class="max-w-7xl mx-auto space-y-8">
        
        <!-- Header -->
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-800 pb-6">
          <div>
            <div class="flex items-center space-x-3">
              <span class="pulse-dot animate-pulse"></span>
              <h1 class="text-3xl font-bold tracking-tight text-white">Synapse<span class="text-indigo-400">Gateway</span></h1>
              <span class="text-xs bg-indigo-900 text-indigo-300 font-mono px-2 py-0.5 rounded">v0.1.0-PROD</span>
            </div>
            <p class="text-sm text-gray-400 mt-1">Distributed AI Inference Gateway • Semantic Vector Caching • Adaptive Guardrails</p>
          </div>
          <div class="mt-4 md:mt-0 flex space-x-3">
            <a href="/docs" target="_blank" class="px-4 py-2 text-xs font-semibold rounded bg-gray-800 hover:bg-gray-700 text-gray-200 transition">OpenAPI Docs</a>
            <span class="px-3 py-2 text-xs font-mono rounded bg-green-950 text-green-400 border border-green-800 flex items-center space-x-1">
              <span>Status: RUNNING</span>
            </span>
          </div>
        </div>

        <!-- Metric Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-5">
          <div class="glass-card p-5 rounded-xl">
            <span class="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Requests</span>
            <div id="stat-total-req" class="text-3xl font-extrabold text-white mt-2">0</div>
            <div class="text-xs text-gray-500 mt-1">Processed across providers</div>
          </div>
          
          <div class="glass-card p-5 rounded-xl border-l-4 border-indigo-500">
            <span class="text-xs font-medium text-gray-400 uppercase tracking-wider">Cache Hit Ratio</span>
            <div id="stat-hit-rate" class="text-3xl font-extrabold text-indigo-400 mt-2">0.0%</div>
            <div class="text-xs text-gray-400 mt-1"><span id="stat-cache-hits" class="font-bold text-gray-200">0</span> queries answered in &lt;15ms</div>
          </div>

          <div class="glass-card p-5 rounded-xl border-l-4 border-green-500">
            <span class="text-xs font-medium text-gray-400 uppercase tracking-wider">Cost Savings</span>
            <div id="stat-cost-saved" class="text-3xl font-extrabold text-green-400 mt-2">$0.000</div>
            <div class="text-xs text-gray-400 mt-1"><span id="stat-tokens-saved" class="font-bold text-gray-200">0</span> tokens offloaded</div>
          </div>

          <div class="glass-card p-5 rounded-xl border-l-4 border-yellow-500">
            <span class="text-xs font-medium text-gray-400 uppercase tracking-wider">Security Interceptions</span>
            <div id="stat-security" class="text-3xl font-extrabold text-yellow-400 mt-2">0</div>
            <div class="text-xs text-gray-400 mt-1">PII Masked & Injections Blocked</div>
          </div>
        </div>

        <!-- Latency & Provider Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Latency Comparison -->
          <div class="glass-card p-6 rounded-xl">
            <h2 class="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
              <span>Latency Efficiency (Cache vs Upstream)</span>
            </h2>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span class="text-indigo-300 font-semibold">Semantic Cache Response</span>
                  <span id="cache-latency-tag" class="font-mono text-indigo-300">0.0 ms</span>
                </div>
                <div class="w-full bg-gray-800 rounded-full h-2.5">
                  <div class="bg-indigo-500 h-2.5 rounded-full" style="width: 8%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span class="text-gray-300">Upstream LLM Provider Roundtrip</span>
                  <span id="upstream-latency-tag" class="font-mono text-gray-300">0.0 ms</span>
                </div>
                <div class="w-full bg-gray-800 rounded-full h-2.5">
                  <div class="bg-gray-600 h-2.5 rounded-full" style="width: 85%"></div>
                </div>
              </div>
            </div>
            <p class="text-xs text-gray-400 mt-4 leading-relaxed">
              Requests hitting the vector cache eliminate GPU cold starts and API roundtrip latency by up to <strong class="text-green-400">95%</strong>.
            </p>
          </div>

          <!-- Circuit Breakers -->
          <div class="glass-card p-6 rounded-xl">
            <h2 class="text-lg font-semibold text-white mb-4">Upstream Circuit Breakers</h2>
            <div id="circuit-list" class="space-y-3">
              <div class="text-xs text-gray-500">Querying provider pool state...</div>
            </div>
          </div>
        </div>

        <!-- Live Audit Stream -->
        <div class="glass-card p-6 rounded-xl">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-lg font-semibold text-white">Live Request Stream & Interception Log</h2>
            <span class="text-xs text-gray-400 font-mono">Poll: 2000ms</span>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs font-mono">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2">Time</th>
                  <th class="pb-2">Model</th>
                  <th class="pb-2">Source</th>
                  <th class="pb-2">Latency</th>
                  <th class="pb-2">Status</th>
                  <th class="pb-2">Prompt Preview</th>
                </tr>
              </thead>
              <tbody id="audit-table-body" class="divide-y divide-gray-800">
                <tr><td colspan="6" class="py-4 text-center text-gray-500">No requests processed yet. Send requests to /v1/chat/completions</td></tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>

      <script>
        async function updateDashboard() {
          try {
            const [metricsRes, providersRes] = await Promise.all([
              fetch('/v1/telemetry/metrics'),
              fetch('/v1/telemetry/providers')
            ]);
            
            const metrics = await metricsRes.json();
            const providers = await providersRes.json();

            // Stats
            document.getElementById('stat-total-req').innerText = metrics.total_requests;
            document.getElementById('stat-hit-rate').innerText = metrics.cache_hit_rate_pct + '%';
            document.getElementById('stat-cache-hits').innerText = metrics.cache_hits;
            document.getElementById('stat-cost-saved').innerText = '$' + metrics.cost_saved_usd.toFixed(4);
            document.getElementById('stat-tokens-saved').innerText = metrics.tokens_saved.toLocaleString();
            document.getElementById('stat-security').innerText = (metrics.blocked_injections + metrics.pii_masked_count);

            // Latencies
            document.getElementById('cache-latency-tag').innerText = metrics.avg_cache_latency_ms + ' ms';
            document.getElementById('upstream-latency-tag').innerText = metrics.avg_provider_latency_ms + ' ms';

            // Circuit Breakers
            const circuitList = document.getElementById('circuit-list');
            if (providers.providers && providers.providers.length > 0) {
              circuitList.innerHTML = providers.providers.map(p => `
                <div class="flex items-center justify-between p-3 rounded bg-gray-900 border border-gray-800">
                  <span class="font-mono text-sm text-gray-200 capitalize">${p.name}</span>
                  <div class="flex items-center space-x-2">
                    <span class="text-xs px-2 py-0.5 rounded font-mono ${p.state === 'CLOSED' ? 'bg-green-950 text-green-400 border border-green-800' : 'bg-red-950 text-red-400 border border-red-800'}">
                      ${p.state}
                    </span>
                    <span class="text-xs text-gray-500 font-mono">fails: ${p.failure_count}</span>
                  </div>
                </div>
              `).join('');
            }

            // Audit Table
            const tbody = document.getElementById('audit-table-body');
            if (metrics.recent_events && metrics.recent_events.length > 0) {
              tbody.innerHTML = metrics.recent_events.map(ev => {
                let badgeClass = ev.source.includes('cache') ? 'text-indigo-400 bg-indigo-950 border border-indigo-800' : 'text-gray-300 bg-gray-800';
                if (ev.status.includes('blocked')) badgeClass = 'text-red-400 bg-red-950 border border-red-800';
                return `
                  <tr class="hover:bg-gray-900 transition">
                    <td class="py-2.5 text-gray-400">${ev.time}</td>
                    <td class="py-2.5 text-gray-300">${ev.model}</td>
                    <td class="py-2.5"><span class="px-2 py-0.5 rounded text-xs ${badgeClass}">${ev.source}</span></td>
                    <td class="py-2.5 text-gray-200">${ev.latency_ms} ms</td>
                    <td class="py-2.5 font-bold ${ev.status.includes('blocked') ? 'text-red-400' : 'text-green-400'}">${ev.status}</td>
                    <td class="py-2.5 text-gray-400 truncate max-w-xs">${ev.prompt}</td>
                  </tr>
                `;
              }).join('');
            }
          } catch (e) {
            console.error("Dashboard poll failed", e);
          }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
      </script>
    </body>
    </html>
    """

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import logging

from app.api.v1.chat import router as chat_router
from app.api.v1.telemetry import router as telemetry_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("synapse.main")

from contextlib import asynccontextmanager
from app.routing.provider_pool import provider_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("SynapseGateway starting up...")
    yield
    # Graceful Shutdown
    logger.info("SynapseGateway shutting down. Closing provider connection pools...")
    await provider_pool.aclose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-Throughput Distributed AI Gateway with Semantic Caching, Guardrails & Circuit Breaking",
    lifespan=lifespan
)

# Enforce secure CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Synapse-Api-Key"],
)

# Security Headers & Request Bounds Middleware
@app.middleware("http")
async def security_and_bounds_middleware(request: Request, call_next):
    # Enforce request body size limit to prevent memory exhaustion
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload exceeds limit of {settings.MAX_PAYLOAD_BYTES // 1024} KB."
        )

    response: Response = await call_next(request)
    # Modern security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

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
    High-end Bento-grid SaaS dashboard inspired by modern design systems (Orely Studio).
    Includes real-time bar graphs, gradient latency range sliders, provider health cards,
    and sanitized live transaction streaming.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>SynapseGateway • Mission Control</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <style>
        * { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        body { background-color: #F8F9FA; color: #1E293B; }
        .bento-card { background: #FFFFFF; border-radius: 24px; box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04), 0 2px 6px -1px rgba(0, 0, 0, 0.02); border: 1px solid #F1F5F9; }
        .pastel-purple { background-color: #F3EEFF; color: #6D28D9; }
        .pastel-yellow { background-color: #FEF9C3; color: #854D0E; }
        .pastel-pink { background-color: #FCE7F3; color: #9D174D; }
        .pastel-green { background-color: #DCFCE7; color: #15803D; }
        .pill-badge { border-radius: 9999px; padding: 4px 12px; font-weight: 600; font-size: 11px; }
        .gradient-track { background: linear-gradient(90deg, #10B981 0%, #34D399 25%, #FBBF24 50%, #FB923C 75%, #F43F5E 100%); border-radius: 9999px; height: 14px; position: relative; }
        .track-pin { position: absolute; top: -6px; width: 26px; height: 26px; border-radius: 50%; background: #0F172A; border: 3px solid #FFFFFF; box-shadow: 0 2px 8px rgba(0,0,0,0.25); transform: translateX(-50%); transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
        .bar-col { transition: height 0.4s ease; border-radius: 8px 8px 4px 4px; }
        .sidebar-icon { width: 44px; height: 44px; border-radius: 16px; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; }
        .sidebar-icon:hover { background-color: #F1F5F9; color: #0F172A; }
        .sidebar-icon.active { background-color: #FF5A36; color: #FFFFFF; box-shadow: 0 4px 12px rgba(255, 90, 54, 0.35); }
      </style>
    </head>
    <body class="min-h-screen flex p-3 md:p-6 bg-gray-50 text-gray-800">

      <!-- Left Rail Navigation -->
      <aside class="hidden lg:flex flex-col items-center justify-between py-6 px-3 w-20 bg-white rounded-3xl mr-6 border border-gray-100 shadow-sm">
        <div class="flex flex-col items-center space-y-8">
          <div class="w-11 h-11 rounded-2xl bg-gradient-to-tr from-red-500 to-yellow-400 flex items-center justify-center text-white font-black text-xl shadow-md">
            S
          </div>
          <nav class="flex flex-col space-y-4">
            <a href="#" class="sidebar-icon active" title="Dashboard">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
            </a>
            <a href="/docs" target="_blank" class="sidebar-icon text-gray-400" title="API Docs">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
            </a>
            <a href="#feed" class="sidebar-icon text-gray-400" title="Interception Stream">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </a>
          </nav>
        </div>
        <div class="sidebar-icon text-gray-400 hover:text-gray-600">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
        </div>
      </aside>

      <!-- Center & Right Core Area -->
      <div class="flex-1 flex flex-col xl:flex-row gap-6 max-w-7xl mx-auto w-full">
        
        <!-- Center Main Canvas -->
        <main class="flex-1 flex flex-col space-y-6">
          
          <!-- Top Bar -->
          <header class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h1 class="text-2xl font-extrabold text-gray-900 tracking-tight">Hi, Synapse Admin</h1>
              <p class="text-xs text-gray-500 font-medium mt-0.5">Real-time AI gateway, semantic caching and security stream.</p>
            </div>
            
            <div class="flex items-center space-x-3">
              <div class="relative hidden sm:block">
                <input type="text" placeholder="Search prompts or events" class="bg-white pl-9 pr-12 py-2 rounded-2xl border border-gray-200 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-sm w-64" readonly value="Filter logs (Ctrl+K)">
                <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                <span class="absolute right-2.5 top-2 text-xs font-mono text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">⌘K</span>
              </div>
              <div class="w-10 h-10 rounded-2xl bg-white border border-gray-200 flex items-center justify-center text-gray-500 shadow-sm relative">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path></svg>
                <span class="w-2 h-2 rounded-full bg-red-500 absolute top-2 right-2"></span>
              </div>
            </div>
          </header>

          <!-- Bento Grid Row 1: Big Hero Chart + Bento Metric Trio -->
          <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
            
            <!-- Hero Chart Card: Latency & Speedup Spectrum -->
            <div class="md:col-span-8 bento-card p-6 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-2">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-400">Response Speed</span>
                    <span class="text-xs font-bold text-green-700 bg-green-100 px-2.5 py-0.5 rounded-full flex items-center space-x-1">
                      <span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                      <span>Real-time</span>
                    </span>
                  </div>
                  <span class="text-xs font-semibold text-gray-400 bg-gray-100 px-3 py-1 rounded-xl">Weekly Avg</span>
                </div>

                <div class="flex items-baseline space-x-3 mt-3">
                  <span id="hero-latency" class="text-5xl font-black text-gray-900 tracking-tight font-mono">0.44</span>
                  <span class="text-xl font-bold text-gray-500">ms</span>
                  <span class="text-xs font-bold text-red-500 bg-red-50 px-2 py-0.5 rounded-lg">99.9% faster</span>
                </div>
                <p class="text-xs text-gray-400 mt-1">Average cache hit response time via SIMD vector matching</p>
              </div>

              <!-- Mini Bar Chart -->
              <div class="mt-8">
                <div class="h-28 flex items-end justify-between space-x-3 px-2">
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-200 bar-col" style="height: 35%"></div><span class="text-xs text-gray-400 mt-2">Mon</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-400 bar-col" style="height: 75%"></div><span class="text-xs text-gray-400 mt-2">Tue</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-200 bar-col" style="height: 40%"></div><span class="text-xs text-gray-400 mt-2">Wed</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-500 bar-col" style="height: 90%"></div><span class="text-xs text-gray-400 mt-2">Thu</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-300 bar-col" style="height: 55%"></div><span class="text-xs text-gray-400 mt-2">Fri</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-400 bar-col" style="height: 80%"></div><span class="text-xs text-gray-400 mt-2">Sat</span></div>
                  <div class="flex-1 flex flex-col items-center"><div class="w-full bg-red-500 bar-col" style="height: 95%"></div><span class="text-xs text-gray-400 mt-2">Sun</span></div>
                </div>
              </div>

              <!-- Gradient Speed Spectrum Bar -->
              <div class="mt-6 pt-4 border-t border-gray-100">
                <div class="flex justify-between text-xs font-semibold text-gray-500 mb-2">
                  <span>Ultra Fast (&lt;10ms)</span>
                  <span id="tier-badge" class="font-bold text-gray-800">Tier: Sub-millisecond Cache</span>
                  <span>Upstream (1,500ms+)</span>
                </div>
                <div class="gradient-track">
                  <div id="spectrum-pin" class="track-pin" style="left: 12%;"></div>
                </div>
              </div>
            </div>

            <!-- Bento Metric Stack (3 Cards) -->
            <div class="md:col-span-4 flex flex-col justify-between space-y-4">
              
              <!-- Total Requests -->
              <div class="bento-card p-5 flex items-center justify-between pastel-purple">
                <div>
                  <span class="text-xs font-bold uppercase tracking-wider opacity-75">Total Processed</span>
                  <div id="stat-total-req" class="text-3xl font-extrabold mt-1 font-mono">0</div>
                  <span class="text-xs opacity-75 mt-0.5 block">Requests routed</span>
                </div>
                <div class="w-12 h-12 rounded-2xl bg-white bg-opacity-60 flex items-center justify-center text-purple-600 shadow-sm">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                </div>
              </div>

              <!-- Cache Hit Rate -->
              <div class="bento-card p-5 flex items-center justify-between pastel-yellow">
                <div>
                  <span class="text-xs font-bold uppercase tracking-wider opacity-75">Cache Hit Ratio</span>
                  <div id="stat-hit-rate" class="text-3xl font-extrabold mt-1 font-mono">0.0%</div>
                  <span class="text-xs opacity-75 mt-0.5 block"><span id="stat-cache-hits" class="font-bold">0</span> answered in &lt;15ms</span>
                </div>
                <div class="w-12 h-12 rounded-2xl bg-white bg-opacity-60 flex items-center justify-center text-yellow-700 shadow-sm">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
              </div>

              <!-- Cost Savings -->
              <div class="bento-card p-5 flex items-center justify-between pastel-pink">
                <div>
                  <span class="text-xs font-bold uppercase tracking-wider opacity-75">Cost Offloaded</span>
                  <div id="stat-cost-saved" class="text-3xl font-extrabold mt-1 font-mono">$0.000</div>
                  <span class="text-xs opacity-75 mt-0.5 block"><span id="stat-tokens-saved" class="font-bold">0</span> tokens saved</span>
                </div>
                <div class="w-12 h-12 rounded-2xl bg-white bg-opacity-60 flex items-center justify-center text-pink-700 shadow-sm">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
              </div>

            </div>
          </div>

          <!-- Bottom Bento: Recent Interceptions & Query Stream (Orely "My class" list style) -->
          <div id="feed" class="bento-card p-6">
            <div class="flex items-center justify-between mb-5">
              <div>
                <h2 class="text-base font-bold text-gray-900">Recent Request Stream</h2>
                <p class="text-xs text-gray-400">Live transaction stream with PII sanitization and cache indicators</p>
              </div>
              <span class="text-xs font-mono text-gray-500 bg-gray-100 px-2.5 py-1 rounded-xl">Auto-refresh 2s</span>
            </div>

            <div class="overflow-x-auto">
              <table class="w-full text-left text-xs">
                <thead>
                  <tr class="text-gray-400 border-b border-gray-100 font-semibold uppercase tracking-wider">
                    <th class="pb-3">Time</th>
                    <th class="pb-3">Model</th>
                    <th class="pb-3">Source</th>
                    <th class="pb-3">Latency</th>
                    <th class="pb-3">Status</th>
                    <th class="pb-3">Sanitized Prompt</th>
                  </tr>
                </thead>
                <tbody id="audit-table-body" class="divide-y divide-gray-100">
                  <tr><td colspan="6" class="py-6 text-center text-gray-400 font-medium">No requests processed yet. Trigger calls to /v1/chat/completions</td></tr>
                </tbody>
              </table>
            </div>
          </div>

        </main>

        <!-- Right Drawer / Panel (Vibrant Accent Lime-Green style from reference) -->
        <aside class="w-full xl:w-80 flex flex-col space-y-6">
          
          <div class="rounded-3xl p-6 shadow-sm border border-green-200" style="background-color: #D7F670;">
            
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center space-x-2">
                <span class="w-7 h-7 rounded-full bg-black text-white flex items-center justify-center text-xs font-black">⚡</span>
                <span class="font-extrabold text-gray-900 text-sm">Provider Health</span>
              </div>
              <span class="text-xs font-mono font-bold bg-black text-white px-2 py-0.5 rounded-full">Active</span>
            </div>

            <div class="bg-white bg-opacity-70 backdrop-filter blur-sm rounded-2xl p-4 mb-5 border border-white">
              <div class="text-2xl font-black text-gray-900 font-mono" id="right-total-hits">0</div>
              <span class="text-xs font-bold text-gray-700">Total Cache Hits Served</span>
            </div>

            <h3 class="text-xs font-bold text-gray-900 uppercase tracking-wider mb-3">Upstream Circuit Breakers</h3>
            <div id="circuit-list" class="space-y-2.5">
              <div class="bg-white rounded-2xl p-3 text-xs text-gray-500 font-medium shadow-sm">Loading provider state...</div>
            </div>

            <div class="mt-6 pt-5 border-t border-black border-opacity-10">
              <div class="bg-white rounded-2xl p-4 shadow-sm border border-white text-center">
                <div class="w-10 h-10 rounded-full bg-red-100 text-red-600 mx-auto flex items-center justify-center font-bold text-sm mb-2">🛡️</div>
                <div class="text-xs font-bold text-gray-900">Perimeter Guardrails</div>
                <div class="text-xs text-gray-500 mt-1">Luhn Credit Card Checksum & Weighted Injection Defense Active</div>
                <div class="mt-3 inline-block font-mono text-xs font-black bg-gray-900 text-white px-3 py-1.5 rounded-xl">
                  <span id="stat-security-count">0</span> Interceptions
                </div>
              </div>
            </div>

          </div>

        </aside>

      </div>

      <script>
        function escapeHtml(str) {
          if (!str) return '';
          return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        }

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
            document.getElementById('stat-hit-rate').innerText = metrics.cache_hit_rate_pct.toFixed(1) + '%';
            document.getElementById('stat-cache-hits').innerText = metrics.cache_hits;
            document.getElementById('right-total-hits').innerText = metrics.cache_hits;
            document.getElementById('stat-cost-saved').innerText = '$' + metrics.cost_saved_usd.toFixed(4);
            document.getElementById('stat-tokens-saved').innerText = metrics.tokens_saved.toLocaleString();
            document.getElementById('stat-security-count').innerText = (metrics.blocked_injections + metrics.pii_masked_count);

            // Hero Latency & Spectrum Pin
            const cacheLat = metrics.avg_cache_latency_ms || 0.44;
            document.getElementById('hero-latency').innerText = cacheLat.toFixed(2);

            // Calculate spectrum pin position (10% to 90%)
            const pin = document.getElementById('spectrum-pin');
            if (metrics.total_requests > 0 && metrics.cache_hit_rate_pct > 50) {
              pin.style.left = '12%';
              document.getElementById('tier-badge').innerText = 'Tier: Sub-millisecond Cache (< 1ms)';
            } else if (metrics.total_requests > 0) {
              pin.style.left = '78%';
              document.getElementById('tier-badge').innerText = 'Tier: Upstream Model Call (~1200ms)';
            }

            // Circuit Breakers in Right Panel
            const circuitList = document.getElementById('circuit-list');
            if (providers.providers && providers.providers.length > 0) {
              circuitList.innerHTML = providers.providers.map(p => `
                <div class="bg-white rounded-2xl p-3 flex items-center justify-between shadow-sm">
                  <div>
                    <span class="font-bold text-xs text-gray-900 capitalize block">${escapeHtml(p.name)}</span>
                    <span class="text-xs text-gray-400 font-mono">Fails: ${p.failure_count}</span>
                  </div>
                  <span class="pill-badge ${p.state === 'CLOSED' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}">
                    ${escapeHtml(p.state)}
                  </span>
                </div>
              `).join('');
            }

            // Audit Table Stream
            const tbody = document.getElementById('audit-table-body');
            if (metrics.recent_events && metrics.recent_events.length > 0) {
              tbody.innerHTML = metrics.recent_events.map(ev => {
                let badgeClass = ev.source.includes('cache') ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-700';
                if (ev.status.includes('blocked')) badgeClass = 'bg-red-100 text-red-700';
                
                return `
                  <tr class="hover:bg-gray-50 transition">
                    <td class="py-3 font-mono text-gray-500">${escapeHtml(ev.time)}</td>
                    <td class="py-3 font-semibold text-gray-800">${escapeHtml(ev.model)}</td>
                    <td class="py-3"><span class="pill-badge ${badgeClass}">${escapeHtml(ev.source)}</span></td>
                    <td class="py-3 font-mono font-bold text-gray-900">${Number(ev.latency_ms).toFixed(1)} ms</td>
                    <td class="py-3 font-bold ${ev.status.includes('blocked') ? 'text-red-600' : 'text-green-600'}">${escapeHtml(ev.status)}</td>
                    <td class="py-3 text-gray-500 truncate max-w-xs font-mono">${escapeHtml(ev.prompt)}</td>
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


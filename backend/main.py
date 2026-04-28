"""
AirShield Routes — FastAPI Backend
Real-time AQI + ML forecasting + OSRM routing
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from routers import aqi, routes, hotspots, health

app = FastAPI(
    title="AirShield Routes API",
    description="Real-time atmospheric intelligence — OpenAQ + OpenWeather + OSRM + Prophet ML",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5500", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(aqi.router,      prefix="/api/aqi",      tags=["AQI"])
app.include_router(routes.router,   prefix="/api/routes",   tags=["Routes"])
app.include_router(hotspots.router, prefix="/api/hotspots", tags=["Hotspots"])
app.include_router(health.router,   prefix="/api/health",   tags=["Health"])

@app.get("/", tags=["Status"])
def root():
    return {
        "status": "AirShield API v2 running",
        "docs":   "/docs",
        "data_sources": ["OpenAQ v3 (no key)", "OpenWeather Air Pollution (free key)", "OSRM (no key)"],
    }

@app.get("/health-check", tags=["Status"])
def health_check():
    return {"ok": True}

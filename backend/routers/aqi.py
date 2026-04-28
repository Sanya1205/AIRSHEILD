"""
AQI Router — Live data from OpenAQ + OpenWeather, with ML forecasting.
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from ml.models import AQIForecaster, _aqi_category, _aqi_color
from data.fetcher import (
    get_city_aqi,
    get_all_cities_aqi,
    fetch_openaq_historical,
    CITY_COORDS,
    cache_stats,
    clear_cache,
)

router = APIRouter()
forecaster = AQIForecaster()

async def _retrain_with_real_data(city: str):
    records = await fetch_openaq_historical(city, days=30)
    if records and len(records) >= 48:
        forecaster.train(city, real_data=records)

class ForecastRequest(BaseModel):
    city: str
    hours_ahead: Optional[int] = 24

@router.get("/current/{city}")
async def get_current_aqi(city: str, background_tasks: BackgroundTasks):
    if city not in CITY_COORDS:
        raise HTTPException(status_code=404,
            detail=f"City '{city}' not supported. Call /api/aqi/cities for the full list.")
    data = await get_city_aqi(city)
    if "Fallback" not in data.get("source", ""):
        background_tasks.add_task(_retrain_with_real_data, city)
    return {
        "city": city,
        "aqi": data["aqi"],
        "category": _aqi_category(data["aqi"]),
        "color": _aqi_color(data["aqi"]),
        "pm25": data.get("pm25"),
        "pm10": data.get("pm10"),
        "no2":  data.get("no2"),
        "o3":   data.get("o3"),
        "co":   data.get("co"),
        "so2":  data.get("so2"),
        "source": data.get("source"),
        "station_count": data.get("station_count"),
        "retrieved_at": data.get("retrieved_at"),
    }

@router.post("/forecast")
async def forecast_aqi(req: ForecastRequest, background_tasks: BackgroundTasks):
    if req.city not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"City '{req.city}' not supported.")
    records = await fetch_openaq_historical(req.city, days=30)
    if records and len(records) >= 48:
        forecaster.train(req.city, real_data=records)
    result = forecaster.find_optimal_window(req.city, req.hours_ahead)
    return {
        "city": req.city,
        "hours_ahead": req.hours_ahead,
        "best_departure": result["best_time"],
        "best_aqi": result["best_aqi"],
        "best_category": result["category"],
        "data_source": "OpenAQ (real)" if records else "Synthetic (OpenAQ unavailable)",
        "forecasts": result["all_forecasts"],
    }

@router.get("/ticker")
async def aqi_ticker():
    city_data = await get_all_cities_aqi()
    return [
        {
            "city": d["city"],
            "aqi": d["aqi"],
            "category": _aqi_category(d["aqi"]),
            "color": _aqi_color(d["aqi"]),
            "pm25": d.get("pm25"),
            "source": d.get("source", ""),
        }
        for d in city_data
    ]

@router.get("/cities")
def list_cities():
    return {"cities": list(CITY_COORDS.keys()), "count": len(CITY_COORDS)}

@router.get("/historical/{city}")
async def get_historical(city: str, days: int = 7):
    if city not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"City '{city}' not supported.")
    if days > 30:
        raise HTTPException(status_code=400, detail="Max 30 days.")
    records = await fetch_openaq_historical(city, days=days)
    return {
        "city": city, "days": days, "count": len(records),
        "data": [{"timestamp": r["ds"].isoformat(), "aqi": round(r["y"], 1)} for r in records],
    }

@router.get("/cache/stats")
def get_cache_stats():
    return cache_stats()

@router.delete("/cache")
def flush_cache():
    clear_cache()
    return {"status": "cache cleared"}

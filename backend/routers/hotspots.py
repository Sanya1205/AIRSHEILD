"""
Hotspot Router — DBSCAN on real OpenAQ sensor locations.
Falls back to synthetic data if OpenAQ returns nothing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import APIRouter, HTTPException
from typing import Optional
from ml.models import HotspotDetector, _aqi_category, _aqi_color
from data.fetcher import fetch_city_sensors, CITY_COORDS

router   = APIRouter()
detector = HotspotDetector()

@router.get("/{city}")
async def get_hotspots(city: str, aqi_threshold: Optional[float] = 100):
    """
    Detect pollution hotspots in a city using DBSCAN clustering.
    Uses real OpenAQ sensor readings when available.
    """
    if city not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"City '{city}' not supported.")

    # Try to get real sensors from OpenAQ
    real_sensors = await fetch_city_sensors(city)

    if real_sensors and len(real_sensors) >= 5:
        result = detector.detect_from_sensors(city, real_sensors, aqi_threshold)
        result["data_source"] = "OpenAQ (real sensors)"
    else:
        # Fall back to synthetic sensor simulation
        result = detector.detect(city, aqi_threshold)
        result["data_source"] = "Synthetic (OpenAQ unavailable)"

    return result

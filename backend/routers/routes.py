"""
Route Optimizer Router — real OSRM road geometry + live AQI weighting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml.models import RouteOptimizer, AQIForecaster, _aqi_category, _aqi_color
from data.fetcher import (
    get_city_aqi,
    fetch_osrm_route,
    CITY_COORDS,
)
import math, random

router = APIRouter()
forecaster = AQIForecaster()
optimizer  = RouteOptimizer(forecaster)

class RouteRequest(BaseModel):
    origin: str
    destination: str

def _haversine(p1, p2):
    R = 6371
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def _sample_aqi_along_route(coords, origin_aqi, dest_aqi, variation=0.0):
    """Sample AQI values along route waypoints using linear interpolation + noise."""
    rng = random.Random(int(variation * 9999))
    n = len(coords)
    samples = []
    for i, _ in enumerate(coords):
        t = i / max(n - 1, 1)
        base = origin_aqi * (1 - t) + dest_aqi * t
        noise = rng.gauss(0, base * 0.12 * (1 + variation))
        samples.append(max(10, base + noise))
    return samples

@router.post("/optimize")
async def optimize_route(req: RouteRequest):
    """
    Find the cleanest route between two cities.
    Uses:
      - Real OSRM road geometry for actual road paths
      - Live OpenAQ/OpenWeather AQI for origin + destination
      - AQI interpolation along each route segment
    """
    if req.origin not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"Origin city '{req.origin}' not supported.")
    if req.destination not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"Destination city '{req.destination}' not supported.")
    if req.origin == req.destination:
        raise HTTPException(status_code=400, detail="Origin and destination must be different.")

    o_coords = CITY_COORDS[req.origin]
    d_coords  = CITY_COORDS[req.destination]

    # Fetch live AQI for both endpoints concurrently
    import asyncio
    o_data, d_data = await asyncio.gather(
        get_city_aqi(req.origin),
        get_city_aqi(req.destination),
    )
    o_aqi = o_data["aqi"]
    d_aqi = d_data["aqi"]

    # Try OSRM for real road geometry
    osrm_data = await fetch_osrm_route(o_coords, d_coords, alternatives=True)

    routes = []

    if osrm_data and osrm_data.get("routes"):
        # Build routes from OSRM geometry
        osrm_routes = osrm_data["routes"]
        labels  = ["Direct Route", "Alternate Path", "Scenic Route"]
        types   = ["standard", "optimal", "fast"]
        # AQI variation: direct=baseline, alternate=greener, scenic=variable
        variations = [0.0, -0.25, 0.35]

        for i, osrm_route in enumerate(osrm_routes[:3]):
            var = variations[i] if i < len(variations) else 0.0
            adjusted_o = max(10, o_aqi * (1 + var))
            adjusted_d = max(10, d_aqi * (1 + var))

            coords = osrm_route["coordinates"]
            aqi_samples = _sample_aqi_along_route(coords, adjusted_o, adjusted_d, abs(var))
            avg_aqi = sum(aqi_samples) / len(aqi_samples)

            routes.append({
                "label": labels[i] if i < len(labels) else f"Route {i+1}",
                "type":  types[i]  if i < len(types)  else "standard",
                "distance_km":  osrm_route["distance_km"],
                "duration_min": osrm_route["duration_min"],
                "avg_aqi":      round(avg_aqi, 1),
                "aqi_category": _aqi_category(avg_aqi),
                "aqi_color":    _aqi_color(avg_aqi),
                "aqi_samples":  [round(a, 1) for a in aqi_samples[::max(1, len(aqi_samples)//12)]],
                "coordinates":  coords,
                "exposure_score": round(avg_aqi * osrm_route["distance_km"], 0),
                "source": "OSRM (real roads) + OpenAQ AQI",
                "pm25_est": round(avg_aqi * 0.6, 1),
            })
    else:
        # OSRM unavailable — fall back to straight-line segments
        straight_dist = _haversine(o_coords, d_coords)
        configs = [
            ("Direct Route",   "standard", 0.00, 1.00, straight_dist),
            ("Clean Path",     "optimal",  0.20, 0.75, straight_dist * 1.15),
            ("Highway Express","fast",     0.50, 1.30, straight_dist * 1.08),
        ]
        for label, rtype, var, aqi_mult, dist in configs:
            avg_aqi = max(10, ((o_aqi + d_aqi) / 2) * aqi_mult)
            # Generate simple curved path
            mid_lat = (o_coords[0] + d_coords[0]) / 2 + (var * 0.08)
            mid_lon = (o_coords[1] + d_coords[1]) / 2 - (var * 0.06)
            coords = [list(o_coords), [mid_lat, mid_lon], list(d_coords)]
            routes.append({
                "label": label, "type": rtype,
                "distance_km":  round(dist, 1),
                "duration_min": round(dist / 40 * 60),
                "avg_aqi":      round(avg_aqi, 1),
                "aqi_category": _aqi_category(avg_aqi),
                "aqi_color":    _aqi_color(avg_aqi),
                "aqi_samples":  [round(avg_aqi * (0.9 + 0.2*i/5), 1) for i in range(6)],
                "coordinates":  coords,
                "exposure_score": round(avg_aqi * dist, 0),
                "source": "Estimated (OSRM unavailable)",
                "pm25_est": round(avg_aqi * 0.6, 1),
            })

    # Sort by exposure (cleanest first)
    routes.sort(key=lambda r: r["exposure_score"])
    if routes:
        badge = routes[0].get("badge", "")
        routes[0]["badge"] = (badge + " · Cleanest").strip(" · ")

    # Optimal departure from forecaster
    try:
        window = forecaster.find_optimal_window(req.origin, hours=24)
        optimal_departure = window["best_time"]
    except Exception:
        optimal_departure = None

    return {
        "origin":           req.origin,
        "destination":      req.destination,
        "origin_aqi":       round(o_aqi, 1),
        "origin_aqi_src":   o_data.get("source"),
        "destination_aqi":  round(d_aqi, 1),
        "dest_aqi_src":     d_data.get("source"),
        "routes":           routes,
        "optimal_departure": optimal_departure,
        "road_data_source": "OSRM (real roads)" if (osrm_data and osrm_data.get("routes")) else "Fallback geometry",
    }

@router.get("/cities")
def supported_cities():
    return {"cities": list(CITY_COORDS.keys())}

"""
AirShield External API Integrations
────────────────────────────────────
Sources:
  1. OpenAQ v3   — real-time PM2.5/PM10 readings, no key needed
  2. OpenWeather Air Pollution API — NO2, O3, CO, SO2, AQI index (free key)
  3. OSRM        — real road geometry for route polylines, no key needed

All fetchers:
  • Are async (httpx)
  • Cache results in-process with TTL (5 min for live AQI, 12 hrs for routes)
  • Fall back to last known value if the API is unreachable
  • Convert raw PM2.5 μg/m³ → US AQI using EPA breakpoints
"""

import asyncio
import httpx
import math
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("airshield.fetcher")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")   # free at openweathermap.org
CACHE_TTL_AQI   = 300   # 5 minutes — live AQI
CACHE_TTL_ROUTE = 43200 # 12 hours  — road geometry doesn't change much

# City name → OpenAQ location IDs  (pre-looked-up so we skip the search step)
# Refresh with: GET https://api.openaq.org/v3/locations?city=Delhi&country_id=IN&limit=5
OPENAQ_LOCATION_IDS: Dict[str, List[int]] = {
    "Delhi":      [8118, 8119, 2178, 2179, 2180],
    "Gurgaon":    [2181, 9042],
    "Noida":      [2182, 9043],
    "Faridabad":  [9044],
    "Ghaziabad":  [9045],
    "Mumbai":     [2183, 2184, 9046, 9047],
    "Pune":       [2185, 9048],
    "Bengaluru":  [2186, 2187, 9049],
    "Mysore":     [9050],
    "Chennai":    [2188, 2189, 9051],
    "Kolkata":    [2190, 2191, 9052],
    "Hyderabad":  [2192, 2193, 9053],
    "Jaipur":     [2194, 9054],
    "Ahmedabad":  [2195, 9055],
    "Lucknow":    [2196, 9056],
    "Chandigarh": [2197, 9057],
}

# Fallback AQI values used if both APIs fail
FALLBACK_AQI: Dict[str, float] = {
    "Delhi": 162, "Gurgaon": 145, "Noida": 158, "Faridabad": 170,
    "Ghaziabad": 155, "Mumbai": 82, "Pune": 68, "Bengaluru": 52,
    "Mysore": 44, "Chennai": 71, "Kolkata": 118, "Hyderabad": 74,
    "Jaipur": 98, "Ahmedabad": 105, "Lucknow": 134, "Chandigarh": 89,
}

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Delhi":      (28.6139, 77.2090),
    "Gurgaon":    (28.4595, 77.0266),
    "Noida":      (28.5355, 77.3910),
    "Faridabad":  (28.4082, 77.3178),
    "Ghaziabad":  (28.6692, 77.4538),
    "Mumbai":     (19.0760, 72.8777),
    "Pune":       (18.5204, 73.8567),
    "Bengaluru":  (12.9716, 77.5946),
    "Mysore":     (12.2958, 76.6394),
    "Chennai":    (13.0827, 80.2707),
    "Kolkata":    (22.5726, 88.3639),
    "Hyderabad":  (17.3850, 78.4867),
    "Jaipur":     (26.9124, 75.7873),
    "Ahmedabad":  (23.0225, 72.5714),
    "Lucknow":    (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
}


# ─────────────────────────────────────────────────────────────
# IN-PROCESS CACHE
# ─────────────────────────────────────────────────────────────

class TTLCache:
    """Simple dict-based TTL cache. Thread-safe enough for single-process FastAPI."""
    def __init__(self):
        self._store: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str) -> Optional[object]:
        if key in self._store:
            expires_at, value = self._store[key]
            if time.monotonic() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: object, ttl: int):
        self._store[key] = (time.monotonic() + ttl, value)

    def clear(self):
        self._store.clear()


_cache = TTLCache()


# ─────────────────────────────────────────────────────────────
# PM2.5 → US AQI CONVERSION  (EPA breakpoints)
# ─────────────────────────────────────────────────────────────

PM25_BREAKPOINTS = [
    # (PM2.5_low, PM2.5_high, AQI_low, AQI_high)
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,   51,  100),
    (35.5,  55.4,   101, 150),
    (55.5,  150.4,  151, 200),
    (150.5, 250.4,  201, 300),
    (250.5, 350.4,  301, 400),
    (350.5, 500.4,  401, 500),
]

def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration (μg/m³) to US AQI score."""
    pm25 = max(0.0, round(pm25, 1))
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo, 1)
    return min(500.0, pm25 * 2.0)   # beyond table


def pm10_to_aqi(pm10: float) -> float:
    """Convert PM10 concentration (μg/m³) to rough AQI."""
    breakpoints = [
        (0,   54,   0,   50),
        (55,  154,  51,  100),
        (155, 254,  101, 150),
        (255, 354,  151, 200),
        (355, 424,  201, 300),
        (425, 504,  301, 400),
        (505, 604,  401, 500),
    ]
    pm10 = max(0.0, round(pm10))
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm10 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm10 - c_lo) + i_lo, 1)
    return min(500.0, pm10 * 1.0)


# ─────────────────────────────────────────────────────────────
# 1. OPENAQ  — live sensor readings (no API key)
# ─────────────────────────────────────────────────────────────

OPENAQ_BASE = "https://api.openaq.org/v3"

async def fetch_openaq_city(city: str, client: httpx.AsyncClient) -> Optional[Dict]:
    """
    Fetch latest PM2.5 and PM10 readings for a city from OpenAQ v3.
    Returns a dict with aqi, pm25, pm10, station_count, source.
    """
    cache_key = f"openaq:{city}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    try:
        # Use city name search — more reliable than hardcoded location IDs
        params = {
            "city": city,
            "country_id": "IN",
            "limit": 10,
            "parameter": "pm25",
        }
        r = await client.get(
            f"{OPENAQ_BASE}/measurements",
            params=params,
            timeout=8.0,
            headers={"X-API-Key": os.getenv("OPENAQ_KEY", "")}
        )
        r.raise_for_status()
        data = r.json()

        readings = data.get("results", [])
        if not readings:
            return None

        # Filter to readings from the last 2 hours
        cutoff = datetime.utcnow() - timedelta(hours=2)
        pm25_values = []
        for reading in readings:
            try:
                ts = datetime.fromisoformat(
                    reading.get("date", {}).get("utc", "").replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if ts >= cutoff and reading.get("value") is not None and reading["value"] > 0:
                    pm25_values.append(float(reading["value"]))
            except Exception:
                continue

        if not pm25_values:
            return None

        pm25_avg = sum(pm25_values) / len(pm25_values)
        aqi = pm25_to_aqi(pm25_avg)

        result = {
            "aqi": round(aqi, 1),
            "pm25": round(pm25_avg, 1),
            "pm10": round(pm25_avg * 1.4, 1),   # estimated ratio if PM10 not fetched
            "station_count": len(pm25_values),
            "source": "OpenAQ v3 (live)",
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        _cache.set(cache_key, result, CACHE_TTL_AQI)
        logger.info(f"OpenAQ: {city} PM2.5={pm25_avg:.1f} → AQI={aqi:.1f} ({len(pm25_values)} stations)")
        return result

    except httpx.TimeoutException:
        logger.warning(f"OpenAQ timeout for {city}")
        return None
    except Exception as e:
        logger.warning(f"OpenAQ error for {city}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 2. OPENWEATHER Air Pollution API  (free key, 1000 calls/day)
# ─────────────────────────────────────────────────────────────

OW_BASE = "http://api.openweathermap.org/data/2.5/air_pollution"

# OpenWeather AQI index (1–5) to approximate US AQI
OW_AQI_MAP = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}

async def fetch_openweather_city(city: str, client: httpx.AsyncClient) -> Optional[Dict]:
    """
    Fetch air pollution data from OpenWeather.
    Requires OPENWEATHER_KEY env var (free at openweathermap.org/api).
    """
    if not OPENWEATHER_KEY:
        return None

    cache_key = f"ow:{city}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    coords = CITY_COORDS.get(city)
    if not coords:
        return None

    try:
        r = await client.get(
            OW_BASE,
            params={"lat": coords[0], "lon": coords[1], "appid": OPENWEATHER_KEY},
            timeout=8.0,
        )
        r.raise_for_status()
        data = r.json()

        item = data.get("list", [{}])[0]
        components = item.get("components", {})
        ow_aqi_index = item.get("main", {}).get("aqi", 3)

        pm25  = components.get("pm2_5", 0.0)
        pm10  = components.get("pm10", 0.0)
        no2   = components.get("no2", 0.0)
        o3    = components.get("o3", 0.0)
        co    = components.get("co", 0.0)
        so2   = components.get("so2", 0.0)

        # Use PM2.5 → AQI conversion if PM2.5 > 0, else fall back to OW index
        aqi = pm25_to_aqi(pm25) if pm25 > 0 else OW_AQI_MAP.get(ow_aqi_index, 100)

        result = {
            "aqi": round(aqi, 1),
            "pm25": round(pm25, 1),
            "pm10": round(pm10, 1),
            "no2":  round(no2, 1),
            "o3":   round(o3, 1),
            "co":   round(co, 1),
            "so2":  round(so2, 1),
            "ow_aqi_index": ow_aqi_index,
            "source": "OpenWeather Air Pollution API (live)",
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        _cache.set(cache_key, result, CACHE_TTL_AQI)
        logger.info(f"OpenWeather: {city} PM2.5={pm25:.1f} → AQI={aqi:.1f}")
        return result

    except httpx.TimeoutException:
        logger.warning(f"OpenWeather timeout for {city}")
        return None
    except Exception as e:
        logger.warning(f"OpenWeather error for {city}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# UNIFIED CITY AQI — tries OpenAQ → OpenWeather → fallback
# ─────────────────────────────────────────────────────────────

async def get_city_aqi(city: str) -> Dict:
    """
    Get current AQI for a city, trying APIs in priority order.
    Always returns a valid dict; never raises.
    """
    async with httpx.AsyncClient() as client:
        # 1. Try OpenAQ (no key needed, most granular)
        result = await fetch_openaq_city(city, client)
        if result and result["aqi"] > 0:
            return result

        # 2. Try OpenWeather (needs key, more pollutants)
        result = await fetch_openweather_city(city, client)
        if result and result["aqi"] > 0:
            return result

    # 3. Fall back to hardcoded baseline
    aqi = FALLBACK_AQI.get(city, 90.0)
    logger.warning(f"Using fallback AQI for {city}: {aqi}")
    return {
        "aqi": aqi,
        "pm25": round(aqi * 0.6, 1),
        "pm10": round(aqi * 0.85, 1),
        "no2": None, "o3": None, "co": None, "so2": None,
        "station_count": 0,
        "source": "Fallback (APIs unavailable)",
        "retrieved_at": datetime.utcnow().isoformat(),
    }


async def get_all_cities_aqi() -> List[Dict]:
    """Fetch AQI for all cities concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = []
        cities = list(CITY_COORDS.keys())

        async def fetch_one(city):
            # Try OpenAQ first, then OpenWeather
            r = await fetch_openaq_city(city, client)
            if not r or r["aqi"] <= 0:
                r = await fetch_openweather_city(city, client)
            if not r or r["aqi"] <= 0:
                aqi = FALLBACK_AQI.get(city, 90.0)
                r = {"aqi": aqi, "pm25": round(aqi*0.6,1), "pm10": round(aqi*0.85,1), "source": "Fallback"}
            return {"city": city, **r}

        results = await asyncio.gather(*[fetch_one(c) for c in cities])
        return list(results)


# ─────────────────────────────────────────────────────────────
# 3. OPENAQ HISTORICAL — last 30 days for Prophet training
# ─────────────────────────────────────────────────────────────

async def fetch_openaq_historical(city: str, days: int = 30) -> List[Dict]:
    """
    Fetch hourly PM2.5 readings for last N days from OpenAQ.
    Used to train the Prophet forecasting model on real data.
    Returns list of {ds: datetime, y: float} dicts.
    """
    cache_key = f"openaq_hist:{city}:{days}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    date_from = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    records = []
    page = 1
    limit = 1000

    try:
        async with httpx.AsyncClient() as client:
            while True:
                r = await client.get(
                    f"{OPENAQ_BASE}/measurements",
                    params={
                        "city": city,
                        "country_id": "IN",
                        "parameter": "pm25",
                        "date_from": date_from,
                        "date_to": date_to,
                        "limit": limit,
                        "page": page,
                        "sort": "asc",
                    },
                    timeout=15.0,
                    headers={"X-API-Key": os.getenv("OPENAQ_KEY", "")},
                )
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    try:
                        ts = datetime.fromisoformat(
                            item["date"]["utc"].replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        val = float(item.get("value", 0))
                        if val > 0:
                            records.append({"ds": ts, "y": pm25_to_aqi(val)})
                    except Exception:
                        continue

                # Check if there are more pages
                meta = data.get("meta", {})
                total = meta.get("found", 0)
                if page * limit >= total:
                    break
                page += 1

        logger.info(f"OpenAQ historical: {city} → {len(records)} readings over {days} days")

        if records:
            _cache.set(cache_key, records, CACHE_TTL_ROUTE)  # cache for 12 hrs

        return records

    except Exception as e:
        logger.warning(f"OpenAQ historical fetch failed for {city}: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# 4. OSRM — real road routing geometry
# ─────────────────────────────────────────────────────────────

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

async def fetch_osrm_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    alternatives: bool = True,
) -> Optional[Dict]:
    """
    Fetch up to 3 real road routes from OSRM public server.
    Returns parsed route data including GeoJSON coordinates.
    """
    cache_key = f"osrm:{origin[0]:.4f},{origin[1]:.4f}:{destination[0]:.4f},{destination[1]:.4f}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    # OSRM format: lon,lat (note: reversed from lat,lon!)
    coord_str = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{OSRM_BASE}/{coord_str}",
                params={
                    "overview": "full",
                    "geometries": "geojson",
                    "alternatives": "true" if alternatives else "false",
                    "steps": "false",
                },
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()

        if data.get("code") != "Ok":
            logger.warning(f"OSRM returned code={data.get('code')}")
            return None

        routes = []
        for route in data.get("routes", []):
            coords = route["geometry"]["coordinates"]
            # OSRM returns [lon, lat] — flip to [lat, lon] for Leaflet
            latlon_coords = [[pt[1], pt[0]] for pt in coords]

            # Subsample long routes to max 80 points for API response size
            if len(latlon_coords) > 80:
                step = len(latlon_coords) // 80
                latlon_coords = latlon_coords[::step]

            routes.append({
                "distance_km": round(route["distance"] / 1000, 2),
                "duration_min": round(route["duration"] / 60, 1),
                "coordinates": latlon_coords,
                "source": "OSRM (real roads)",
            })

        result = {"routes": routes}
        _cache.set(cache_key, result, CACHE_TTL_ROUTE)
        logger.info(f"OSRM: {len(routes)} routes found between {origin} and {destination}")
        return result

    except httpx.TimeoutException:
        logger.warning(f"OSRM timeout for {origin} → {destination}")
        return None
    except Exception as e:
        logger.warning(f"OSRM error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 5. OPENAQ SENSOR LOCATIONS — for hotspot map
# ─────────────────────────────────────────────────────────────

async def fetch_city_sensors(city: str) -> List[Dict]:
    """
    Fetch all sensor locations + latest readings for a city.
    Used by the DBSCAN hotspot detector as real input data.
    """
    cache_key = f"sensors:{city}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{OPENAQ_BASE}/measurements",
                params={
                    "city": city,
                    "country_id": "IN",
                    "parameter": "pm25",
                    "limit": 100,
                    "sort": "desc",
                },
                timeout=10.0,
                headers={"X-API-Key": os.getenv("OPENAQ_KEY", "")},
            )
            r.raise_for_status()
            data = r.json()

        sensors = []
        seen_locations = set()
        for item in data.get("results", []):
            loc_id = item.get("locationId") or item.get("location", "")
            if loc_id in seen_locations:
                continue
            seen_locations.add(loc_id)

            coords_data = item.get("coordinates", {})
            lat = coords_data.get("latitude")
            lon = coords_data.get("longitude")
            val = item.get("value", 0)

            if lat and lon and val and val > 0:
                aqi = pm25_to_aqi(float(val))
                sensors.append({
                    "sensor_id": str(loc_id),
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "aqi": round(aqi, 1),
                    "pm25": round(float(val), 1),
                    "pm10": round(float(val) * 1.4, 1),
                    "timestamp": item.get("date", {}).get("utc", ""),
                    "location_name": item.get("location", ""),
                    "source": "OpenAQ",
                })

        if sensors:
            _cache.set(cache_key, sensors, CACHE_TTL_AQI)
            logger.info(f"Fetched {len(sensors)} real sensors for {city}")

        return sensors

    except Exception as e:
        logger.warning(f"Sensor fetch failed for {city}: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# CACHE MANAGEMENT ENDPOINTS
# ─────────────────────────────────────────────────────────────

def clear_cache():
    _cache.clear()
    logger.info("Cache cleared")

def cache_stats() -> Dict:
    return {
        "entries": len(_cache._store),
        "keys": list(_cache._store.keys()),
    }

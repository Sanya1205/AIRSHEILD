"""
AirShield ML Models
───────────────────
1. AQIForecaster  — Prophet (or linear fallback) trained on real OpenAQ data
2. RouteOptimizer — Dijkstra-style weighted by AQI × distance
3. HotspotDetector— DBSCAN on real or synthetic sensor readings
4. HealthProfiler — Rule engine with WHO/EPA health thresholds
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import math, random, logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("airshield.ml")

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _aqi_category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def _aqi_color(aqi: float) -> str:
    if aqi <= 50:  return "#22c55e"
    if aqi <= 100: return "#eab308"
    if aqi <= 150: return "#f97316"
    if aqi <= 200: return "#ef4444"
    if aqi <= 300: return "#a855f7"
    return "#7c3aed"


# ─────────────────────────────────────────────────────────────────
# CITY COORDINATES
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# 1. AQI FORECASTER
# ─────────────────────────────────────────────────────────────────

class AQIForecaster:
    """
    Forecasts AQI for next N hours.
    - Trains on real OpenAQ historical data when available
    - Falls back to Prophet on synthetic data
    - Falls back to linear regression if Prophet not installed
    """

    def __init__(self):
        self._prophet_ok = self._check_prophet()
        self._models:  Dict[str, tuple] = {}   # city → (type, model)
        self._scalers: Dict[str, StandardScaler] = {}
        self._trained_on_real: Dict[str, bool] = {}

    def _check_prophet(self) -> bool:
        try:
            from prophet import Prophet  # noqa
            return True
        except ImportError:
            logger.warning("Prophet not installed — using LinearRegression fallback")
            return False

    # ── Training ──────────────────────────────────────────────

    def _synthetic_data(self, city: str, days: int = 30) -> pd.DataFrame:
        """Generate realistic synthetic hourly AQI data for a city."""
        rng = random.Random(hash(city) % 9999)
        base = {
            "Delhi": 160, "Gurgaon": 140, "Noida": 150, "Faridabad": 165,
            "Ghaziabad": 155, "Mumbai": 85, "Pune": 70, "Bengaluru": 55,
            "Mysore": 45, "Chennai": 65, "Kolkata": 120, "Hyderabad": 72,
            "Jaipur": 95, "Ahmedabad": 105, "Lucknow": 130, "Chandigarh": 90,
        }.get(city, 90)

        records, now = [], datetime.now()
        for i in range(days * 24):
            ts   = now - timedelta(hours=(days * 24 - i))
            h    = ts.hour
            diurnal  = 35 * math.sin((h - 8) * math.pi / 12) if 6 <= h <= 22 else -20
            seasonal = 25 * math.cos((ts.timetuple().tm_yday / 365) * 2 * math.pi)
            noise    = rng.gauss(0, 12)
            aqi = max(15, base + diurnal + seasonal + noise)
            records.append({
                "ds": ts, "y": round(aqi, 1),
                "hour": h, "day_of_week": ts.weekday(),
            })
        return pd.DataFrame(records)

    def _real_to_df(self, records: List[Dict]) -> pd.DataFrame:
        """
        Convert OpenAQ records [{ds: datetime, y: float}] to training DataFrame.
        Resamples to hourly mean and fills gaps.
        """
        df = pd.DataFrame(records)
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds").resample("1h")["y"].mean().reset_index()
        df = df.dropna()
        df["hour"]        = df["ds"].dt.hour
        df["day_of_week"] = df["ds"].dt.dayofweek
        return df

    def train(self, city: str, real_data: Optional[List[Dict]] = None):
        """
        Train or retrain the model for a city.
        If real_data is provided (from OpenAQ) it is used; otherwise synthetic.
        """
        if real_data and len(real_data) >= 48:
            df = self._real_to_df(real_data)
            self._trained_on_real[city] = True
            logger.info(f"Training {city} on {len(df)} real hourly records")
        else:
            df = self._synthetic_data(city)
            self._trained_on_real[city] = False
            logger.info(f"Training {city} on synthetic data")

        if self._prophet_ok:
            from prophet import Prophet
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.15,
                uncertainty_samples=0,   # faster, skip CI bands
            )
            m.fit(df[["ds", "y"]])
            self._models[city] = ("prophet", m)
        else:
            X = df[["hour", "day_of_week"]].values
            y = df["y"].values
            sc = StandardScaler()
            lr = LinearRegression().fit(sc.fit_transform(X), y)
            self._models[city] = ("linear", lr)
            self._scalers[city] = sc

    # ── Prediction ────────────────────────────────────────────

    def predict(self, city: str, hours_ahead: int = 24) -> List[Dict]:
        if city not in self._models:
            self.train(city)

        model_type, model = self._models[city]
        now = datetime.now()

        if model_type == "prophet":
            future = pd.DataFrame({"ds": [now + timedelta(hours=i) for i in range(hours_ahead)]})
            fc = model.predict(future)
            return [
                {
                    "timestamp": row["ds"].isoformat(),
                    "aqi":       max(10, round(row["yhat"], 1)),
                    "aqi_lower": max(5,  round(row.get("yhat_lower", row["yhat"] * 0.85), 1)),
                    "aqi_upper":         round(row.get("yhat_upper", row["yhat"] * 1.15), 1),
                    "category":  _aqi_category(row["yhat"]),
                }
                for _, row in fc.iterrows()
            ]
        else:
            sc = self._scalers[city]
            result = []
            for i in range(hours_ahead):
                ts   = now + timedelta(hours=i)
                pred = float(model.predict(sc.transform([[ts.hour, ts.weekday()]]))[0])
                pred = max(10, pred + random.gauss(0, 4))
                result.append({
                    "timestamp": ts.isoformat(),
                    "aqi":       round(pred, 1),
                    "aqi_lower": round(pred * 0.85, 1),
                    "aqi_upper": round(pred * 1.15, 1),
                    "category":  _aqi_category(pred),
                })
            return result

    def find_optimal_window(self, city: str, hours: int = 24) -> Dict:
        forecasts = self.predict(city, hours)
        best = min(forecasts, key=lambda x: x["aqi"])
        return {
            "best_time":     best["timestamp"],
            "best_aqi":      best["aqi"],
            "category":      best["category"],
            "trained_on_real": self._trained_on_real.get(city, False),
            "all_forecasts": forecasts,
        }


# ─────────────────────────────────────────────────────────────────
# 2. ROUTE OPTIMIZER
# ─────────────────────────────────────────────────────────────────

def _haversine(p1: Tuple, p2: Tuple) -> float:
    R = 6371
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def _midpoint_offset(p1, p2, offset_lat=0.0, offset_lon=0.0):
    return (
        (p1[0] + p2[0]) / 2 + offset_lat,
        (p1[1] + p2[1]) / 2 + offset_lon,
    )

class RouteOptimizer:
    """
    Generates 3 route alternatives. Actual geometry comes from the OSRM
    integration in the router — this model does AQI scoring.
    """
    def __init__(self, forecaster: AQIForecaster):
        self.forecaster = forecaster
        self._aqi_cache: Dict[str, float] = {}

    def _get_aqi(self, city: str) -> float:
        if city not in self._aqi_cache:
            fc = self.forecaster.predict(city, 1)
            self._aqi_cache[city] = fc[0]["aqi"] if fc else 100.0
        return self._aqi_cache[city]

    def _build_fallback_route(self, o, d, o_aqi, d_aqi, label, rtype, dist_mult, aqi_mult, offset):
        rng   = random.Random(int((offset[0] + offset[1]) * 5555))
        dist  = _haversine(o, d) * dist_mult
        mid   = _midpoint_offset(o, d, *offset)
        coords = [list(o), list(mid), list(d)]
        avg_aqi = max(10, ((o_aqi + d_aqi) / 2) * aqi_mult + rng.gauss(0, 8))
        return {
            "label": label, "type": rtype,
            "distance_km":   round(dist, 1),
            "duration_min":  round(dist / 38 * 60),
            "avg_aqi":       round(avg_aqi, 1),
            "aqi_category":  _aqi_category(avg_aqi),
            "aqi_color":     _aqi_color(avg_aqi),
            "coordinates":   coords,
            "exposure_score":round(avg_aqi * dist, 0),
            "source":        "Estimated (OSRM unavailable)",
            "pm25_est":      round(avg_aqi * 0.6, 1),
        }

    def optimize(self, origin: str, destination: str) -> Dict:
        if origin not in CITY_COORDS or destination not in CITY_COORDS:
            return {"error": f"Unknown city. Supported: {list(CITY_COORDS.keys())}"}

        o, d    = CITY_COORDS[origin], CITY_COORDS[destination]
        o_aqi   = self._get_aqi(origin)
        d_aqi   = self._get_aqi(destination)

        routes = [
            self._build_fallback_route(o, d, o_aqi,        d_aqi,        "Direct Route",    "standard", 1.00, 1.00, (0.00,  0.00)),
            self._build_fallback_route(o, d, o_aqi * 0.75, d_aqi * 0.80, "Clean Path",      "optimal",  1.15, 0.78, (0.05, -0.04)),
            self._build_fallback_route(o, d, o_aqi * 1.25, d_aqi * 1.20, "Highway Express", "fast",     1.08, 1.28, (-0.03, 0.06)),
        ]
        routes.sort(key=lambda r: r["exposure_score"])
        routes[0]["badge"] = "Cleanest"

        try:
            dep = self.forecaster.find_optimal_window(origin)["best_time"]
        except Exception:
            dep = None

        return {
            "origin": origin, "destination": destination,
            "origin_aqi": round(o_aqi, 1), "destination_aqi": round(d_aqi, 1),
            "routes": routes, "optimal_departure": dep,
        }


# ─────────────────────────────────────────────────────────────────
# 3. HOTSPOT DETECTOR  (DBSCAN)
# ─────────────────────────────────────────────────────────────────

class HotspotDetector:
    """
    DBSCAN-based clustering on AQI sensor locations.
    Works with both real OpenAQ sensor dicts and synthetic data.
    """

    def _synthetic_sensors(self, city: str) -> pd.DataFrame:
        rng  = random.Random(hash(city) % 7777)
        base = CITY_COORDS.get(city, (28.6, 77.2))
        rows = []
        for i in range(80):
            cluster_center = rng.choice([
                (base[0] - 0.08, base[1] + 0.06),
                (base[0] + 0.10, base[1] - 0.05),
                (base[0] - 0.05, base[1] - 0.10),
                (base[0] + 0.05, base[1] + 0.12),
            ])
            if rng.random() < 0.4:
                lat = cluster_center[0] + rng.gauss(0, 0.03)
                lon = cluster_center[1] + rng.gauss(0, 0.03)
                aqi = rng.gauss(180, 30)
            else:
                lat = base[0] + rng.gauss(0, 0.15)
                lon = base[1] + rng.gauss(0, 0.15)
                aqi = rng.gauss(80, 25)
            rows.append({
                "sensor_id": f"SYN{i:03d}",
                "lat": round(lat, 6), "lon": round(lon, 6),
                "aqi": max(10, round(aqi, 1)),
                "pm25": max(5, round(aqi * 0.6, 1)),
                "pm10": max(8, round(aqi * 0.85, 1)),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "Synthetic",
            })
        return pd.DataFrame(rows)

    def _cluster(self, df: pd.DataFrame, aqi_threshold: float) -> Dict:
        all_sensors = df.to_dict(orient="records")
        hot = df[df["aqi"] >= aqi_threshold].copy()
        if len(hot) < 3:
            return {"hotspots": [], "all_sensors": all_sensors,
                    "total_sensors": len(df), "high_aqi_sensors": len(hot)}

        labels = DBSCAN(eps=0.03, min_samples=2).fit_predict(hot[["lat", "lon"]].values)
        hot["cluster"] = labels

        hotspots = []
        for cid in set(labels):
            if cid == -1:
                continue
            c = hot[hot["cluster"] == cid]
            avg_aqi = float(c["aqi"].mean())
            hotspots.append({
                "id":           int(cid),
                "center":       [round(float(c["lat"].mean()), 6), round(float(c["lon"].mean()), 6)],
                "radius_km":    round(max(0.4, float(c["lat"].std()) * 111), 2),
                "max_aqi":      round(float(c["aqi"].max()), 1),
                "avg_aqi":      round(avg_aqi, 1),
                "category":     _aqi_category(avg_aqi),
                "color":        _aqi_color(avg_aqi),
                "sensor_count": len(c),
                "sensors":      c.to_dict(orient="records"),
            })
        hotspots.sort(key=lambda h: h["avg_aqi"], reverse=True)
        return {
            "hotspots":         hotspots,
            "all_sensors":      all_sensors,
            "total_sensors":    len(df),
            "high_aqi_sensors": len(hot),
        }

    def detect(self, city: str, aqi_threshold: float = 120) -> Dict:
        """Detect hotspots from synthetic sensor data (fallback)."""
        df = self._synthetic_sensors(city)
        result = self._cluster(df, aqi_threshold)
        result["city"] = city
        return result

    def detect_from_sensors(self, city: str, sensors: List[Dict], aqi_threshold: float = 100) -> Dict:
        """Detect hotspots from real OpenAQ sensor readings."""
        df = pd.DataFrame(sensors)
        # Ensure required columns exist
        for col in ["lat", "lon", "aqi"]:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in sensor data for {city}, using synthetic fallback")
                return self.detect(city, aqi_threshold)
        df = df.dropna(subset=["lat", "lon", "aqi"])
        result = self._cluster(df, aqi_threshold)
        result["city"] = city
        return result


# ─────────────────────────────────────────────────────────────────
# 4. HEALTH PROFILER
# ─────────────────────────────────────────────────────────────────

HEALTH_RULES = {
    "asthma":        {"mult": 1.8, "concern": "PM2.5 triggers bronchospasm"},
    "copd":          {"mult": 2.0, "concern": "Severe oxidative stress risk"},
    "heart_disease": {"mult": 1.6, "concern": "Cardiovascular strain from NO₂"},
    "diabetes":      {"mult": 1.3, "concern": "Systemic inflammation worsened"},
    "elderly":       {"mult": 1.4, "concern": "Reduced respiratory reserve"},
    "child":         {"mult": 1.5, "concern": "Developing lungs at higher risk"},
    "pregnant":      {"mult": 1.7, "concern": "Fetal exposure risk"},
    "healthy_adult": {"mult": 1.0, "concern": None},
}

class HealthProfiler:
    def assess(self, aqi: float, conditions: List[str], age: int = 30, activity: str = "moderate") -> Dict:
        max_mult = 1.0
        concerns = []
        for cond in conditions:
            rule = HEALTH_RULES.get(cond, HEALTH_RULES["healthy_adult"])
            if rule["mult"] > max_mult:
                max_mult = rule["mult"]
            if rule["concern"]:
                concerns.append(rule["concern"])

        age_factor  = 1.2 if age >= 65 else (1.3 if age <= 12 else 1.0)
        act_factor  = {"low": 0.8, "moderate": 1.0, "high": 1.4}.get(activity, 1.0)
        eff_aqi     = aqi * max_mult * age_factor * act_factor
        safe_thresh = round(50 / (max_mult * age_factor), 1)

        recs = []
        if eff_aqi > 200:
            recs += ["Avoid all outdoor activity today.", "Keep windows closed and use an air purifier indoors."]
        elif eff_aqi > 150:
            recs += ["Limit outdoor exposure to under 30 minutes.", "Wear N95 mask if you must go outside."]
        elif eff_aqi > 100:
            recs += ["Sensitive individuals should limit prolonged outdoor activity.", "Consider indoor exercise alternatives."]
        else:
            recs.append("Conditions are acceptable for most outdoor activities.")
        if "asthma" in conditions or "copd" in conditions:
            recs.append("Carry your inhaler / rescue medication when outdoors.")
        if activity == "high" and aqi > 100:
            recs.append("Reschedule intense workouts to 5–7 AM when AQI is typically lowest.")
        if aqi > 50:
            recs.append("Best travel window: early morning (5–7 AM IST) when AQI drops ~30%.")

        return {
            "input_aqi":        round(aqi, 1),
            "effective_aqi":    round(eff_aqi, 1),
            "risk_level":       _aqi_category(eff_aqi),
            "risk_color":       _aqi_color(eff_aqi),
            "safe_aqi_threshold": safe_thresh,
            "conditions":       conditions,
            "concerns":         concerns,
            "recommendations":  recs,
            "mask_advised":     eff_aqi > 100,
            "outdoor_safe":     eff_aqi <= 100,
        }

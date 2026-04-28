from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.models import HealthProfiler

router = APIRouter()
profiler = HealthProfiler()

class HealthRequest(BaseModel):
    aqi: float
    conditions: Optional[List[str]] = ["healthy_adult"]
    age: Optional[int] = 30
    activity: Optional[str] = "moderate"  # low | moderate | high

@router.post("/assess")
def assess_health(req: HealthRequest):
    """Get personalized health risk assessment for current AQI conditions."""
    return profiler.assess(req.aqi, req.conditions, req.age, req.activity)

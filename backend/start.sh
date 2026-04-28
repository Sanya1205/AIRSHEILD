#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AirShield ML Backend  v2  — FastAPI        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Copy .env if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "⚠  Created .env from .env.example — add your OPENWEATHER_KEY for extra pollutant data"
  echo ""
fi

# Create venv if needed
if [ ! -d "venv" ]; then
  echo "→ Creating Python virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "→ Installing / updating dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "✓  Data sources wired:"
echo "   • OpenAQ v3       — live PM2.5/PM10  (no key required)"
echo "   • OpenWeather     — NO2/O3/CO/SO2    (add OPENWEATHER_KEY to .env)"
echo "   • OSRM            — real road routes  (no key required)"
echo "   • Prophet / LR    — AQI forecasting   (trains on real data)"
echo ""
echo "→ Starting server at http://localhost:8000"
echo "   Swagger UI: http://localhost:8000/docs"
echo ""

uvicorn main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --reload \
  --log-level info

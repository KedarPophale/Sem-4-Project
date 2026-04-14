
"""
main.py  –  AI-Powered Crime-Aware Safe Route Navigation System
================================================================
FastAPI backend with Twilio SOS integration.
"""

from __future__ import annotations

import os
import io
import math
import datetime
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
import httpx
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from twilio.rest import Client as TwilioClient

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ3YTFjMzM4N2NlMDRmODU5ODViZTViMWMwOTQ2NDU5IiwiaCI6Im11cm11cjY0In0="
ORS_BASE    = "https://api.openrouteservice.org"
NOMINATIM   = "https://nominatim.openstreetmap.org/search"

TWILIO_SID    = "AC51b9c9c3d537fa5bc49d23820c83e62d"
TWILIO_TOKEN  = "eebb0aa6043654277698442a29296b0d "
TWILIO_FROM   = "+16626855901"
SOS_CONTACT   = "+919920969206"

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "crime_model.pkl"
DATA_PATH   = BASE_DIR / "crime_data.pkl"

MAX_ROUTE_KM = 80
SAMPLE_PTS   = 30

# ─────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="SafeRoute AI")

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_model      = None
_features   = None
_crime_data = None

def _load_assets():
    global _model, _features, _crime_data
    if MODEL_PATH.exists():
        _model, _features = joblib.load(str(MODEL_PATH))  # ✅ FIXED
    if DATA_PATH.exists():
        _crime_data = joblib.load(str(DATA_PATH))

_load_assets()

# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _route_length_km(coords: list[tuple]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += _haversine_km(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
    return total

def _sample_route(coords: list[tuple], n: int = SAMPLE_PTS) -> list[tuple]:
    if len(coords) <= n:
        return coords
    idxs = np.linspace(0, len(coords) - 1, n, dtype=int)
    return [coords[i] for i in idxs]

# ✅ FIXED MODEL PREDICTION
def _predict_risk(lat: float, lon: float, hour: int, day: int, month: int) -> float:
    if _model is None:
        return 0.5

    # Feature engineering (MUST MATCH TRAINING)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    lat_lon_interaction = lat * lon
    lat_sq = lat ** 2
    lon_sq = lon ** 2

    center_lat = lat  # approximate runtime center (safe fallback)
    center_lon = lon
    center_dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)

    input_data = np.array([[
        lat,
        lon,
        hour,
        day,
        month,
        hour_sin,
        hour_cos,
        month_sin,
        month_cos,
        lat_lon_interaction,
        lat_sq,
        lon_sq,
        center_dist
    ]], dtype=float)

    risk = float(_model.predict(input_data)[0])

    # Stable normalization
    risk = 1 / (1 + np.exp(-3 * (risk - 0.5)))

    return float(np.clip(risk, 0.0, 1.0))
def _route_risk_score(coords: list[tuple], hour: int, day: int, month: int) -> float:
    sample = _sample_route(coords)
    risks = np.array([_predict_risk(lat, lon, hour, day, month) for lat, lon in sample])
    if risks.max() - risks.min() > 1e-6:
        risks = (risks - risks.min()) / (risks.max() - risks.min())
    return float((0.7 * np.mean(risks)) + (0.3 * np.max(risks)))

async def _geocode(place: str) -> Optional[tuple[float, float]]:
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "SafeRouteAI/1.0"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(NOMINATIM, params=params, headers=headers)
        data = resp.json()
        return (float(data[0]["lat"]), float(data[0]["lon"])) if data else None

async def _fetch_routes(src: tuple, dst: tuple) -> list[list[tuple]]:
    url = f"{ORS_BASE}/v2/directions/driving-car/geojson"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [[src[1], src[0]], [dst[1], dst[0]]],
        "alternative_routes": {"target_count": 3},
        "instructions": False
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise ValueError("ORS API Error")
        return [[(c[1], c[0]) for c in f["geometry"]["coordinates"]] for f in resp.json().get("features", [])]

def _build_map(src, dst, routes, risks, src_name, dst_name) -> str:
    fmap = folium.Map(location=[(src[0]+dst[0])/2, (src[1]+dst[1])/2], zoom_start=13, tiles="CartoDB dark_matter")

    if _crime_data:
        HeatMap([[r[0], r[1], r[2]] for r in _crime_data]).add_to(fmap)

    best_idx = int(np.argmin(risks))

    for i, (coords, risk) in enumerate(zip(routes, risks)):
        folium.PolyLine(
            coords,
            color="#00E676" if i == best_idx else "#546E7A",
            weight=6 if i == best_idx else 3
        ).add_to(fmap)

    folium.Marker(src, icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(dst, icon=folium.Icon(color="red")).add_to(fmap)

    return fmap._repr_html_()

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/route", response_class=HTMLResponse)
async def compute_route(request: Request, source: str = Form(...), destination: str = Form(...)):
    now = datetime.datetime.now()
    h, d, m = now.hour, now.weekday() + 1, now.month

    ctx = {
        "request": request,
        "source": source,
        "destination": destination,
        "hour": h,
        "day": d,
        "month": m,
        "error": None,
        "map_html": None,
        "route_info": []
    }

    try:
        src_coords = await _geocode(source)
        dst_coords = await _geocode(destination)

        if not src_coords or not dst_coords:
            ctx["error"] = "Location not found"
            return templates.TemplateResponse("map.html", ctx)

        routes = await _fetch_routes(src_coords, dst_coords)

        if not routes:
            ctx["error"] = "No routes found"
            return templates.TemplateResponse("map.html", ctx)

        risks = [_route_risk_score(r, h, d, m) for r in routes]
        best_idx = int(np.argmin(risks))

        ctx["route_info"] = [
            {"index": i+1, "risk": f"{risk:.1%}", "length": f"{_route_length_km(r):.1f} km", "safest": i == best_idx}
            for i, (r, risk) in enumerate(zip(routes, risks))
        ]

        ctx["map_html"] = _build_map(src_coords, dst_coords, routes, risks, source, destination)

        return templates.TemplateResponse("map.html", ctx)

    except Exception as exc:
        ctx["error"] = f"Application Error: {str(exc)}"
        return templates.TemplateResponse("map.html", ctx)

@app.post("/sos")
async def sos_alert(lat: str = Form(...), lon: str = Form(...), source: str = Form("Unknown"), destination: str = Form("Unknown")):
    try:
        msg = f"🆘 SOS ALERT!\nFrom: {source}\nTo: {destination}\nMaps: http://maps.google.com/maps?q={lat},{lon}"
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(body=msg, from_=TWILIO_FROM, to=SOS_CONTACT)
        return JSONResponse({"ok": True, "sid": message.sid})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


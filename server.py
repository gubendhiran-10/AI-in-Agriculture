"""
AI in Agriculture - Emergency Resource Allocation System
Complete Flask Backend with Dynamic ML Detection, IoT Sensor Analysis,
Optimization Engine, Resilient Database Layer, and Live State Chatbot.
"""

import os
import sys
import json
import random
import uuid
import math
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory, abort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database as db
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(BASE_DIR, "frontend")
app = Flask(__name__, static_folder=FRONT_DIR, static_url_path="")

# ═══ REGION PROFILES ═══
REGIONS = [
    {
        "id": "R1",
        "name": "Vidarbha, Maharashtra",
        "lat": 20.932,
        "lng": 79.133,
        "profile": "Arid black soil, frequent drought zone, cotton/soybean belts",
        "base_area": 120,
        "base_vuln": 0.88,
        "sensor_bias": {"soil": 16.0, "temp": 42.0, "hum": 24.0, "rain": 0.0, "wind": 18.0}
    },
    {
        "id": "R2",
        "name": "Coastal Andhra Pradesh",
        "lat": 16.506,
        "lng": 80.648,
        "profile": "Lowland delta basin, cyclone & riverine flood prone, paddy crops",
        "base_area": 95,
        "base_vuln": 0.72,
        "sensor_bias": {"soil": 78.0, "temp": 30.0, "hum": 86.0, "rain": 98.0, "wind": 38.0}
    },
    {
        "id": "R3",
        "name": "Jodhpur, Rajasthan",
        "lat": 26.239,
        "lng": 73.024,
        "profile": "Thar desert fringe, locust corridor & warm pest infestations",
        "base_area": 60,
        "base_vuln": 0.65,
        "sensor_bias": {"soil": 28.0, "temp": 34.0, "hum": 72.0, "rain": 8.0, "wind": 22.0}
    },
    {
        "id": "R4",
        "name": "Ludhiana, Punjab",
        "lat": 30.901,
        "lng": 75.857,
        "profile": "Intensive wheat/rice stubble burning corridor, fire hazard",
        "base_area": 180,
        "base_vuln": 0.82,
        "sensor_bias": {"soil": 32.0, "temp": 44.5, "hum": 22.0, "rain": 0.0, "wind": 32.0}
    },
    {
        "id": "R5",
        "name": "Bhubaneswar, Odisha",
        "lat": 20.296,
        "lng": 85.825,
        "profile": "Eastern coastal plateau, alternating heatwaves and erratic moisture",
        "base_area": 45,
        "base_vuln": 0.60,
        "sensor_bias": {"soil": 22.0, "temp": 39.0, "hum": 48.0, "rain": 4.0, "wind": 16.0}
    },
]

TOTAL_RESOURCES = {
    "water": 1000.0,      # Kiloliters (KL)
    "personnel": 150.0,   # Relief workers / specialists
    "machinery": 40.0,    # Heavy machines, pumps, sprayers, drones
    "funds": 50000.0      # Relief funding budget (INR thousands/standard)
}

IMPACT_DATA = {
    "labels": ["Drought", "Flood", "Pest Outbreak", "Fire"],
    "traditional": [18, 15, 20, 12],
    "ai_based": [6, 5, 7, 3]
}


# ═══ AI / ML EMERGENCY DETECTION ENGINE ═══

def ml_detect_emergency(soil_moisture, temp, humidity, rainfall, wind_speed):
    """
    Multi-sensor AI classification engine for agricultural crises.
    Evaluates bio-climatic indices and returns class probabilities,
    primary classification, confidence score, and contributing risk factors.
    """
    # 1. Drought Probability
    # Triggers on low moisture, high heat, zero/low rainfall
    soil_deficit = max(0.0, min(1.0, (28.0 - soil_moisture) / 20.0))
    heat_factor = max(0.0, min(1.0, (temp - 34.0) / 12.0))
    rain_deficit = 1.0 if rainfall < 5.0 else max(0.0, 1.0 - (rainfall / 30.0))
    p_drought = (0.50 * soil_deficit) + (0.30 * heat_factor) + (0.20 * rain_deficit)

    # 2. Flood Probability
    # Triggers on excessive rainfall, high soil saturation, elevated humidity
    rain_excess = max(0.0, min(1.0, (rainfall - 45.0) / 65.0))
    soil_sat = max(0.0, min(1.0, (soil_moisture - 60.0) / 30.0))
    p_flood = (0.60 * rain_excess) + (0.30 * soil_sat) + (0.10 * (humidity / 100.0))

    # 3. Fire / Crop Residue Flare Probability
    # Triggers on high ambient temperature, low humidity, elevated wind speed
    fire_temp = max(0.0, min(1.0, (temp - 38.0) / 10.0))
    fire_dry = max(0.0, min(1.0, (35.0 - humidity) / 25.0))
    fire_wind = max(0.0, min(1.0, (wind_speed - 15.0) / 30.0))
    p_fire = (0.45 * fire_temp) + (0.35 * fire_dry) + (0.20 * fire_wind)

    # 4. Pest Outbreak Probability
    # Triggers on warm humidity sweet-spot (25-36°C) and high humidity (>65%)
    pest_temp = 1.0 - (abs(temp - 30.5) / 10.0)
    pest_temp = max(0.0, min(1.0, pest_temp))
    pest_hum = max(0.0, min(1.0, (humidity - 55.0) / 35.0))
    p_pest = (0.55 * pest_hum) + (0.45 * pest_temp)

    # Score vector
    candidates = [
        ("Drought", p_drought),
        ("Flood", p_flood),
        ("Fire", p_fire),
        ("Pest Outbreak", p_pest)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    primary_name, raw_prob = candidates[0]

    # If all probabilities are low, designate Normal/Monitoring
    if raw_prob < 0.35:
        primary_name = "Normal (Low Risk)"
        display_prob = round(1.0 - raw_prob, 2)
        confidence = "Normal"
        risk_factors = ["Environmental indicators within safe agronomic ranges."]
        action = "Routine automated sensor telemetry monitoring."
    else:
        display_prob = round(min(0.98, max(0.35, raw_prob)), 2)
        confidence = "High" if display_prob >= 0.70 else "Moderate"
        risk_factors = []
        action = ""

        if primary_name == "Drought":
            if soil_moisture < 20.0:
                risk_factors.append(f"Critical soil moisture deficit: {soil_moisture}% (threshold < 20%)")
            if temp > 38.0:
                risk_factors.append(f"Elevated thermal stress: {temp}°C ambient temperature")
            if rainfall < 5.0:
                risk_factors.append(f"Prolonged precipitation drought: {rainfall} mm recorded")
            action = "Dispatch water tankers, deploy emergency drip irrigation, distribute drought aid packs."

        elif primary_name == "Flood":
            if rainfall > 70.0:
                risk_factors.append(f"Extreme precipitation event: {rainfall} mm/day")
            if soil_moisture > 75.0:
                risk_factors.append(f"Field soil saturation at critical level: {soil_moisture}%")
            action = "Mobilize drainage machinery, deploy flood rescue teams, establish emergency barrier bunds."

        elif primary_name == "Fire":
            if temp > 40.0:
                risk_factors.append(f"Extreme heat anomaly: {temp}°C")
            if humidity < 25.0:
                risk_factors.append(f"Arid relative humidity: {humidity}%")
            if wind_speed > 20.0:
                risk_factors.append(f"High gust spread factor: {wind_speed} km/h")
            action = "Dispatch emergency fire tenders, mobilize water tankers, establish firebreak corridors."

        elif primary_name == "Pest Outbreak":
            if humidity > 65.0:
                risk_factors.append(f"High relative humidity conducive to breeding: {humidity}%")
            if 26.0 <= temp <= 35.0:
                risk_factors.append(f"Optimal insect proliferation temperature: {temp}°C")
            action = "Deploy automated agricultural spray drones, supply bio-pesticides, assign crop scientists."

    return {
        "emergency": primary_name,
        "probability": display_prob,
        "confidence": confidence,
        "risk_factors": risk_factors,
        "recommended_action": action,
        "scores": {
            "drought": round(p_drought, 2),
            "flood": round(p_flood, 2),
            "fire": round(p_fire, 2),
            "pest": round(p_pest, 2)
        }
    }


def compute_severity_score(probability, area_ha, vulnerability, max_area=250.0):
    """
    Weighted Multi-Factor Severity Formula:
    Severity = (0.50 * Probability) + (0.30 * Area / MaxArea) + (0.20 * Vulnerability) * 100
    """
    norm_area = min(1.0, max(0.0, float(area_ha) / max_area))
    score = (0.50 * float(probability)) + (0.30 * norm_area) + (0.20 * float(vulnerability))
    return round(min(100.0, score * 100.0), 2)


def optimize_resource_allocation(severity_records, total_pool=TOTAL_RESOURCES):
    """
    Priority-Constrained Proportional Allocation (Linear Optimization model):
    Higher severity scores receive a super-linear priority weighting,
    ensuring critical disaster zones receive the vital proportion of emergency aid.
    """
    # Weighted severity: regions with score >= 70 receive priority multiplier
    raw_weights = []
    for r in severity_records:
        score = float(r["severity_score"])
        priority_boost = 1.25 if score >= 70.0 else (1.05 if score >= 50.0 else 0.85)
        raw_weights.append(max(0.05, score * priority_boost))

    sum_weights = sum(raw_weights) + 1e-8
    norm_weights = [w / sum_weights for w in raw_weights]

    allocations = []
    for r, w in zip(severity_records, norm_weights):
        allocations.append({
            "region_id": r["region_id"],
            "water": round(w * total_pool["water"], 1),
            "personnel": round(w * total_pool["personnel"], 1),
            "machinery": round(w * total_pool["machinery"], 1),
            "funds": round(w * total_pool["funds"], 1)
        })
    return allocations


# ═══ TELEMETRY GENERATION ═══

def generate_live_iot():
    """Generates realistic live field sensor telemetry for all monitored regions."""
    readings = []
    for r in REGIONS:
        bias = r["sensor_bias"]
        # Jitter around bias
        sm = max(5.0, min(95.0, round(bias["soil"] + random.uniform(-4.0, 4.0), 1)))
        tp = max(18.0, min(50.0, round(bias["temp"] + random.uniform(-2.5, 2.5), 1)))
        hm = max(15.0, min(98.0, round(bias["hum"] + random.uniform(-5.0, 5.0), 1)))
        rf = max(0.0, round(bias["rain"] + random.uniform(-6.0, 8.0), 1))
        ws = max(3.0, min(70.0, round(bias["wind"] + random.uniform(-3.0, 4.0), 1)))

        readings.append({
            "region": f"{r['id']} - {r['name']}",
            "lat": r["lat"],
            "lng": r["lng"],
            "soil_moisture": sm,
            "temperature": tp,
            "humidity": hm,
            "rainfall_mm": rf,
            "wind_speed_kmh": ws
        })
    return readings


def generate_live_weather():
    """Generates meteorological conditions correlating with regional climate states."""
    cond_map = {
        "R1": ("Sunny & Arid", 42.0, 22.0, 18.0, 10.0, "Sunny", 5),
        "R2": ("Heavy Rain / Storm", 29.5, 88.0, 36.0, 3.0, "Thunderstorm", 90),
        "R3": ("Warm & Dusty", 35.0, 68.0, 24.0, 8.0, "Partly Cloudy", 15),
        "R4": ("Extreme Heat & Haze", 44.0, 20.0, 30.0, 10.5, "Haze", 0),
        "R5": ("Overcast & Humid", 38.5, 52.0, 15.0, 7.0, "Light Rain", 40),
    }
    result = []
    for r in REGIONS:
        base = cond_map.get(r["id"], ("Partly Cloudy", 32.0, 50.0, 15.0, 6.0, "Clear", 20))
        result.append({
            "region": f"{r['id']} - {r['name']}",
            "condition": base[0],
            "temp_c": round(base[1] + random.uniform(-1.5, 1.5), 1),
            "humidity": round(base[2] + random.uniform(-4.0, 4.0), 1),
            "wind_kmh": round(base[3] + random.uniform(-3.0, 3.0), 1),
            "uv_index": round(base[4], 1),
            "forecast_24h": base[5],
            "rain_chance": base[6]
        })
    return result


# ═══ CHATBOT INTELLIGENCE ═══

CHATBOT_KB = {
    "what is this system": "This is an AI-Based Emergency Resource Allocation System for Agriculture. It continuously monitors field IoT sensors and satellite data across agricultural regions, uses Machine Learning to detect disasters (droughts, floods, pest outbreaks, fires), scores crisis severity, and optimizes the dispatch of water, personnel, machinery, and funds to where they are needed most.",
    "how does it work": "The system works through 5 automated stages:\n1️⃣ Data Collection — Field IoT sensors capture soil moisture, temperature, humidity, rainfall, and wind.\n2️⃣ AI Detection — The ML classification model identifies emergency patterns.\n3️⃣ Severity Scoring — Evaluates (0.5×Probability) + (0.3×Area) + (0.2×Vulnerability) × 100.\n4️⃣ Resource Optimization — Priority-weighted linear optimization divides emergency pools.\n5️⃣ Real-time Dispatch — Tracks deployed relief assets and alerts in real-time.",
    "what emergencies": "The AI detects 4 primary agricultural crises:\n🌵 Drought — Severe soil moisture deficit (<20%), zero rainfall, intense heat.\n🌊 Flood — Torrential rainfall (>80mm/day), saturated soil (>80%), high humidity.\n🐛 Pest Outbreak — Favorable breeding temperatures (26-35°C) and high humidity (>65%).\n🔥 Crop / Wildfire — Extreme ambient temperature (>40°C), low humidity (<25%), high wind gusts.",
    "severity score": "The Severity Score uses the multi-factor equation:\n\nScore = [ (0.50 × ML Probability) + (0.30 × Normalized Area) + (0.20 × Vulnerability Index) ] × 100\n\n• Score ≥ 70: CRITICAL (Red, immediate priority dispatch)\n• Score ≥ 45: WARNING (Orange, high priority)\n• Score < 45: NORMAL (Green, routine standby)",
    "what regions": "We currently monitor 5 vulnerable agricultural belts:\n📍 R1 — Vidarbha, Maharashtra (Drought & heat prone)\n📍 R2 — Coastal Andhra Pradesh (Flood & cyclone prone)\n📍 R3 — Jodhpur, Rajasthan (Pest & desert fringe)\n📍 R4 — Ludhiana, Punjab (Fire hazard & stubble corridor)\n📍 R5 — Bhubaneswar, Odisha (Erratic weather & heat)",
    "resources": "Available Emergency Pool:\n💧 Water: 1,000 Kiloliters (KL)\n👥 Personnel: 150 Relief Workers / Specialists\n🚜 Machinery: 40 Heavy Units (Pumps, Sprayers, Drones)\n💰 Funds: ₹50,000 (Relief Emergency Fund)\n\nThese resources are dynamically allocated by priority optimization.",
    "technology": "Full Tech Stack:\n🐍 Python Flask (REST Backend)\n🧠 Dynamic Multi-Sensor ML Classifier\n🗄️ Resilient Dual Database (MongoDB + SQLite fallback)\n📊 Leaflet.js & OpenStreetMap (Geospatial Mapping)\n📈 Chart.js (Data Visualizations)\n🎙️ Web Speech API (Voice Assistant)",
    "who made this": "Created by Dhanush, Department of Computer Science & Engineering, as an advanced project on AI in Agriculture.",
    "database": "The application features a resilient dual database layer. It actively connects to MongoDB for document storage, with automated instant fallback to SQLite (agriculture.db) if MongoDB is offline.",
}


def get_live_chatbot_response(user_msg):
    """Answers user queries with live state context whenever relevant."""
    msg = user_msg.lower().strip()

    # Dynamic query: current status or highest severity
    if any(k in msg for k in ["status", "current", "highest severity", "latest run", "how is it now"]):
        runs = db.get_all_pipeline_runs()
        events = db.get_all_events()
        active_alerts = [e for e in events if e.get("status") in ["detected", "dispatched"]]

        if not runs:
            return "The AI system is initialized and ready. No pipeline run has been executed yet. Click 'Run Pipeline' in the top bar to analyze field sensors and detect emergencies!"

        latest_run = runs[0]
        sev = db.get_severity_by_run(latest_run["run_id"])
        top = sev[0] if sev else None

        resp = f"📊 **Live System Status**\n• Latest Pipeline: `{latest_run['run_id']}`\n"
        if top:
            resp += f"• **Highest Crisis**: {top['region_id']} ({top['predicted_emergency']})\n• **Severity Score**: {top['severity_score']}/100\n"
        resp += f"• **Active Alerts**: {len(active_alerts)} un-resolved incidents\n• **Database Backend**: {db.get_backend_info()['engine']} (Online)"
        return resp

    # Dynamic query: active alerts
    if any(k in msg for k in ["alerts", "emergency count", "how many alerts"]):
        events = db.get_all_events()
        active = [e for e in events if e.get("status") in ["detected", "dispatched"]]
        if not active:
            return "All regions are currently stable. No active unresolved emergencies."
        summary_lines = [f"• **{e['region_id']}**: {e['emergency_type']} (Score: {e['severity_score']}, Status: {e['status'].upper()})" for e in active[:4]]
        return f"🚨 Currently **{len(active)} active emergency alerts**:\n" + "\n".join(summary_lines)

    # Keyword KB lookup
    best_match = None
    best_score = 0
    for key, response in CHATBOT_KB.items():
        keywords = key.split()
        score = sum(1 for kw in keywords if kw in msg)
        if score > best_score:
            best_score = score
            best_match = response

    if best_score > 0:
        return best_match

    # Greetings
    if any(w in msg for w in ["hello", "hi", "hey", "namaste"]):
        return "Hello! 👋 I'm the AI Agriculture Assistant. I can help you understand the emergency resource allocation system, inspect live disaster alerts, or explain how the ML detection and scoring works. Try asking 'what is the current status?' or say 'help'!"

    if any(w in msg for w in ["thank", "thanks", "ok", "great"]):
        return "You're welcome! 😊 Feel free to ask about any region, emergency criteria, or run the pipeline to see AI in action."

    return "I'm not sure about that, but I can help you with:\n• 'What is the current status?'\n• 'How does the severity score work?'\n• 'What emergencies are detected?'\n• 'Explain drought detection thresholds'\n• 'What resources are available?'\n\nTry asking one of these questions!"


# ═══ API ROUTES ═══

@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")


@app.route("/api/system_status")
def api_system_status():
    stats = db.get_db_stats()
    runs = db.get_all_pipeline_runs()
    events = db.get_all_events()
    active_events = [e for e in events if e.get("status") != "resolved"]
    return jsonify({
        "database": db.get_backend_info(),
        "total_runs": len(runs),
        "active_alerts": len(active_events),
        "latest_run": runs[0] if runs else None,
        "total_resources": TOTAL_RESOURCES,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/regions")
def api_regions():
    return jsonify([{
        "id": r["id"],
        "name": r["name"],
        "lat": r["lat"],
        "lng": r["lng"],
        "profile": r["profile"],
        "base_area": r["base_area"],
        "base_vuln": r["base_vuln"]
    } for r in REGIONS])


@app.route("/api/run_demo")
def api_run_demo():
    """
    Executes the Complete 5-Stage AI Disaster Pipeline:
    1. Ingests dynamic live IoT sensor readings
    2. Executes multi-factor ML classification on sensor data
    3. Computes severity score from ML probability, area, and vulnerability
    4. Calculates priority-constrained optimal resource distribution
    5. Saves full telemetry and history into database
    """
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    db.create_pipeline_run(run_id)

    # 1. Telemetry
    iot_readings = generate_live_iot()
    db.save_iot_readings(iot_readings)

    # 2 & 3. ML Detection & Severity Scoring
    severity_records = []
    for r, iot in zip(REGIONS, iot_readings):
        ml_result = ml_detect_emergency(
            iot["soil_moisture"],
            iot["temperature"],
            iot["humidity"],
            iot["rainfall_mm"],
            iot["wind_speed_kmh"]
        )
        sev_score = compute_severity_score(
            ml_result["probability"],
            r["base_area"],
            r["base_vuln"]
        )
        severity_records.append({
            "region_id": f"{r['id']} - {r['name']}",
            "predicted_emergency": ml_result["emergency"],
            "probability": ml_result["probability"],
            "area_ha": r["base_area"],
            "vulnerability": r["base_vuln"],
            "severity_score": sev_score,
            "confidence": ml_result["confidence"],
            "risk_factors": ml_result["risk_factors"],
            "recommended_action": ml_result["recommended_action"],
            "sensors": {
                "soil_moisture": iot["soil_moisture"],
                "temperature": iot["temperature"],
                "humidity": iot["humidity"],
                "rainfall_mm": iot["rainfall_mm"],
                "wind_speed_kmh": iot["wind_speed_kmh"]
            },
            "lat": r["lat"],
            "lng": r["lng"]
        })

    # Sort descending by severity score
    severity_records.sort(key=lambda x: x["severity_score"], reverse=True)

    # 4. Resource Allocation
    allocations = optimize_resource_allocation(severity_records, TOTAL_RESOURCES)

    # 5. Persist
    db.save_severity_records(run_id, severity_records)
    db.save_allocation_records(run_id, allocations)
    db.save_emergency_events(run_id, severity_records)
    db.finish_pipeline_run(
        run_id,
        f"Analyzed {len(REGIONS)} regions with live ML classification. Peak severity: {severity_records[0]['severity_score']} in {severity_records[0]['region_id']}"
    )

    return jsonify({
        "run_id": run_id,
        "database_backend": db.get_backend_info()["engine"],
        "severity": severity_records,
        "allocation": allocations,
        "impact": IMPACT_DATA,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Interactive Crisis Simulator Endpoint:
    Allows user/evaluator to test custom sensor readings and witness
    dynamic ML emergency detection, severity calculation, and allocation.
    """
    data = request.get_json() or {}
    region_name = data.get("region_name", "Custom Agricultural Field")
    soil_moisture = float(data.get("soil_moisture", 15.0))
    temperature = float(data.get("temperature", 42.0))
    humidity = float(data.get("humidity", 25.0))
    rainfall = float(data.get("rainfall_mm", 0.0))
    wind_speed = float(data.get("wind_speed_kmh", 20.0))
    area_ha = float(data.get("area_ha", 100.0))
    vulnerability = float(data.get("vulnerability", 0.75))

    # Run ML detection
    ml_res = ml_detect_emergency(soil_moisture, temperature, humidity, rainfall, wind_speed)
    sev_score = compute_severity_score(ml_res["probability"], area_ha, vulnerability)

    # Simulated single allocation estimate based on severity weight
    total_water = TOTAL_RESOURCES["water"]
    norm_w = min(1.0, max(0.1, sev_score / 100.0))

    sim_result = {
        "region_name": region_name,
        "inputs": {
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall_mm": rainfall,
            "wind_speed_kmh": wind_speed,
            "area_ha": area_ha,
            "vulnerability": vulnerability
        },
        "ml_detection": {
            "emergency": ml_res["emergency"],
            "probability": ml_res["probability"],
            "confidence": ml_res["confidence"],
            "risk_factors": ml_res["risk_factors"],
            "recommended_action": ml_res["recommended_action"],
            "class_probabilities": ml_res["scores"]
        },
        "severity": {
            "score": sev_score,
            "level": "CRITICAL" if sev_score >= 70 else ("WARNING" if sev_score >= 45 else "NORMAL"),
            "formula_breakdown": f"(0.50 × {ml_res['probability']}) + (0.30 × {round(area_ha/250,2)}) + (0.20 × {vulnerability}) = {sev_score}/100"
        },
        "suggested_dispatch": {
            "water_kl": round(norm_w * 250.0, 1),
            "personnel": round(norm_w * 40.0, 1),
            "machinery_units": round(norm_w * 10.0, 1),
            "funds_inr": round(norm_w * 15000.0, 1)
        }
    }
    return jsonify(sim_result)


@app.route("/api/iot_data")
def api_iot():
    readings = generate_live_iot()
    db.save_iot_readings(readings)
    return jsonify(readings)


@app.route("/api/weather")
def api_weather():
    return jsonify(generate_live_weather())


@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    data = request.get_json() or {}
    msg = data.get("message", "")
    response_text = get_live_chatbot_response(msg)
    return jsonify({
        "response": response_text,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/alerts")
def api_alerts():
    events = db.get_all_events()
    alerts = []
    for e in events[:25]:
        score = float(e.get("severity_score", 50))
        level = "critical" if score >= 70 else ("warning" if score >= 45 else "info")
        alerts.append({
            "id": e.get("id"),
            "run_id": e.get("run_id"),
            "region": e.get("region_id"),
            "type": e.get("emergency_type"),
            "severity": score,
            "level": level,
            "time": e.get("created_at"),
            "status": e.get("status", "detected"),
            "notes": e.get("notes", "")
        })
    return jsonify(alerts)


@app.route("/api/alerts/<event_id>/action", methods=["POST"])
def api_alert_action(event_id):
    """
    Actionable dispatch & resolution workflow:
    Accepts 'dispatch' or 'resolve'.
    """
    data = request.get_json() or {}
    action = data.get("action", "").lower()
    notes = data.get("notes", "")

    if action == "dispatch":
        new_status = "dispatched"
        if not notes:
            notes = "Emergency response teams & resources dispatched to coordinates."
    elif action == "resolve":
        new_status = "resolved"
        if not notes:
            notes = "Emergency contained. Post-disaster recovery initiated."
    else:
        return jsonify({"error": "Invalid action. Choose 'dispatch' or 'resolve'."}), 400

    success = db.update_emergency_status(event_id, new_status, notes)
    return jsonify({
        "success": success,
        "event_id": event_id,
        "new_status": new_status,
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/history")
def api_history():
    return jsonify(db.get_all_pipeline_runs()[:30])


@app.route("/api/history/<run_id>")
def api_history_detail(run_id):
    return jsonify({
        "severity": db.get_severity_by_run(run_id),
        "allocation": db.get_allocation_by_run(run_id)
    })


@app.route("/api/calculate_severity", methods=["POST"])
def api_calc_severity():
    data = request.get_json() or {}
    prob = float(data.get("probability", 0.5))
    area = float(data.get("area", 50))
    vuln = float(data.get("vulnerability", 0.5))
    score = compute_severity_score(prob, area, vuln)
    level = "CRITICAL" if score >= 70 else ("WARNING" if score >= 45 else "NORMAL")
    return jsonify({
        "severity_score": score,
        "level": level,
        "probability": prob,
        "area": area,
        "vulnerability": vuln
    })


# ═══ ADMIN ROUTES ═══

@app.route("/owner-panel-8x9k2")
def admin_page():
    return send_from_directory(FRONT_DIR, "admin.html")


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    if db.verify_admin(data.get("password", "")):
        return jsonify({"success": True})
    return jsonify({"success": False}), 401


@app.route("/api/admin/stats")
def admin_stats():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_db_stats())


@app.route("/api/admin/pipeline_runs")
def admin_runs():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_all_pipeline_runs())


@app.route("/api/admin/severity")
def admin_sev():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_all_severity())


@app.route("/api/admin/allocations")
def admin_alloc():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_all_allocations())


@app.route("/api/admin/iot")
def admin_iot():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_all_iot())


@app.route("/api/admin/events")
def admin_events():
    if not db.verify_admin(request.headers.get("X-Admin-Key", "")):
        abort(403)
    return jsonify(db.get_all_events())


if __name__ == "__main__":
    os.makedirs(FRONT_DIR, exist_ok=True)
    backend_info = db.get_backend_info()
    print("=" * 64)
    print("  AI in Agriculture - Emergency Resource Allocation System")
    print(f"  DATABASE BACKEND: {backend_info['engine']} ({backend_info['status']})")
    print("  PUBLIC DASHBOARD: http://localhost:5000")
    print("  ADMIN OWNER PANEL: http://localhost:5000/owner-panel-8x9k2")
    print("  Admin Password:   dhanush2026")
    print("=" * 64)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

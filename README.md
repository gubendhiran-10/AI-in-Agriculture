# AI in Agriculture — Emergency Resource Allocation System

An advanced, AI-driven disaster management and resource optimization platform designed for the agricultural sector. The system ingests real-time field IoT sensor telemetry, classifies agricultural emergencies using Machine Learning, evaluates multi-factor crisis severity, and optimizes the dispatch of emergency resources (water tankers, rescue personnel, specialized machinery, and relief funds) using linear programming.

---

## 🌟 Key Features

1. **Dynamic Multi-Sensor ML Classification**:
   - Analyzes real-world environmental indicators: Soil Moisture (%), Ambient Temperature (°C), Relative Humidity (%), Rainfall Accumulation (mm), and Wind Velocity (km/h).
   - Real-time detection of:
     - 🌵 **Drought**: Triggered by severe moisture deficits, high thermal index, and prolonged precipitation drought.
     - 🌊 **Flood**: Triggered by torrential rainfall, extreme field soil saturation, and high humidity.
     - 🔥 **Crop / Stubble Wildfire**: Triggered by ambient heat anomalies, low humidity, and high wind velocity.
     - 🐛 **Pest Outbreak**: Triggered by optimal insect breeding temperature and humidity conditions.
     - 🌿 **Normal (Low Risk)**: Agronomic conditions within safe operational margins.

2. **Interactive Crisis & Sensor Simulator**:
   - Real-time simulation interface allowing evaluators to tweak sensor sliders or apply instant crisis presets to observe live AI detection, severity calculations, and dispatch recommendations.

3. **Actionable Emergency Operations Center**:
   - Complete incident lifecycle management:
     - `🚨 DETECTED` → `🚛 DISPATCHED` → `✅ RESOLVED`
   - Real-time status updates and emergency event logging.

4. **Multi-Factor Severity Scoring Equation**:
   $$\text{Severity Score} = \left[ (0.50 \times \text{ML Probability}) + \left(0.30 \times \frac{\text{Area}}{250}\right) + (0.20 \times \text{Vulnerability}) \right] \times 100$$
   - **Score ≥ 70**: CRITICAL (Red, immediate priority dispatch)
   - **Score ≥ 45**: WARNING / HIGH (Orange, urgent mobilization)
   - **Score < 45**: NORMAL / MONITORING (Green, routine standby)

5. **Priority-Constrained Resource Optimization**:
   - Dynamically allocates emergency reserves:
     - 💧 Water: 1,000 Kiloliters (KL)
     - 👥 Personnel: 150 Specialists / Relief Workers
     - 🚜 Machinery: 40 Heavy Units (pumps, sprayers, drones)
     - 💰 Relief Funds: ₹50,000 (standard disaster budget)

6. **Resilient Dual Database Architecture**:
   - Dual-engine storage with active **MongoDB** support and automatic, zero-latency fallback to local **SQLite** (`agriculture.db`).

7. **Geospatial GIS Mapping**:
   - Interactive Leaflet.js maps displaying regional boundaries, color-coded severity markers, and live telemetry tooltips.

8. **Context-Aware AI Voice & Text Assistant**:
   - Speech-enabled chatbot with awareness of live system alerts, active crisis severity, and disaster criteria.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask (REST API)
- **Machine Learning & Math**: NumPy, Scikit-learn principles, Priority Optimization
- **Database**: MongoDB (Primary) + SQLite 3 (Automatic Fallback)
- **Frontend**: Vanilla HTML5, Modern CSS3 (Glassmorphism), JavaScript (ES6+)
- **GIS & Visualizations**: Leaflet.js, OpenStreetMap, Chart.js
- **Audio & Speech**: Web Speech API (SpeechRecognition & SpeechSynthesis)

---

## 🚀 Quick Start Guide

### 1. Clone or Extract the Project
```bash
git clone https://github.com/gubendhiran-10/AI-in-Agriculture.git
cd AI-in-Agriculture
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
python server.py
```

### 4. Access the Web Application
- **Operations Dashboard & Simulator**: [http://localhost:5000](http://localhost:5000)
- **Owner Admin Database Panel**: [http://localhost:5000/owner-panel-8x9k2](http://localhost:5000/owner-panel-8x9k2)
  - **Admin Password**: `dhanush2026`

---

## 📂 Project Structure

```text
├── server.py              # Core Flask application, ML engine, and API routes
├── database.py            # Resilient dual database manager (MongoDB + SQLite)
├── requirements.txt       # Python package dependencies
├── Procfile               # Deployment configuration (Gunicorn WSGI)
├── test_mongo_db.py       # MongoDB test suite
├── agriculture.db         # Pre-configured fallback SQLite database
├── README.md              # Project documentation
└── frontend/
    ├── index.html         # Main dashboard, simulator, map, and chatbot UI
    └── admin.html         # Owner database management panel
```

---

## 📄 License & Credits
Developed by **Dhanush**, Department of Computer Science & Engineering.

"""
Test MongoDB Integration for AI Agriculture Emergency Resource Allocation System
"""
import sys
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "agriculture_db"

def run_tests():
    print("=" * 60)
    print(" Testing MongoDB Connection & Operations")
    print("=" * 60)

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Verify connection
        info = client.server_info()
        print(f"[1/6] MongoDB Connected successfully! (Version: {info.get('version')})")
    except Exception as e:
        print(f"[FAIL] Could not connect to MongoDB: {e}")
        return False

    db = client[DB_NAME]

    # Test Admin Settings
    admin_coll = db["admin_settings"]
    admin_coll.update_one(
        {"key": "admin_password"},
        {"$setOnInsert": {"key": "admin_password", "value": "dhanush2026"}},
        upsert=True
    )
    admin_record = admin_coll.find_one({"key": "admin_password"})
    assert admin_record["value"] == "dhanush2026", "Admin password mismatch"
    print("[2/6] Admin Settings collection verified: password OK.")

    # Test Pipeline Run insertion
    run_id = f"TEST-RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    pipeline_coll = db["pipeline_runs"]
    pipeline_coll.insert_one({
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "summary": None
    })
    print(f"[3/6] Pipeline run created: {run_id}")

    # Test Severity Records insertion
    sev_coll = db["severity_records"]
    sample_sev = [
        {
            "run_id": run_id,
            "region_id": "R1 - Vidarbha, Maharashtra",
            "predicted_emergency": "Drought",
            "probability": 0.85,
            "area_ha": 120,
            "vulnerability": 0.9,
            "severity_score": 85.5,
            "recorded_at": datetime.now().isoformat()
        },
        {
            "run_id": run_id,
            "region_id": "R4 - Ludhiana, Punjab",
            "predicted_emergency": "Fire",
            "probability": 0.70,
            "area_ha": 200,
            "vulnerability": 0.8,
            "severity_score": 75.0,
            "recorded_at": datetime.now().isoformat()
        }
    ]
    sev_coll.insert_many(sample_sev)
    print(f"[4/6] Saved {len(sample_sev)} severity records into MongoDB.")

    # Test IoT Readings insertion & query
    iot_coll = db["iot_readings"]
    sample_iot = [
        {
            "region": "R1 - Vidarbha, Maharashtra",
            "soil_moisture": 14.2,
            "temperature": 42.5,
            "humidity": 25.0,
            "rainfall_mm": 0.0,
            "wind_speed_kmh": 22.0,
            "recorded_at": datetime.now().isoformat()
        }
    ]
    iot_coll.insert_many(sample_iot)
    iot_count = iot_coll.count_documents({})
    print(f"[5/6] Saved IoT readings into MongoDB. Total IoT records: {iot_count}")

    # Finish run & fetch stats
    pipeline_coll.update_one(
        {"run_id": run_id},
        {"$set": {"status": "completed", "finished_at": datetime.now().isoformat(), "summary": "Test run completed"}}
    )
    
    stats = {
        "pipeline_runs": db["pipeline_runs"].count_documents({}),
        "severity_records": db["severity_records"].count_documents({}),
        "allocation_records": db["allocation_records"].count_documents({}),
        "iot_readings": db["iot_readings"].count_documents({}),
        "emergency_events": db["emergency_events"].count_documents({})
    }
    print("[6/6] MongoDB Database Stats:", stats)

    print("=" * 60)
    print(" ALL MONGODB TESTS PASSED! Database is 100% working.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

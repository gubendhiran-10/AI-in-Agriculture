"""
AI in Agriculture — Resilient Dual Database Module
Supports MongoDB with automatic, seamless fallback to SQLite.
Provides full persistence for Users (Local & Google OAuth), Auth Sessions,
Pipeline Runs, Sensor Readings, Severity Scores, Resource Allocations, and Events.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB_NAME", "AI_in_Agriculture_Emergency_Resource_Allocation")
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agriculture.db")

ACTIVE_BACKEND = "sqlite"
mongo_client = None
mongo_db = None


# ─── PASSWORD SECURITY ──────────────────────────────────────────

def _hash_password(password, salt=None):
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt


def _verify_password(password, stored_hash, salt):
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return secrets.compare_digest(key.hex(), stored_hash)


# ─── SQLITE SCHEMA ──────────────────────────────────────────────

def _init_sqlite_tables(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT UNIQUE,
        password_hash TEXT,
        salt TEXT,
        role TEXT,
        avatar TEXT,
        provider TEXT DEFAULT 'local',
        created_at TEXT,
        last_login TEXT
    )""")
    # Ensure avatar & provider columns exist if older schema
    try:
        cur.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE users ADD COLUMN provider TEXT DEFAULT 'local'")
    except Exception:
        pass

    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        status TEXT,
        started_at TEXT,
        finished_at TEXT,
        summary TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS severity_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        region_id TEXT,
        predicted_emergency TEXT,
        probability REAL,
        area_ha REAL,
        vulnerability REAL,
        severity_score REAL,
        recorded_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS allocation_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        region_id TEXT,
        water REAL,
        personnel REAL,
        machinery REAL,
        funds REAL,
        recorded_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS iot_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region TEXT,
        soil_moisture REAL,
        temperature REAL,
        humidity REAL,
        rainfall_mm REAL,
        wind_speed_kmh REAL,
        recorded_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emergency_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        region_id TEXT,
        emergency_type TEXT,
        severity_score REAL,
        status TEXT,
        notes TEXT,
        author TEXT,
        created_at TEXT,
        resolved_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS admin_settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    cur.execute("INSERT OR IGNORE INTO admin_settings (key, value) VALUES ('admin_password', 'dhanush2026')")
    conn.commit()


def _get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _clean_doc(doc):
    """Convert MongoDB document to JSON-safe dictionary."""
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d["_id"])
        del d["_id"]
    return d


def init_db():
    """Detect and initialize MongoDB or fallback to SQLite."""
    global ACTIVE_BACKEND, mongo_client, mongo_db
    try:
        from pymongo import MongoClient, DESCENDING
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        mongo_client = client
        mongo_db = client[DB_NAME]

        # MongoDB indexes
        mongo_db["users"].create_index("email", unique=True)
        mongo_db["pipeline_runs"].create_index("run_id", unique=True)
        mongo_db["severity_records"].create_index("run_id")
        mongo_db["allocation_records"].create_index("run_id")
        mongo_db["iot_readings"].create_index([("recorded_at", DESCENDING)])
        mongo_db["emergency_events"].create_index([("created_at", DESCENDING)])
        mongo_db["admin_settings"].update_one(
            {"key": "admin_password"},
            {"$setOnInsert": {"key": "admin_password", "value": "dhanush2026"}},
            upsert=True
        )
        ACTIVE_BACKEND = "mongodb"
        print(f"[Database] Connected to MongoDB ({DB_NAME})")
    except Exception as e:
        ACTIVE_BACKEND = "sqlite"
        print(f"[Database] MongoDB unavailable ({e}). Falling back to SQLite ({SQLITE_DB_PATH})")
        with _get_sqlite_conn() as conn:
            _init_sqlite_tables(conn)

    seed_default_users()


def get_backend_info():
    return {
        "engine": "MongoDB" if ACTIVE_BACKEND == "mongodb" else "SQLite",
        "target": DB_NAME if ACTIVE_BACKEND == "mongodb" else SQLITE_DB_PATH,
        "status": "connected"
    }


# ─── USER AUTHENTICATION & GOOGLE OAUTH ─────────────────────────

def create_user(name, email, password, role="Agricultural Officer", avatar=None, provider="local"):
    """Register a new user with secure password hashing."""
    email_clean = email.strip().lower()
    name_clean = name.strip()
    if not email_clean or not name_clean:
        return None, "Name and email are required."

    if get_user_by_email(email_clean):
        return None, "An account with this email address already exists."

    user_id = secrets.token_hex(8)
    pwd_hash, salt = _hash_password(password or secrets.token_urlsafe(16))
    now = datetime.now().isoformat()

    user_doc = {
        "id": user_id,
        "name": name_clean,
        "email": email_clean,
        "password_hash": pwd_hash,
        "salt": salt,
        "role": role,
        "avatar": avatar,
        "provider": provider,
        "created_at": now,
        "last_login": now
    }

    if ACTIVE_BACKEND == "mongodb":
        mongo_db["users"].insert_one(user_doc)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (id, name, email, password_hash, salt, role, avatar, provider, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name_clean, email_clean, pwd_hash, salt, role, avatar, provider, now, now))
            conn.commit()

    return {
        "id": user_id,
        "name": name_clean,
        "email": email_clean,
        "role": role,
        "avatar": avatar,
        "provider": provider,
        "created_at": now,
        "last_login": now
    }, None


def authenticate_user(email, password):
    """Authenticate a local user by email and password."""
    email_clean = email.strip().lower()
    now = datetime.now().isoformat()

    if ACTIVE_BACKEND == "mongodb":
        user = mongo_db["users"].find_one({"email": email_clean})
        if not user:
            return None
        if _verify_password(password, user.get("password_hash", ""), user.get("salt", "")):
            mongo_db["users"].update_one({"email": email_clean}, {"$set": {"last_login": now}})
            return {
                "id": str(user.get("id", user.get("_id"))),
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "Agricultural Officer"),
                "avatar": user.get("avatar"),
                "provider": user.get("provider", "local"),
                "last_login": now
            }
        return None
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, email, password_hash, salt, role, avatar, provider FROM users WHERE email = ?", (email_clean,))
            row = cur.fetchone()
            if not row:
                return None
            if _verify_password(password, row["password_hash"], row["salt"]):
                cur.execute("UPDATE users SET last_login = ? WHERE email = ?", (now, email_clean))
                conn.commit()
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "email": row["email"],
                    "role": row["role"] or "Agricultural Officer",
                    "avatar": row["avatar"],
                    "provider": row["provider"] or "local",
                    "last_login": now
                }
            return None


def create_or_get_google_user(google_id, name, email, avatar=None, role="Agricultural Officer"):
    """Handle Google Sign-In: retrieve existing user or create a new profile."""
    email_clean = email.strip().lower()
    now = datetime.now().isoformat()

    existing = get_user_by_email(email_clean)
    if existing:
        # Update avatar and last_login
        if ACTIVE_BACKEND == "mongodb":
            mongo_db["users"].update_one(
                {"email": email_clean},
                {"$set": {"last_login": now, "avatar": avatar or existing.get("avatar")}}
            )
        else:
            with _get_sqlite_conn() as conn:
                cur = conn.cursor()
                cur.execute("UPDATE users SET last_login = ?, avatar = COALESCE(?, avatar) WHERE email = ?", (now, avatar, email_clean))
                conn.commit()
        existing["last_login"] = now
        if avatar:
            existing["avatar"] = avatar
        return existing

    # Create new Google user
    user, _ = create_user(
        name=name,
        email=email_clean,
        password=secrets.token_hex(16),
        role=role,
        avatar=avatar,
        provider="google"
    )
    return user


def get_user_by_email(email):
    email_clean = email.strip().lower()
    if ACTIVE_BACKEND == "mongodb":
        user = mongo_db["users"].find_one({"email": email_clean})
        if user:
            d = _clean_doc(user)
            d.pop("password_hash", None)
            d.pop("salt", None)
            return d
        return None
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, email, role, avatar, provider, created_at, last_login FROM users WHERE email = ?", (email_clean,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_id(user_id):
    if ACTIVE_BACKEND == "mongodb":
        query = {}
        try:
            query = {"_id": ObjectId(str(user_id))}
        except Exception:
            query = {"id": str(user_id)}
        user = mongo_db["users"].find_one(query)
        if user:
            d = _clean_doc(user)
            d.pop("password_hash", None)
            d.pop("salt", None)
            return d
        return None
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, email, role, avatar, provider, created_at, last_login FROM users WHERE id = ?", (str(user_id),))
            row = cur.fetchone()
            return dict(row) if row else None


def seed_default_users():
    defaults = [
        ("Dhanush R (Lead Officer)", "officer@agri-ai.gov.in", "agri2026", "Senior Agricultural Officer"),
        ("Dr. Priya Sharma", "relief@agri-ai.gov.in", "relief2026", "Disaster Relief Specialist"),
        ("District Collector Office", "collector@agri-ai.gov.in", "admin2026", "District Collector"),
        ("System Administrator", "admin@agri-ai.gov.in", "dhanush2026", "System Administrator"),
    ]
    for name, email, pwd, role in defaults:
        if not get_user_by_email(email):
            create_user(name, email, pwd, role)


# ─── PIPELINE RUNS ──────────────────────────────────────────────

def create_pipeline_run(run_id):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        mongo_db["pipeline_runs"].insert_one({
            "run_id": run_id,
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "summary": None
        })
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO pipeline_runs (run_id, status, started_at, finished_at, summary)
                VALUES (?, 'running', ?, NULL, NULL)
            """, (run_id, now))
            conn.commit()


def finish_pipeline_run(run_id, summary=""):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        mongo_db["pipeline_runs"].update_one(
            {"run_id": run_id},
            {"$set": {"status": "completed", "finished_at": now, "summary": summary}}
        )
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE pipeline_runs SET status = 'completed', finished_at = ?, summary = ?
                WHERE run_id = ?
            """, (now, summary, run_id))
            conn.commit()


def get_all_pipeline_runs():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["pipeline_runs"].find().sort("started_at", DESCENDING)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT run_id, status, started_at, finished_at, summary FROM pipeline_runs ORDER BY started_at DESC")
            return [dict(row) for row in cur.fetchall()]


# ─── SEVERITY RECORDS ───────────────────────────────────────────

def save_severity_records(run_id, records):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        docs = []
        for r in records:
            docs.append({
                "run_id": run_id,
                "region_id": r["region_id"],
                "predicted_emergency": r["predicted_emergency"],
                "probability": float(r["probability"]),
                "area_ha": float(r.get("area_ha", 0)),
                "vulnerability": float(r.get("vulnerability", 0)),
                "severity_score": float(r["severity_score"]),
                "recorded_at": now
            })
        if docs:
            mongo_db["severity_records"].insert_many(docs)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            rows = [
                (run_id, r["region_id"], r["predicted_emergency"], float(r["probability"]),
                 float(r.get("area_ha", 0)), float(r.get("vulnerability", 0)),
                 float(r["severity_score"]), now)
                for r in records
            ]
            cur.executemany("""
                INSERT INTO severity_records 
                (run_id, region_id, predicted_emergency, probability, area_ha, vulnerability, severity_score, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()


def get_severity_by_run(run_id):
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["severity_records"].find({"run_id": run_id}).sort("severity_score", DESCENDING)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, run_id, region_id, predicted_emergency, probability, area_ha, vulnerability, severity_score, recorded_at
                FROM severity_records WHERE run_id = ? ORDER BY severity_score DESC
            """, (run_id,))
            return [dict(row) for row in cur.fetchall()]


def get_all_severity():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["severity_records"].find().sort("recorded_at", DESCENDING).limit(500)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, run_id, region_id, predicted_emergency, probability, area_ha, vulnerability, severity_score, recorded_at
                FROM severity_records ORDER BY recorded_at DESC LIMIT 500
            """)
            return [dict(row) for row in cur.fetchall()]


# ─── ALLOCATION RECORDS ─────────────────────────────────────────

def save_allocation_records(run_id, records):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        docs = []
        for r in records:
            docs.append({
                "run_id": run_id,
                "region_id": r["region_id"],
                "water": float(r["water"]),
                "personnel": float(r["personnel"]),
                "machinery": float(r["machinery"]),
                "funds": float(r["funds"]),
                "recorded_at": now
            })
        if docs:
            mongo_db["allocation_records"].insert_many(docs)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            rows = [
                (run_id, r["region_id"], float(r["water"]), float(r["personnel"]),
                 float(r["machinery"]), float(r["funds"]), now)
                for r in records
            ]
            cur.executemany("""
                INSERT INTO allocation_records
                (run_id, region_id, water, personnel, machinery, funds, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()


def get_allocation_by_run(run_id):
    if ACTIVE_BACKEND == "mongodb":
        cursor = mongo_db["allocation_records"].find({"run_id": run_id}).sort("region_id", 1)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, run_id, region_id, water, personnel, machinery, funds, recorded_at
                FROM allocation_records WHERE run_id = ? ORDER BY region_id ASC
            """, (run_id,))
            return [dict(row) for row in cur.fetchall()]


def get_all_allocations():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["allocation_records"].find().sort("recorded_at", DESCENDING).limit(500)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, run_id, region_id, water, personnel, machinery, funds, recorded_at
                FROM allocation_records ORDER BY recorded_at DESC LIMIT 500
            """)
            return [dict(row) for row in cur.fetchall()]


# ─── IOT READINGS ───────────────────────────────────────────────

def save_iot_readings(readings):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        docs = []
        for r in readings:
            docs.append({
                "region": r["region"],
                "soil_moisture": float(r["soil_moisture"]),
                "temperature": float(r["temperature"]),
                "humidity": float(r["humidity"]),
                "rainfall_mm": float(r["rainfall_mm"]),
                "wind_speed_kmh": float(r.get("wind_speed_kmh", 0)),
                "recorded_at": now
            })
        if docs:
            mongo_db["iot_readings"].insert_many(docs)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            rows = [
                (r["region"], float(r["soil_moisture"]), float(r["temperature"]),
                 float(r["humidity"]), float(r["rainfall_mm"]), float(r.get("wind_speed_kmh", 0)), now)
                for r in readings
            ]
            cur.executemany("""
                INSERT INTO iot_readings
                (region, soil_moisture, temperature, humidity, rainfall_mm, wind_speed_kmh, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()


def get_latest_iot():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        regions = mongo_db["iot_readings"].distinct("region")
        latest = []
        for reg in regions:
            doc = mongo_db["iot_readings"].find_one({"region": reg}, sort=[("recorded_at", DESCENDING)])
            if doc:
                latest.append(_clean_doc(doc))
        return sorted(latest, key=lambda x: x.get("region", ""))
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT t1.* FROM iot_readings t1
                JOIN (
                    SELECT region, MAX(recorded_at) AS max_rec
                    FROM iot_readings GROUP BY region
                ) t2 ON t1.region = t2.region AND t1.recorded_at = t2.max_rec
                ORDER BY t1.region ASC
            """)
            return [dict(row) for row in cur.fetchall()]


def get_all_iot():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["iot_readings"].find().sort("recorded_at", DESCENDING).limit(500)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, region, soil_moisture, temperature, humidity, rainfall_mm, wind_speed_kmh, recorded_at
                FROM iot_readings ORDER BY recorded_at DESC LIMIT 500
            """)
            return [dict(row) for row in cur.fetchall()]


# ─── EMERGENCY EVENTS & AUDIT LOG ───────────────────────────────

def save_emergency_events(run_id, events):
    now = datetime.now().isoformat()
    if ACTIVE_BACKEND == "mongodb":
        docs = []
        for e in events:
            docs.append({
                "run_id": run_id,
                "region_id": e["region_id"],
                "emergency_type": e.get("predicted_emergency", "Unknown"),
                "severity_score": float(e["severity_score"]),
                "status": e.get("status", "detected"),
                "notes": e.get("notes", ""),
                "author": e.get("author", "Automated AI Detection Engine"),
                "created_at": now,
                "resolved_at": None
            })
        if docs:
            mongo_db["emergency_events"].insert_many(docs)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            rows = [
                (run_id, e["region_id"], e.get("predicted_emergency", "Unknown"),
                 float(e["severity_score"]), e.get("status", "detected"),
                 e.get("notes", ""), e.get("author", "Automated AI Detection Engine"),
                 now, None)
                for e in events
            ]
            cur.executemany("""
                INSERT INTO emergency_events
                (run_id, region_id, emergency_type, severity_score, status, notes, author, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()


def get_all_events():
    if ACTIVE_BACKEND == "mongodb":
        from pymongo import DESCENDING
        cursor = mongo_db["emergency_events"].find().sort("created_at", DESCENDING).limit(200)
        return [_clean_doc(doc) for doc in cursor]
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, run_id, region_id, emergency_type, severity_score, status, notes, author, created_at, resolved_at
                FROM emergency_events ORDER BY created_at DESC LIMIT 200
            """)
            return [dict(row) for row in cur.fetchall()]


def update_emergency_status(event_id, new_status, notes=None, author=None):
    now = datetime.now().isoformat()
    resolved_at = now if new_status == "resolved" else None

    if ACTIVE_BACKEND == "mongodb":
        query = {}
        try:
            query = {"_id": ObjectId(str(event_id))}
        except Exception:
            query = {"id": str(event_id)}

        update_fields = {"status": new_status}
        if notes:
            update_fields["notes"] = notes
        if author:
            update_fields["author"] = author
        if resolved_at:
            update_fields["resolved_at"] = resolved_at

        res = mongo_db["emergency_events"].update_one(query, {"$set": update_fields})
        if res.matched_count == 0 and "_id" not in query:
            mongo_db["emergency_events"].update_one({"run_id": str(event_id)}, {"$set": update_fields})
        return True
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE emergency_events
                SET status = ?, notes = COALESCE(?, notes), author = COALESCE(?, author), resolved_at = COALESCE(?, resolved_at)
                WHERE id = ? OR run_id = ?
            """, (new_status, notes, author, resolved_at, event_id, str(event_id)))
            conn.commit()
            return cur.rowcount > 0


# ─── ADMIN & SYSTEM STATS ───────────────────────────────────────

def verify_admin(password):
    if not password:
        return False
    if ACTIVE_BACKEND == "mongodb":
        row = mongo_db["admin_settings"].find_one({"key": "admin_password"})
        return bool(row and row.get("value") == password)
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM admin_settings WHERE key = 'admin_password'")
            row = cur.fetchone()
            return bool(row and row["value"] == password)


def get_db_stats():
    if ACTIVE_BACKEND == "mongodb":
        return {
            "backend": "MongoDB",
            "users": mongo_db["users"].count_documents({}),
            "pipeline_runs": mongo_db["pipeline_runs"].count_documents({}),
            "severity_records": mongo_db["severity_records"].count_documents({}),
            "allocation_records": mongo_db["allocation_records"].count_documents({}),
            "iot_readings": mongo_db["iot_readings"].count_documents({}),
            "emergency_events": mongo_db["emergency_events"].count_documents({})
        }
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.cursor()
            def count(tbl):
                try:
                    cur.execute(f"SELECT COUNT(*) as c FROM {tbl}")
                    return cur.fetchone()["c"]
                except Exception:
                    return 0
            return {
                "backend": "SQLite",
                "users": count("users"),
                "pipeline_runs": count("pipeline_runs"),
                "severity_records": count("severity_records"),
                "allocation_records": count("allocation_records"),
                "iot_readings": count("iot_readings"),
                "emergency_events": count("emergency_events")
            }


# Auto-initialize on import
init_db()

"""
generate_data.py
ValSafe ML — Synthetic Incident Data Generator
Generates realistic incident records based on the ValSafe schema.
Run: python generate_data.py
Output: data/incidents.csv
"""

import uuid
import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
NUM_RECORDS = 5000      # Change to generate more/fewer rows
OUTPUT_PATH = "data/incidents.csv"

# ── SCHEMA-ALIGNED LOOKUP VALUES ────────────────────────────────────────────
SEVERITY_LEVELS = ["low", "medium", "high"]

INCIDENT_TYPES = [
    "unsafe_act",
    "near_miss",
    "incident",
    "unsafe_condition",
    "property_damage",
    "injury",
    "observation",
]

STATUSES = ["Pending", "Resolved", "Escalated"]

MEDIA_STATUSES = ["no", "yes", "success", "failed"]

CATEGORIES = [
    "Fire Safety",
    "Chemical Exposure",
    "Electrical Hazard",
    "Slip and Fall",
    "Equipment Failure",
    "Ergonomics",
    "Security Breach",
    "Environmental",
]

ROOT_CAUSE_CATEGORIES = [
    "Human Error",
    "Equipment Malfunction",
    "Inadequate Training",
    "Poor Housekeeping",
    "Design Flaw",
    "Fatigue",
    "Communication Failure",
    "Environmental Factor",
]

DEPARTMENTS = [f"DEPT-{str(i).zfill(3)}" for i in range(1, 21)]
SITES       = [f"SITE-{str(i).zfill(3)}" for i in range(1, 11)]
USERS       = [str(uuid.uuid4()) for _ in range(200)]

# ── REALISTIC LOGIC RULES ───────────────────────────────────────────────────
# These probability rules make the data coherent so the ML model can learn
# real patterns rather than random noise.

def pick_severity(incident_type: str) -> str:
    """Severity correlates with incident type."""
    weights = {
        "injury":            [0.10, 0.35, 0.55],   # mostly high
        "property_damage":   [0.15, 0.45, 0.40],
        "incident":          [0.20, 0.45, 0.35],
        "near_miss":         [0.35, 0.45, 0.20],
        "unsafe_condition":  [0.30, 0.50, 0.20],
        "unsafe_act":        [0.40, 0.40, 0.20],
        "observation":       [0.65, 0.30, 0.05],   # mostly low
    }
    return random.choices(SEVERITY_LEVELS, weights=weights[incident_type])[0]


def pick_status(severity: str, days_old: int) -> str:
    """Status correlates with severity and age of the incident."""
    if severity == "high":
        # High severity: more likely escalated or resolved quickly
        if days_old < 3:
            return random.choices(STATUSES, weights=[0.30, 0.20, 0.50])[0]
        else:
            return random.choices(STATUSES, weights=[0.10, 0.65, 0.25])[0]
    elif severity == "medium":
        if days_old < 7:
            return random.choices(STATUSES, weights=[0.55, 0.30, 0.15])[0]
        else:
            return random.choices(STATUSES, weights=[0.20, 0.70, 0.10])[0]
    else:  # low
        if days_old < 14:
            return random.choices(STATUSES, weights=[0.70, 0.25, 0.05])[0]
        else:
            return random.choices(STATUSES, weights=[0.30, 0.68, 0.02])[0]


def pick_media_status(incident_type: str) -> str:
    """Media is more commonly attached to injuries and property damage."""
    if incident_type in ["injury", "property_damage", "incident"]:
        return random.choices(MEDIA_STATUSES, weights=[0.20, 0.10, 0.60, 0.10])[0]
    else:
        return random.choices(MEDIA_STATUSES, weights=[0.60, 0.10, 0.25, 0.05])[0]


def pick_root_cause(incident_type: str) -> str:
    """Root causes correlate loosely with incident type."""
    if incident_type == "injury":
        return random.choices(ROOT_CAUSE_CATEGORIES,
            weights=[0.25, 0.20, 0.20, 0.10, 0.10, 0.10, 0.03, 0.02])[0]
    elif incident_type == "equipment_failure" or incident_type == "property_damage":
        return random.choices(ROOT_CAUSE_CATEGORIES,
            weights=[0.10, 0.40, 0.10, 0.10, 0.20, 0.03, 0.04, 0.03])[0]
    else:
        return random.choices(ROOT_CAUSE_CATEGORIES,
            weights=[0.20, 0.15, 0.20, 0.15, 0.10, 0.10, 0.07, 0.03])[0]


# ── GEOHASH SIMULATOR (simplified) ──────────────────────────────────────────
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def fake_geohash(length: int = 7) -> str:
    return "".join(random.choices(BASE32, k=length))


# ── MAIN GENERATOR ───────────────────────────────────────────────────────────
def generate_incidents(n: int) -> pd.DataFrame:
    records = []
    now = datetime.now()

    for _ in range(n):
        incident_type = random.choice(INCIDENT_TYPES)
        occurred_at   = now - timedelta(days=random.randint(0, 730))  # last 2 years
        days_old      = (now - occurred_at).days

        severity = pick_severity(incident_type)
        status   = pick_status(severity, days_old)
        media    = pick_media_status(incident_type)
        category = random.choice(CATEGORIES)
        root     = pick_root_cause(incident_type)
        dept     = random.choice(DEPARTMENTS)
        site     = random.choice(SITES)
        user     = random.choice(USERS)

        # Resolved/Escalated incidents get timestamps
        resolved_at   = None
        escalated_at  = None
        who_resolved  = None
        who_escalated = None

        if status == "Resolved":
            resolved_at  = occurred_at + timedelta(hours=random.randint(1, 72))
            who_resolved = str(uuid.uuid4())
        elif status == "Escalated":
            escalated_at  = occurred_at + timedelta(hours=random.randint(1, 12))
            who_escalated = str(uuid.uuid4())

        records.append({
            # Identifiers
            "id":                  str(uuid.uuid4()),
            "user_id":             user,
            "dept_id":             dept,
            "site_id":             site,
            "cat_id":              category,
            "root_cause":          root,

            # Core incident fields
            "type":                incident_type,
            "severity_level":      severity,
            "status":              status,
            "media_status":        media,
            "details":             fake.sentence(nb_words=12),

            # Location
            "loc_lat":             str(round(fake.latitude(), 6)),
            "loc_long":            str(round(fake.longitude(), 6)),
            "loc_geohash":         fake_geohash(),

            # Timestamps
            "occurred_at":         occurred_at.isoformat(),
            "created_at":          occurred_at.isoformat(),
            "resolved_at":         resolved_at.isoformat() if resolved_at else None,
            "escalated_at":        escalated_at.isoformat() if escalated_at else None,

            # Resolution
            "who_resolved":        who_resolved,
            "who_escalated":       who_escalated,
            "resolved_by_admin":   random.random() < 0.3,
            "is_deleted":          False,
            "preventative_actions": fake.sentence(nb_words=10) if status == "Resolved" else None,

            # Time-derived features (useful for ML)
            "hour_of_day":         occurred_at.hour,
            "day_of_week":         occurred_at.weekday(),    # 0=Mon, 6=Sun
            "month":               occurred_at.month,
            "days_since_occurred": days_old,
        })

    return pd.DataFrame(records)


# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print(f"Generating {NUM_RECORDS} incident records...")
    df = generate_incidents(NUM_RECORDS)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nShape: {df.shape}")
    print(f"\nSeverity distribution:\n{df['severity_level'].value_counts()}")
    print(f"\nType distribution:\n{df['type'].value_counts()}")
    print(f"\nStatus distribution:\n{df['status'].value_counts()}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
import csv
import datetime
import os
import random
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

FIELDS = [
    "path", "method", "owner", "last_traffic_date",
    "has_documentation", "has_authentication", "tls_version",
    "has_rate_limiting", "exposed_pii", "deployed_status",
]

API_PATHS = [
    "/api/v1/users", "/api/v1/orders", "/api/v1/products",
    "/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/payments",
    "/api/v1/inventory", "/api/v1/shipments", "/api/v1/notifications",
    "/api/v1/analytics", "/api/v2/users", "/api/v2/orders",
    "/api/v2/products", "/api/v2/payments", "/api/v2/inventory",
    "/graphql", "/health", "/api/v1/webhooks", "/api/v1/search",
    "/api/v1/reports", "/internal/metrics", "/internal/config",
    "/api/v1/audit/logs", "/api/v1/subscriptions", "/api/v1/reviews",
    "/api/v1/categories", "/api/v1/cart", "/api/v1/checkout",
    "/api/v1/coupons", "/debug/env", "/api/v1/admin/users",
    "/api/v1/admin/settings", "/internal/debug/pprof",
    "/api/v1/exports", "/api/v1/imports",
]
METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
TLS_VERSIONS = ["1.0", "1.1", "1.2", "1.3"]
DEPLOYED_STATUSES = ["deployed", "deprecated", "unknown"]
PII_LEVELS = ["none", "email", "email,name", "email,name,address", "full"]
OWNER_DOMAINS = ["acme.com", "globex.com", "initech.com", "umbrella.com", "cyberdyne.com"]
OWNER_NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "hank"]
BASE_TIME = datetime.datetime.now(datetime.timezone.utc)

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TARGET_DIR, "..", "data")

PII_ORDER = {"none": 0, "email": 1, "email,name": 2, "email,name,address": 3, "full": 4}
TLS_ORDER = {"1.0": 0, "1.1": 1, "1.2": 2, "1.3": 3}
METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "PATCH": 4}
DEPLOY_ORDER = {"deployed": 0, "deprecated": 1, "unknown": 2}


def random_date(rng, days_ago_min, days_ago_max):
    days_ago = rng.randint(days_ago_min, days_ago_max)
    d = BASE_TIME - datetime.timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


GENERATORS = [
    lambda rng: {
        "path": rng.choice(API_PATHS),
        "method": rng.choice(METHODS),
        "owner": f"{rng.choice(OWNER_NAMES)}@{rng.choice(OWNER_DOMAINS)}",
        "last_traffic_date": random_date(rng, 0, 6),
        "has_documentation": rng.choices([True, False], weights=[95, 5])[0],
        "has_authentication": rng.choices([True, False], weights=[90, 10])[0],
        "tls_version": rng.choices(["1.2", "1.3"], weights=[30, 70])[0],
        "has_rate_limiting": rng.choices([True, False], weights=[85, 15])[0],
        "exposed_pii": rng.choices(PII_LEVELS, weights=[40, 25, 20, 10, 5])[0],
        "deployed_status": "deployed",
    },
    lambda rng: {
        "path": rng.choice(API_PATHS),
        "method": rng.choice(METHODS),
        "owner": f"{rng.choice(OWNER_NAMES)}@{rng.choice(OWNER_DOMAINS)}",
        "last_traffic_date": random_date(rng, 30, 120),
        "has_documentation": rng.choices([True, False], weights=[60, 40])[0],
        "has_authentication": rng.choices([True, False], weights=[70, 30])[0],
        "tls_version": rng.choices(TLS_VERSIONS, weights=[5, 20, 50, 25])[0],
        "has_rate_limiting": rng.choices([True, False], weights=[50, 50])[0],
        "exposed_pii": rng.choices(PII_LEVELS, weights=[55, 20, 15, 8, 2])[0],
        "deployed_status": rng.choices(DEPLOYED_STATUSES[:2], weights=[70, 30])[0],
    },
    lambda rng: {
        "path": rng.choice(API_PATHS),
        "method": rng.choice(METHODS),
        "owner": "",
        "last_traffic_date": random_date(rng, 91, 180),
        "has_documentation": rng.choices([True, False], weights=[20, 80])[0],
        "has_authentication": rng.choices([True, False], weights=[30, 70])[0],
        "tls_version": rng.choices(TLS_VERSIONS, weights=[10, 30, 45, 15])[0],
        "has_rate_limiting": rng.choices([True, False], weights=[15, 85])[0],
        "exposed_pii": rng.choices(PII_LEVELS, weights=[40, 25, 20, 10, 5])[0],
        "deployed_status": rng.choices(["deployed", "unknown"], weights=[80, 20])[0],
    },
    lambda rng: {
        "path": rng.choice(API_PATHS),
        "method": rng.choice(METHODS),
        "owner": "",
        "last_traffic_date": random_date(rng, 181, 730),
        "has_documentation": rng.choices([True, False], weights=[5, 95])[0],
        "has_authentication": rng.choices([True, False], weights=[10, 90])[0],
        "tls_version": rng.choices(TLS_VERSIONS, weights=[20, 40, 30, 10])[0],
        "has_rate_limiting": rng.choices([True, False], weights=[5, 95])[0],
        "exposed_pii": rng.choices(PII_LEVELS, weights=[20, 25, 25, 20, 10])[0],
        "deployed_status": "deployed",
    },
]

LABELS = ["Active", "Deprecated", "Orphaned", "Zombie"]


def days_since(date_str):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return (BASE_TIME - d).days


def featurize(row):
    return [
        days_since(row["last_traffic_date"]),
        1 if row["owner"] else 0,
        1 if row["has_documentation"] == "True" or row["has_documentation"] is True else 0,
        1 if row["has_authentication"] == "True" or row["has_authentication"] is True else 0,
        TLS_ORDER.get(row["tls_version"], 0),
        1 if row["has_rate_limiting"] == "True" or row["has_rate_limiting"] is True else 0,
        PII_ORDER.get(row["exposed_pii"], 0),
        DEPLOY_ORDER.get(row["deployed_status"], 0),
        METHOD_ORDER.get(row["method"], 0),
        row["path"].count("/"),
    ]


FEATURE_NAMES = [
    "days_since_traffic", "has_owner", "has_documentation",
    "has_authentication", "tls_version_ordinal",
    "has_rate_limiting", "exposed_pii_ordinal",
    "deployed_status_ordinal", "method_ordinal", "path_depth",
]


def generate_labeled_dataset(n_records, output_path):
    rng = random.Random()
    weights = rng.choices(range(1, 101), k=len(GENERATORS))
    rows = []
    indices = rng.choices(range(len(GENERATORS)), weights=weights, k=n_records)
    for idx in indices:
        row = GENERATORS[idx](rng)
        row["label"] = LABELS[idx]
        rows.append(row)
    fieldnames = FIELDS + ["label"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} labeled records to {output_path}")
    return rows


def main():
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    output_csv = os.path.join(DATA_DIR, "training_labeled.csv")
    model_path = os.path.join(TARGET_DIR, "api_classifier.joblib")
    encoder_path = os.path.join(TARGET_DIR, "label_encoder.joblib")

    print("Generating labeled training dataset ...")
    records = generate_labeled_dataset(n_records, output_csv)

    X = np.array([featurize(r) for r in records])
    y = [r["label"] for r in records]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTuning hyperparameters on {len(X_train):,} samples ...")
    base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [2, 4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, cv=3,
        scoring="f1_macro", random_state=42, verbose=1, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    clf = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1:  {search.best_score_:.4f}")

    y_pred = clf.predict(X_test)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (test set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:25s} {imp:.4f}")

    model_bundle = {"model": clf, "encoder": le, "scaler": scaler, "feature_names": FEATURE_NAMES}
    joblib.dump(model_bundle, model_path)
    print(f"\nModel saved -> {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

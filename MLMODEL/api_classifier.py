import datetime
import os

import joblib
import numpy as np

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TARGET_DIR, "api_classifier.joblib")

PII_ORDER = {"none": 0, "email": 1, "email,name": 2, "email,name,address": 3, "full": 4}
TLS_ORDER = {"1.0": 0, "1.1": 1, "1.2": 2, "1.3": 3}
METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "PATCH": 4}
DEPLOY_ORDER = {"deployed": 0, "deprecated": 1, "unknown": 2}
BASE_TIME = datetime.datetime.now(datetime.timezone.utc)


def _days_since(date_str):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return (BASE_TIME - d).days


def _featurize(row):
    return [
        _days_since(row["last_traffic_date"]),
        1 if row.get("owner") else 0,
        1 if row.get("has_documentation") in (True, "True") else 0,
        1 if row.get("has_authentication") in (True, "True") else 0,
        TLS_ORDER.get(row.get("tls_version", ""), 0),
        1 if row.get("has_rate_limiting") in (True, "True") else 0,
        PII_ORDER.get(row.get("exposed_pii", ""), 0),
        DEPLOY_ORDER.get(row.get("deployed_status", ""), 0),
        METHOD_ORDER.get(row.get("method", ""), 0),
        str(row.get("path", "")).count("/"),
    ]


class APIClassifier:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            self.model = bundle["model"]
            self.encoder = bundle["encoder"]
            self.scaler = bundle.get("scaler")
        else:
            self.model = bundle
            self.encoder = joblib.load(os.path.join(TARGET_DIR, "label_encoder.joblib"))
            self.scaler = None

    def predict(self, row):
        """Classify a single API record dict.

        Returns one of: 'Active', 'Deprecated', 'Orphaned', 'Zombie'.
        """
        vec = np.array([_featurize(row)], dtype=float)
        if self.scaler:
            vec = self.scaler.transform(vec)
        pred = self.model.predict(vec)[0]
        return self.encoder.inverse_transform([pred])[0]

    def predict_proba(self, row):
        """Return per-class probabilities as a dict."""
        vec = np.array([_featurize(row)], dtype=float)
        if self.scaler:
            vec = self.scaler.transform(vec)
        probs = self.model.predict_proba(vec)[0]
        return dict(zip(self.encoder.classes_, probs))

    def predict_batch(self, rows):
        """Classify many rows at once. Returns list of labels."""
        X = np.array([_featurize(r) for r in rows], dtype=float)
        if self.scaler:
            X = self.scaler.transform(X)
        preds = self.model.predict(X)
        return self.encoder.inverse_transform(preds).tolist()

    def predict_proba_batch(self, rows):
        """Return list of per-class probability dicts for many rows."""
        X = np.array([_featurize(r) for r in rows], dtype=float)
        if self.scaler:
            X = self.scaler.transform(X)
        probas = self.model.predict_proba(X)
        return [dict(zip(self.encoder.classes_, p)) for p in probas]

    def risk_score(self, row):
        """Compute security risk score 0-100."""
        score = 0
        if row.get("has_authentication") in (None, False, "False", ""):
            score += 25
        tls = row.get("tls_version", "")
        if tls in ("1.0", "1.1", None, ""):
            score += 15
        if row.get("has_rate_limiting") in (None, False, "False", ""):
            score += 15
        pii = row.get("exposed_pii", "none")
        if pii and pii != "none":
            score += 20
        if not row.get("owner"):
            score += 10
        return min(score, 100)

    def classify_and_score(self, row):
        """Return dict with classification label, probabilities, and risk score."""
        return {
            "label": self.predict(row),
            "probabilities": self.predict_proba(row),
            "risk_score": self.risk_score(row),
        }


def classify_csv(csv_path, model_path=MODEL_PATH, max_rows=None):
    """Run classification and risk scoring on every row in a CSV. Uses batch prediction."""
    import csv

    clf = APIClassifier(model_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if max_rows is not None:
        rows = rows[:max_rows]

    labels = clf.predict_batch(rows)
    probas = clf.predict_proba_batch(rows)
    results = []
    for row, label, proba in zip(rows, labels, probas):
        results.append({
            "path": row.get("path", ""),
            "method": row.get("method", ""),
            "owner": row.get("owner", ""),
            "last_traffic_date": row.get("last_traffic_date", ""),
            "label": label,
            "probabilities": proba,
            "risk_score": clf.risk_score(row),
        })
    return results


def classify_csv_chunked(csv_path, chunk_size, output_pattern, model_path=MODEL_PATH):
    """Read CSV, classify in batches, write each chunk to a separate file."""
    import csv as csv_mod

    clf = APIClassifier(model_path)
    fieldnames = ["path", "method", "owner", "last_traffic_date", "label", "risk_score"]
    chunk_num = 0

    with open(csv_path, newline="") as f:
        reader = list(csv_mod.DictReader(f))

    offset = 0
    while offset < len(reader):
        chunk = reader[offset:offset + chunk_size]
        labels = clf.predict_batch(chunk)
        chunk_num += 1
        out_path = output_pattern.format(chunk_num)
        with open(out_path, "w", newline="") as out:
            writer = csv_mod.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row, label in zip(chunk, labels):
                writer.writerow({
                    "path": row.get("path", ""),
                    "method": row.get("method", ""),
                    "owner": row.get("owner", ""),
                    "last_traffic_date": row.get("last_traffic_date", ""),
                    "label": label,
                    "risk_score": clf.risk_score(row),
                })
        print(f"  Wrote {len(chunk)} rows -> {out_path}")
        offset += chunk_size

    print(f"Done. {chunk_num} chunk(s) written.")


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python api_classifier.py <csv_path> [max_rows] [--csv output.csv]")
        print("  python api_classifier.py <csv_path> --chunk <size> --out <output_pattern>")
        sys.exit(1)

    csv_path = sys.argv[1]
    args = sys.argv[2:]

    max_rows = 20
    output_csv = None
    chunk_size = None
    output_pattern = None
    i = 0
    while i < len(args):
        if args[i] == "--csv" and i + 1 < len(args):
            output_csv = args[i + 1]
            i += 2
        elif args[i] == "--chunk" and i + 1 < len(args):
            chunk_size = int(args[i + 1])
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            output_pattern = args[i + 1]
            i += 2
        else:
            max_rows = int(args[i])
            i += 1

    if chunk_size is not None:
        if output_pattern is None:
            output_pattern = "output_{:03d}.csv"
        classify_csv_chunked(csv_path, chunk_size, output_pattern)
        sys.exit(0)

    results = classify_csv(csv_path, max_rows=max_rows)

    if output_csv:
        import csv as csv_mod

        fieldnames = ["path", "method", "owner", "last_traffic_date", "label", "risk_score"]
        with open(output_csv, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "path": r["path"],
                    "method": r["method"],
                    "owner": r["owner"],
                    "last_traffic_date": r["last_traffic_date"],
                    "label": r["label"],
                    "risk_score": r["risk_score"],
                })
        print(f"Wrote {len(results)} results to {output_csv}")
    else:
        results_clean = [{k: v for k, v in r.items() if k != "probabilities"} for r in results]
        print(json.dumps(results_clean, indent=2, default=str))

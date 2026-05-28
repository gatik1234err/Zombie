import csv
import random
import datetime
import os
import sys
import time

FIELDS = [
    "path",
    "method",
    "owner",
    "last_traffic_date",
    "has_documentation",
    "has_authentication",
    "tls_version",
    "has_rate_limiting",
    "exposed_pii",
    "deployed_status",
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


def random_date(rng, days_ago_min, days_ago_max):
    days_ago = rng.randint(days_ago_min, days_ago_max)
    d = BASE_TIME - datetime.timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


PROFILE_NAMES = ["Active", "Deprecated", "Orphaned", "Zombie"]


def generate_batch(rng, batch_size, weights):
    rows = []
    indices = rng.choices(range(len(GENERATORS)), weights=weights, k=batch_size)
    for idx in indices:
        rows.append(GENERATORS[idx](rng))
    return rows


GENERATORS = [
    # Active
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
    # Deprecated
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
    # Orphaned
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
    # Zombie
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


def merge_chunks(chunk_paths, output_path):
    print(f"Merging {len(chunk_paths)} files into {output_path} ...")
    first = True
    with open(output_path, "w", newline="") as out:
        writer = None
        for cp in sorted(chunk_paths):
            with open(cp, newline="") as f:
                reader = csv.DictReader(f)
                if first:
                    writer = csv.DictWriter(out, fieldnames=FIELDS)
                    writer.writeheader()
                    first = False
                for row in reader:
                    writer.writerow(row)
            os.remove(cp)
    return output_path


def main():
    total_records = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random()
    weights = rng.choices(range(1, 101), k=len(GENERATORS))
    total_w = sum(weights)
    pcts = [w * 100 / total_w for w in weights]
    print(f"Generating {total_records:,} API records...")
    print(f"  Active:      ({pcts[0]:.1f}%)")
    print(f"  Deprecated:  ({pcts[1]:.1f}%)")
    print(f"  Orphaned:    ({pcts[2]:.1f}%)")
    print(f"  Zombie:      ({pcts[3]:.1f}%)")
    print()

    WRITE_BATCH = 500_000

    t0 = time.time()

    if total_records >= 100_000_000:
        batch_paths = []
        written = 0
        batch_num = 0
        batch_size = 5_000_000
        while written < total_records:
            batch_num += 1
            batch_count = min(batch_size, total_records - written)
            filepath = os.path.join(output_dir, f"batch_{batch_num:04d}.csv")
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                pos = 0
                while pos < batch_count:
                    remaining = min(WRITE_BATCH, batch_count - pos)
                    rows = generate_batch(rng, remaining, weights)
                    writer.writerows(rows)
                    pos += remaining
            batch_paths.append(filepath)
            print(f"  Batch {batch_num}: {filepath} ({batch_count:,} records)")
            written += batch_count
            batch_size = int(batch_size * 1.02)

        elapsed = time.time() - t0
        rate = total_records / elapsed
        print(f"Generated {total_records:,} records in {elapsed:.1f}s ({rate:,.0f} records/sec)")

        final_path = os.path.join(output_dir, "zombie.csv")
        merge_chunks(batch_paths, final_path)

        total_elapsed = time.time() - t0
        print(f"\nDone. File: {final_path}")
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
    elif total_records >= 1_000_000:
        fifty_millions = total_records // 50_000_000
        FILE_BATCH = int(100_000 * (1.10 ** fifty_millions))

        batch_paths = []
        written = 0
        batch_num = 0
        while written < total_records:
            batch_num += 1
            batch_count = min(FILE_BATCH, total_records - written)
            filepath = os.path.join(output_dir, f"batch_{batch_num:04d}.csv")
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                pos = 0
                while pos < batch_count:
                    remaining = min(WRITE_BATCH, batch_count - pos)
                    rows = generate_batch(rng, remaining, weights)
                    writer.writerows(rows)
                    pos += remaining
            batch_paths.append(filepath)
            print(f"  Batch {batch_num}: {filepath} ({batch_count:,} records)")
            written += batch_count

        elapsed = time.time() - t0
        rate = total_records / elapsed
        print(f"Generated {total_records:,} records in {elapsed:.1f}s ({rate:,.0f} records/sec)")

        final_path = os.path.join(output_dir, "zombie.csv")
        merge_chunks(batch_paths, final_path)

        total_elapsed = time.time() - t0
        print(f"\nDone. File: {final_path}")
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
    else:
        final_path = os.path.join(output_dir, "zombie.csv")
        with open(final_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            written = 0
            while written < total_records:
                remaining = min(WRITE_BATCH, total_records - written)
                rows = generate_batch(rng, remaining, weights)
                writer.writerows(rows)
                written += remaining

        elapsed = time.time() - t0
        rate = total_records / elapsed
        print(f"Generated {total_records:,} records in {elapsed:.1f}s ({rate:,.0f} records/sec)")
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

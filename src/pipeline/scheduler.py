from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── path setup 
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

# ── config 
S3_BUCKET         = os.environ.get("S3_BUCKET", "credit-risk-monitoring")
S3_REGION         = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
SIMULATION_START  = (2018, 1)   # (year, month) of month_idx=1
N_MONTHS          = 12          # total simulation window
N_APPS_PER_MONTH  = 10_000

# ── S3 client 

def get_s3_client():
    try:
        import boto3
        endpoint = os.environ.get("AWS_ENDPOINT_URL", None)
        return boto3.client("s3", region_name=S3_REGION, endpoint_url=endpoint)
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 integration.\n"
            "Install with:  pip install boto3"
        )


class S3Store:

    def __init__(self, bucket: str = S3_BUCKET, use_s3: bool = True):
        self.bucket   = bucket
        self.use_s3   = use_s3
        self._client  = None

        if use_s3:
            try:
                self._client = get_s3_client()
                log.info(f"S3Store initialised — bucket: s3://{bucket}")
            except (ImportError, Exception) as e:
                log.warning(f"S3 unavailable ({e}) — falling back to local storage")
                self.use_s3 = False

    # ── upload 

    def upload(self, local_path: Path, s3_key: str) -> bool:
        if not self.use_s3:
            return False
        try:
            self._client.upload_file(str(local_path), self.bucket, s3_key)
            log.info(f"  ↑ s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            log.error(f"  Upload failed: {s3_key} — {e}")
            return False

    def upload_bytes(self, data: bytes, s3_key: str, content_type: str = "text/csv") -> bool:
        if not self.use_s3:
            return False
        try:
            self._client.put_object(Body=data, Bucket=self.bucket, Key=s3_key, ContentType=content_type)
            log.info(f"  ↑ s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            log.error(f"  Upload failed: {s3_key} — {e}")
            return False

    # ── download 

    def download(self, s3_key: str, local_path: Path) -> bool:
        if not self.use_s3:
            return False
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, s3_key, str(local_path))
            log.info(f"  ↓ s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            log.warning(f"  Download failed: {s3_key} — {e}")
            return False

    def list_keys(self, prefix: str) -> list[str]:
        if not self.use_s3:
            return []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except Exception as e:
            log.error(f"  list_keys failed: {prefix} — {e}")
            return []

    def sync_monitoring_results(self, results_dir: Path) -> None:
        for f in results_dir.glob("*"):
            if f.is_file():
                s3_key = f"monitoring_results/{f.name}"
                self.upload(f, s3_key)

    def pull_production_data(self, local_prod_dir: Path) -> list[Path]:
        keys = self.list_keys("production/")
        pulled = []
        for key in keys:
            fname      = Path(key).name
            local_path = local_prod_dir / fname
            if not local_path.exists():   # only download new files
                if self.download(key, local_path):
                    pulled.append(local_path)
        return pulled


# Monthly ingestion job

def _current_month_idx(now: datetime = None) -> tuple[int, int, int]:
    now = now or datetime.utcnow()
    start_year, start_month = SIMULATION_START
    month_idx = (now.year - start_year) * 12 + (now.month - start_month) + 1
    return now.year, now.month, month_idx


def run_monthly_ingestion(dry_run: bool = False) -> None:
    log.info("=" * 60)
    log.info("MONTHLY INGESTION JOB")
    log.info("=" * 60)

    year, month, month_idx = _current_month_idx()
    log.info(f"Calendar month : {year}-{month:02d}")
    log.info(f"Simulation idx : {month_idx} / {N_MONTHS}")

    if month_idx < 1 or month_idx > N_MONTHS:
        log.warning(f"Month index {month_idx} outside simulation window [1, {N_MONTHS}]. Exiting.")
        return

    store   = S3Store(bucket=S3_BUCKET)
    prod_dir = PROJECT_ROOT / "data" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate this month's batch 
    log.info(f"\n[1/4] Generating production batch for {year}-{month:02d}...")
    from simulation.generate_production import generate_next_month
    local_csv = generate_next_month(year=year, month=month, month_idx=month_idx,
                                    n_months=N_MONTHS, n_apps=N_APPS_PER_MONTH)

    if dry_run:
        log.info("Dry run — skipping S3 upload and pipeline execution.")
        return

    # ── 2. Push new batch to S3 
    log.info(f"\n[2/4] Uploading {local_csv.name} to S3...")
    store.upload(local_csv, f"production/{local_csv.name}")

    # ── 3. Pull any other new production files (if running on fresh instance) ─
    log.info("\n[3/4] Syncing production files from S3...")
    pulled = store.pull_production_data(prod_dir)
    log.info(f"  Downloaded {len(pulled)} new file(s)")

    # ── 4. Run monitoring pipeline 
    log.info("\n[4/4] Running monitoring pipeline...")
    from pipeline.monitoring_pipeline import run_pipeline
    results_dir = PROJECT_ROOT / "data" / "monitoring_results"
    run_pipeline(production_dir=prod_dir, output_dir=results_dir)

    # ── 5. Push results to S3 
    log.info("\nPushing monitoring results to S3...")
    store.sync_monitoring_results(results_dir)

    log.info("\nMonthly ingestion complete.")


# S3-aware data loader (used by Streamlit dashboard)

def load_results_from_s3(cache_dir: Path = None) -> dict:
    import pandas as pd
    import json

    cache_dir = cache_dir or (PROJECT_ROOT / "data" / "monitoring_results")
    store     = S3Store(bucket=S3_BUCKET)

    result_files = [
        "performance.csv", "calibration.csv", "portfolio.csv",
        "psi_timeseries.csv", "prediction_drift.csv",
        "alerts.csv", "alert_summary.csv",
    ]

    # Pull latest results from S3 into local cache
    if store.use_s3:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for fname in result_files + ["latest_alert.json"]:
            store.download(f"monitoring_results/{fname}", cache_dir / fname)

    # Load from cache
    data = {}
    for fname in result_files:
        key = fname.replace(".csv", "").replace("psi_timeseries", "psi").replace("prediction_drift", "pred_drift").replace("alert_summary", "alert_summary")
        path = cache_dir / fname
        data[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()

    jp = cache_dir / "latest_alert.json"
    data["aj"] = json.loads(jp.read_text()) if jp.exists() else {}
    return data


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Monitoring Scheduler")
    parser.add_argument("--run-month",   action="store_true", help="Run monthly ingestion job (for cron)")
    parser.add_argument("--dry-run",     action="store_true", help="Generate data but skip S3 and pipeline")
    parser.add_argument("--sync-to-s3",  action="store_true", help="Push existing local results to S3")
    parser.add_argument("--sync-from-s3",action="store_true", help="Pull all production files from S3")
    parser.add_argument("--init-bucket", action="store_true", help="Upload baseline and models to S3")
    args = parser.parse_args()

    store = S3Store(bucket=S3_BUCKET)

    if args.run_month:
        run_monthly_ingestion(dry_run=args.dry_run)

    elif args.sync_to_s3:
        log.info("Syncing monitoring results to S3...")
        store.sync_monitoring_results(PROJECT_ROOT / "data" / "monitoring_results")

        log.info("Pulling production files from S3...")
        log.info(f"Downloaded {len(pulled)} files")

    elif args.init_bucket:
        log.info("Initialising S3 bucket with baseline and model artifacts...")
        baseline = PROJECT_ROOT / "data" / "baseline" / "train_snapshot.csv"
        if baseline.exists():
            store.upload(baseline, "baseline/train_snapshot.csv")
        for pkl in (PROJECT_ROOT / "models").glob("*.pkl"):
            store.upload(pkl, f"models/{pkl.name}")

    else:
        parser.print_help()

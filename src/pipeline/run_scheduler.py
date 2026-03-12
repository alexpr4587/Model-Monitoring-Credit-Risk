import logging
import time
import schedule
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_pipeline():
    log.info("=" * 50)
    log.info("Scheduled run triggered")
    log.info("=" * 50)
    try:
        from pipeline.monitoring_pipeline import run_pipeline as _run
        _run()
        log.info("Pipeline completed successfully")
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)


def should_run_today() -> bool:
    return datetime.now().day == 1


def monthly_job():
    if should_run_today():
        run_pipeline()
    else:
        log.info(f"Not the 1st of the month ({datetime.now().day}), skipping")


if __name__ == "__main__":
    log.info("Scheduler starting...")

    # Run immediately on startup so dashboard has data right away
    log.info("Running initial pipeline on startup...")
    run_pipeline()

    # Schedule monthly check (runs every day at 06:00, but only executes on day 1)
    schedule.every().day.at("06:00").do(monthly_job)
    log.info("Scheduled: monthly pipeline run on day 1 at 06:00")

    while True:
        schedule.run_pending()
        time.sleep(60)

"""
Pytest configuration for the Plotwise test suite.

This module runs BEFORE any test module is imported, so setting the database
path here guarantees the app (imported at the top of test_api.py) opens a
throwaway database instead of the real demo DB (data/plotwise.db). Without this,
tests that POST disease detections would inject phantom rows into the database
the demo heatmap and PDF report read from.
"""

import atexit
import os
import shutil
import tempfile

# Point the app at a disposable DB before `from backend.src.main import app` runs.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="plotwise_test_")
os.environ["PLOTWISE_DB_PATH"] = os.path.join(_TEST_DB_DIR, "test_plotwise.db")

# Never seed demo data into the test DB, even if the developer has it enabled.
os.environ["PLOTWISE_SEED_ON_EMPTY"] = "0"


@atexit.register
def _cleanup_test_db():
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)

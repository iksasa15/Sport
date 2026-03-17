"""Configuration and constants.

All magic numbers and tunables are centralized here for maintainability.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Version (semantic versioning)
VERSION = "1.0.0"

# Paths (centralized)
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"
# Sport-specific structure: sports/{SportName}/videos|models|tests|reports
SPORTS_ROOT = BASE_DIR / "sports"
# Training data: per-sport collected features, scores, errors (for continuous improvement)
TRAINING_DATA_DIR = BASE_DIR / "training_data"

# Performance
MAX_STREAM_FRAMES = 500  # Limit frames buffered for SSE
FRAME_SKIP = 3  # Default: every 3rd frame (~3x speedup, balanced quality)
ANALYSIS_TIMEOUT_SEC = 3600  # 1 hour max per video

# Job cleanup
JOB_EXPIRY_MINUTES = 60

# Ensure directories exist
for d in (UPLOADS_DIR, OUTPUT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Video preprocessing: optimized for speed (short videos analyzed quickly)
VIDEO_TARGET_HEIGHT = 480  # Resize to 480p for max speed (540=balanced, 720=quality)
FRAME_SKIP_FAST = 4  # Every 4th frame in fast mode (~4x speedup)
CHUNK_SECONDS = 8  # Split long videos into chunks (5-10 sec) for memory efficiency
USE_VIDEO_STABILIZATION = False  # Disabled by default: ~30% speedup (enable only for long shaky videos)
CROP_TO_ATHLETE = False  # Crop to pose bbox (experimental)

# Fast processing mode: prioritize speed over minor quality (keeps pose/eval accuracy)
# When True: no stabilization, frame_skip=2 for all videos
# Enable via: FAST_PROCESSING=1 python app.py
FAST_PROCESSING_MODE = os.environ.get("FAST_PROCESSING", "").lower() in ("1", "true", "yes")


def _int_env(name: str, default: int, allow_zero: bool = False) -> int:
    """Read int from env. If allow_zero=False, result is at least 1 (for intervals/counts)."""
    try:
        v = int(os.environ.get(name, str(default)))
        return v if allow_zero else max(1, v)
    except (ValueError, TypeError):
        return default


# YOLO every N frames: 1=every frame (slow), 10=10x faster for short clips
_obj_default = 5 if FAST_PROCESSING_MODE else 7
_live_default = 2 if FAST_PROCESSING_MODE else 1
OBJECT_DETECTION_INTERVAL = _int_env("OBJECT_DETECTION_INTERVAL", _obj_default)  # YOLO every N frames
LIVE_CALLBACK_INTERVAL = _int_env("LIVE_CALLBACK_INTERVAL", _live_default)  # on_frame every N frames

# MediaPipe pose settings
# "heavy" = more accurate body tracking (recommended), "lite" = faster, less accurate
POSE_MODEL_VARIANT = "heavy"
# Hybrid: Lite for scan + Heavy only on key frames (Landing/Jump/Strike/Throw). Faster for long videos.
USE_HYBRID_POSE = True
# Chunked parallel: split video into CHUNK_SECONDS, process with N workers (0 = sequential)
CHUNK_PARALLEL_WORKERS = 0  # 0=seq, 2-4 for multi-core (beware MediaPipe thread safety)
# Dynamic frame skip: 2-5 based on motion (high motion=more frames)
USE_DYNAMIC_FRAME_SKIP = True
# Temporal smoothing: EMA alpha for pose landmarks (0=no smooth, 0.3=moderate, 0.6=strong)
POSE_SMOOTHING_ALPHA = 0.4
# YOLO model: "yolov8n"=fast (2x faster), "yolov8s"=better accuracy
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n")
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Video processing
DEFAULT_FPS = 30
MAX_VIDEO_DURATION_SEC = 300  # 5 minutes
MAX_UPLOAD_MB = 500  # Max video file size
# Security - CORS allowed origins (add production origin when deploying)
# Override via CORS_ORIGINS env: comma-separated list, e.g. "https://app.example.com"
_cors_env = os.environ.get("CORS_ORIGINS", "").strip()
CORS_ORIGINS: List[str] = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8001", "http://127.0.0.1:8001",
        "http://localhost:8002", "http://127.0.0.1:8002",
        "http://localhost:8888", "http://127.0.0.1:8888",
        "http://localhost:34567", "http://127.0.0.1:34567",
        "http://localhost:5050", "http://127.0.0.1:5050",
        "http://localhost:9000", "http://127.0.0.1:9000",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ]
)

# Reports: References & Sources
# False = user-facing reports (PDF/CSV/JSON) do NOT include References section
# True = include References in all reports (for audit/compliance)
INCLUDE_REFERENCES_IN_USER_REPORTS = False
# Export separate _sources.json with full references (developer-only, traceability)
EXPORT_DEV_SOURCES_FILE = True

# Joint groups for scoring
JOINT_GROUPS = {
    "upper_body": ["shoulder", "elbow", "wrist", "hip"],
    "lower_body": ["knee", "ankle", "hip"],
    "core": ["hip", "shoulder"],
}


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure application logging. Returns root sport_analysis logger."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    # Structured format: timestamp | level | module | message (machine-parseable)
    log_format = os.environ.get("LOG_FORMAT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("sport_analysis")

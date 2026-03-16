"""FastAPI server for Sports Movement Analysis."""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import asyncio
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import (
    UPLOADS_DIR, OUTPUT_DIR, REPORTS_DIR, MAX_UPLOAD_MB, CORS_ORIGINS,
    FRONTEND_DIR, MAX_STREAM_FRAMES, JOB_EXPIRY_MINUTES, VERSION,
)
from backend.utils import to_json_safe, strip_arabic_fields
from backend.api.schemas import AnalyzeRequest

from backend.config import setup_logging
from backend.exceptions import VideoSourceError, SportAnalysisError
from backend.pipeline import AnalysisPipeline

setup_logging()
logger = logging.getLogger("sport_analysis.api")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Sports Movement Analysis API",
    description="""
Professional AI sports coach API.

**Features:**
- Pose estimation (MediaPipe)
- Movement recognition (kick, jump, sprint, punch, swing, throw, squat, lunge, rotation)
- Object detection (YOLO: ball, racket, bat)
- Sport inference from movement + objects
- Movement quality scoring (0-10)
- Error detection (knee valgus, hip extension, etc.)
- Corrective exercise recommendations
- Report export (CSV, PDF, JSON)
- Live frame streaming (SSE)

**Flow:** Upload → POST /api/analyze (or /start/video, /start/camera) → Poll /progress or /api/status → /report for downloads.

**Scores:** All movement and overall scores normalized to 0-10.
    """,
    version=__import__("backend").__version__,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(VideoSourceError)
@app.exception_handler(SportAnalysisError)
def handle_analysis_error(request: Request, exc: SportAnalysisError):
    """Return structured JSON error for custom exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "code": exc.code,
            "details": exc.details,
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _preload_yolo():
    """Preload YOLO model in background so first analysis is faster."""
    try:
        from backend.models.object_tracker import ObjectTracker
        _ = ObjectTracker()
        logger.info("YOLO model preloaded for faster first analysis")
    except Exception as e:
        logger.debug("YOLO preload skipped: %s", e)


# Global pipeline instance for stop
_pipeline: Optional[AnalysisPipeline] = None
_results_store: dict = {}  # job_id -> {status, result, progress, error, created_at}
_frame_streams: dict = {}  # job_id -> list of frame data


def _cleanup_expired_jobs():
    """Remove jobs older than JOB_EXPIRY_MINUTES."""
    expiry_sec = JOB_EXPIRY_MINUTES * 60
    now = time.time()
    to_remove = [
        jid for jid, data in _results_store.items()
        if now - data.get("created_at", now) > expiry_sec
    ]
    for jid in to_remove:
        _results_store.pop(jid, None)
        _frame_streams.pop(jid, None)


@app.get("/")
def root():
    """Redirect to frontend UI; API info at /docs."""
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/app/", status_code=302)
    return {"status": "ok", "service": "Sports Movement Analysis API", "docs": "/docs"}


@app.get("/health")
@app.get("/api/health")
def health():
    """Detailed health check for monitoring and load balancers."""
    running = sum(1 for d in _results_store.values() if d.get("status") == "running")
    return {
        "status": "healthy",
        "service": "Sports Movement Analysis API",
        "version": VERSION,
        "checks": {"api": "ok"},
        "metrics": {
            "active_jobs": running,
            "total_jobs": len(_results_store),
        },
    }


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def _sanitize_upload_filename(filename: str) -> str:
    """Sanitize filename: remove path components, limit length, allow only safe chars."""
    if not filename or not filename.strip():
        return "video.mp4"
    # Remove path components (path traversal prevention)
    safe = Path(filename).name.strip()
    # Allow alphanumeric, dash, underscore, dot
    safe = "".join(c for c in safe if c.isalnum() or c in "._- ")
    safe = safe[:128] or "video"
    if not safe.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
        safe = f"{safe}.mp4"
    return safe


@app.post("/api/upload")
@limiter.limit("20/minute")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """Upload a video file for analysis. Max size: 500MB."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
        raise HTTPException(400, "Invalid file type. Use mp4, avi, mov, or webm.")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    content = b""
    size = 0
    while chunk := await file.read(65536):
        content += chunk
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(400, f"File too large. Max {MAX_UPLOAD_MB}MB.")
    safe_name = _sanitize_upload_filename(file.filename)
    path = UPLOADS_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    with open(path, "wb") as f:
        f.write(content)
    logger.info("Uploaded video: %s (%d bytes)", safe_name, size)
    return {"path": str(path), "filename": safe_name}


@app.get("/api/sports")
def list_sports():
    """List all supported sports for selection."""
    from backend.analysis.sport_profiles import SPORT_PROFILES
    sports = [{"id": "auto", "name": "Auto Detect"}]
    sports += [
        {"id": k, "name": v.get("name_en", k.title())}
        for k, v in SPORT_PROFILES.items()
        if k != "unknown"
    ]
    return {"sports": sports}


@app.post("/api/analyze")
@limiter.limit("10/minute")
async def analyze_video(
    request: Request,
    background_tasks: BackgroundTasks,
    body: "AnalyzeRequest" = Body(...),
    export_csv: bool = True,
    export_pdf: bool = True,
    export_json: bool = True,
    live_overlay: bool = True,
):
    """
    Start video analysis with live overlay streaming.
    Body: { "source": "path/to/video", "sport": "football" } or { "use_camera": true, "sport": "football" }
    sport: required
    live_overlay: stream overlay frames for real-time display
    """
    global _pipeline
    _cleanup_expired_jobs()
    source = body.source
    use_camera = body.use_camera
    sport = body.sport.strip().lower()
    if not sport:
        raise HTTPException(400, "sport is required. Select sport or 'Auto Detect'.")
    from backend.analysis.sport_profiles import SPORT_PROFILES
    if sport != "auto" and sport not in SPORT_PROFILES:
        raise HTTPException(400, f"Unknown sport: {sport}. Use /api/sports for valid options.")
    video_source = "0" if use_camera else (str(source).strip() if source else "")
    if not use_camera and not source:
        raise HTTPException(400, "Provide source path (upload video) or use_camera=true")
    if not use_camera:
        src_path = Path(video_source).resolve()
        if not src_path.exists():
            raise HTTPException(400, "Video file not found")
        if not src_path.is_file():
            raise HTTPException(400, "Video path must be a file")
        uploads_resolved = UPLOADS_DIR.resolve()
        try:
            src_path.relative_to(uploads_resolved)
        except ValueError:
            raise HTTPException(400, "Video must be uploaded via /api/upload first")

    job_id = uuid.uuid4().hex
    _results_store[job_id] = {"status": "running", "result": None, "progress": 0, "created_at": time.time()}
    _frame_streams[job_id] = []

    def on_frame(frame_idx: int, total: int, sport_key: str, score: float, img_b64: str, movement: str = "",
                 errors: list = None, feedback: str = "", strengths: list = None, objects: list = None):
        pct = round(100 * frame_idx / max(1, total)) if total else 0
        frames = _frame_streams.setdefault(job_id, [])
        frames.append({
            "frame": frame_idx, "total": total, "pct": pct,
            "sport": sport_key, "score": score, "img": img_b64,
            "movement": movement, "errors": errors or [], "feedback": feedback,
            "strengths": strengths or [], "objects": objects or [],
        })
        if len(frames) > MAX_STREAM_FRAMES:
            frames[len(frames) - MAX_STREAM_FRAMES - 1]["img"] = None

    def run_analysis():
        global _pipeline, _results_store
        try:
            def on_progress(frame_num, total, msg):
                _results_store[job_id]["progress"] = {"frame": frame_num, "total": total, "msg": msg}

            _pipeline = AnalysisPipeline()
            result = _pipeline.run_analysis(
                video_source,
                sport=sport,
                export_csv=export_csv,
                export_pdf=export_pdf,
                export_json=export_json,
                skip_overlay=not live_overlay,
                on_progress=on_progress,
                on_frame=on_frame if live_overlay else None,
            )
            _results_store[job_id]["status"] = "completed"
            _results_store[job_id]["result"] = to_json_safe(strip_arabic_fields(result))
        except (OSError, ValueError, FileNotFoundError) as e:
            logger.warning("Analysis failed (expected error): %s", e)
            _results_store[job_id]["status"] = "error"
            _results_store[job_id]["error"] = str(e)
        except Exception as e:
            logger.exception("Analysis failed (unexpected): %s", e)
            _results_store[job_id]["status"] = "error"
            _results_store[job_id]["error"] = str(e)

    background_tasks.add_task(run_analysis)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/stop")
@app.post("/stop")
def stop_analysis():
    """Stop running analysis."""
    global _pipeline
    if _pipeline:
        _pipeline.stop_analysis()
    return {"status": "stop requested"}


# Spec-compliant: /start/video, /start/camera (aliases for /api/analyze)
@app.post("/start/video")
@limiter.limit("10/minute")
async def start_video_analysis(request: Request, background_tasks: BackgroundTasks, body: "AnalyzeRequest" = Body(...)):
    """Start analysis of uploaded video. Body: {source: path, sport: string}"""
    if body.use_camera:
        raise HTTPException(400, "Use /start/camera for camera analysis.")
    if not body.source:
        raise HTTPException(400, "source is required. Upload video first via /api/upload.")
    return await analyze_video(request, background_tasks, body, True, True, True, True)


@app.post("/start/camera")
@limiter.limit("10/minute")
async def start_camera_analysis(request: Request, background_tasks: BackgroundTasks, body: "AnalyzeRequest" = Body(...)):
    """Start live camera analysis. Body: {sport: string, use_camera: true}"""
    camera_body = body.model_copy(update={"use_camera": True, "source": None})
    return await analyze_video(request, background_tasks, camera_body, True, True, True, True)


@app.get("/progress/{job_id}")
@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    """Return live analysis progress (frame, total, percentage, status)."""
    if not _valid_job_id(job_id):
        raise HTTPException(400, "Invalid job ID")
    if job_id not in _results_store:
        raise HTTPException(404, "Job not found")
    data = _results_store[job_id]
    progress = data.get("progress", {})
    if isinstance(progress, dict):
        frame = progress.get("frame", 0)
        total = progress.get("total", 1)
        pct = round(100 * frame / max(1, total)) if total else 0
    else:
        frame, total, pct = 0, 0, 0
    return {
        "job_id": job_id,
        "status": data.get("status", "unknown"),
        "frame": frame,
        "total": total,
        "progress_pct": pct,
        "message": progress.get("msg", "Processing") if isinstance(progress, dict) else "Processing",
    }


@app.get("/report/{job_id}")
@app.get("/api/report/{job_id}")
def get_report_info(job_id: str):
    """Return report metadata and download URLs for completed analysis."""
    if not _valid_job_id(job_id):
        raise HTTPException(400, "Invalid job ID")
    if job_id not in _results_store:
        raise HTTPException(404, "Job not found")
    data = _results_store[job_id]
    if data.get("status") != "completed":
        status = data.get("status", "unknown")
        raise HTTPException(
            400,
            detail=f"Analysis not complete. Current status: {status}. Poll /api/status/{job_id} for progress.",
        )
    result = data.get("result") or {}
    report_files = result.get("report_files", {}) or {}
    return {
        "job_id": job_id,
        "pdf_url": f"/api/reports/{report_files['pdf']}" if report_files.get("pdf") else None,
        "csv_url": f"/api/reports/{report_files['csv']}" if report_files.get("csv") else None,
        "json_url": f"/api/reports/{report_files['json']}" if report_files.get("json") else None,
        "video_url": f"/api/output/{result['output_filename']}" if result.get("output_filename") else None,
        "sport": result.get("sport_name_en") or result.get("sport"),
        "overall_score": result.get("overall_score"),
        "movements_analyzed": result.get("movements_analyzed", []),
    }


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """Get analysis job status and result."""
    if not _valid_job_id(job_id):
        raise HTTPException(400, "Invalid job ID")
    if job_id not in _results_store:
        raise HTTPException(404, "Job not found")
    data = _results_store[job_id]
    return to_json_safe(data)


@app.get("/api/stream/{job_id}")
async def stream_frames(job_id: str):
    """Server-Sent Events: stream overlay frames for live display."""
    if not _valid_job_id(job_id):
        raise HTTPException(400, "Invalid job ID")
    if job_id not in _results_store and job_id not in _frame_streams:
        raise HTTPException(404, "Job not found")

    async def event_gen():
        seen = 0
        while True:
            store = _results_store.get(job_id)
            frames = _frame_streams.get(job_id, [])
            # Send new frames
            while seen < len(frames):
                item = frames[seen]
                seen += 1
                yield f"data: {json.dumps(item)}\n\n"
            # Check if done
            if store and store.get("status") in ("completed", "error"):
                payload = to_json_safe(store)
                yield f"data: {json.dumps({'event': 'done', 'result': payload})}\n\n"
                return
            await asyncio.sleep(0.05)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _valid_job_id(job_id: str) -> bool:
    """Validate job_id format (32 hex chars)."""
    return bool(job_id) and len(job_id) <= 64 and all(c in "0123456789abcdef" for c in job_id.lower())


def _safe_path(base: Path, filename: str) -> Path:
    """Resolve path ensuring it stays under base (no path traversal)."""
    base = base.resolve()
    path = (base / filename).resolve()
    if not path.exists():
        raise HTTPException(404, "File not found")
    try:
        path.relative_to(base)
    except ValueError:
        raise HTTPException(404, "File not found")
    return path


@app.get("/api/reports/{filepath:path}")
def download_report(filepath: str):
    """Download a report file (CSV, PDF, JSON). Supports subdirs: Football/report_xxx.pdf."""
    if ".." in filepath or "\\" in filepath:
        raise HTTPException(400, "Invalid file path")
    # Normalize forward slashes for cross-platform
    normalized = filepath.replace("\\", "/")
    path = _safe_path(REPORTS_DIR, normalized)
    return FileResponse(path, filename=path.name)


@app.get("/api/output/{filename}")
def download_output(filename: str):
    """Download output video with overlay."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    path = _safe_path(OUTPUT_DIR, filename)
    return FileResponse(path, filename=filename)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server. Preloads YOLO in background for faster first analysis."""
    import threading
    import uvicorn
    preload = threading.Thread(target=_preload_yolo, daemon=True)
    preload.start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

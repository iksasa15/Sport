"""
Video processing pipeline with sport-specific movement analysis.

Processes video frame-by-frame:
- Pose estimation (MediaPipe)
- Movement recognition (generic -> sport-specific mapping)
- Sport-specific evaluation (0-10 scale per movement)
- Per-movement score aggregation for professional report
"""

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import cv2
import numpy as np

from backend.analysis.evaluator import MovementEvaluator, MovementEvaluation, RecommendationEngine, INJURY_RISK_MAP
from backend.analysis.biomechanics import get_coaching_for_error
from backend.analysis.features import extract_frame_features, FrameFeatures
from backend.sources import get_sources_for_sport
from backend.utils import safe_get
from backend.analysis.sport_profiles import (
    get_sport_profile,
    get_movement_by_generic,
    get_technical_movements,
    get_coaching_feedback,
    get_coaching_feedback_with_equipment,
    get_equipment_validation_warnings,
    get_relevant_object_labels,
)
from backend.config import (
    OUTPUT_DIR,
    FRAME_SKIP,
    POSE_SMOOTHING_ALPHA,
    VIDEO_TARGET_HEIGHT,
    FRAME_SKIP_FAST,
    USE_VIDEO_STABILIZATION,
    USE_HYBRID_POSE,
    USE_DYNAMIC_FRAME_SKIP,
    FAST_PROCESSING_MODE,
    OBJECT_DETECTION_INTERVAL,
    LIVE_CALLBACK_INTERVAL,
)
from backend.models.movement_recognizer import MovementRecognizer
from backend.video.preprocessor import VideoPreprocessor, PreprocessOptions
from backend.video.key_frame_detector import KeyFrameDetector
from backend.video.landmark_smoother import LandmarkSmoother
from backend.models.object_tracker import ObjectTracker
from backend.models.pose_estimator import PoseEstimator
from backend.models.hybrid_pose import HybridPoseEstimator
from backend.models.sport_inferencer import infer_sport
from backend.video.overlay import VideoOverlay

logger = logging.getLogger("sport_analysis.processor")


def _create_video_writer(path: str, fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    """Create VideoWriter. Prefer mp4v so writer actually opens (avc1 often fails); overlay is per-frame live."""
    if width <= 0 or height <= 0 or fps <= 0:
        logger.warning("Invalid video writer params: %dx%d @ %.1f fps", width, height, fps)
        return None
    # Try mp4v first so we always get a working writer (otherwise output can be 0 sec)
    for fourcc_name in ("mp4v", "avc1", "H264", "h264", "X264"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            w = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if w.isOpened():
                logger.info("Video writer opened: %s, fourcc=%s, %.0fx%.0f @ %.1f fps", path, fourcc_name, width, height, fps)
                return w
            w.release()
        except Exception as e:
            logger.debug("Codec %s failed: %s", fourcc_name, e)
    logger.error("No video codec opened for %s", path)
    return None


def _valid_output_path(path: Optional[str]) -> Optional[str]:
    """Return path if it exists and has size > 0, else None (so we don't return broken video URL)."""
    if not path:
        return None
    try:
        p = Path(path)
        if p.is_file() and p.stat().st_size > 0:
            return path
    except OSError:
        pass
    return None


def _reencode_to_h264_for_ios(path: str) -> bool:
    """Re-encode overlay video to H.264 so it plays and saves correctly on iOS (Photos app, device)."""
    try:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg not found; output may not play when saved on iOS. Install ffmpeg for H.264.")
            return False
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            return False
        fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="overlay_h264_")
        try:
            os.close(fd)
            # -profile:v main + level 4.0 = broad compatibility (iOS, Photos); -movflags +faststart = plays after save
            rc = subprocess.run(
                [
                    ffmpeg, "-y", "-i", path,
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-profile:v", "main", "-level", "4.0",
                    "-movflags", "+faststart",
                    tmp,
                ],
                capture_output=True,
                timeout=300,
            )
            if rc.returncode == 0 and Path(tmp).is_file():
                shutil.move(tmp, path)
                logger.info("Re-encoded overlay to H.264 for iOS: %s", path)
                return True
        finally:
            if Path(tmp).exists():
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass
    except Exception as e:
        logger.debug("ffmpeg re-encode skipped: %s", e)
    return False


def _score_100_to_10(score_100: float) -> float:
    """Convert 0-100 score to 0-10 scale."""
    return round(min(10.0, max(0.0, score_100 / 10.0)), 1)


class VideoProcessor:
    """Process video through full analysis pipeline with sport-specific scoring."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        on_stop: Optional[Callable] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_frame: Optional[Callable[[int, int, str, float, str], None]] = None,
    ):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_stop = on_stop
        self.on_progress = on_progress
        self.on_frame = on_frame
        self._stop_requested = False

    def stop(self):
        """Request stop of processing."""
        self._stop_requested = True
        if self.on_stop:
            self.on_stop()

    def process_video(
        self,
        source: str,
        sport: str,
        output_path: Optional[str] = None,
        skip_overlay: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: read video, pose, movement recognition, evaluate per sport.
        sport: user-selected sport (required)
        Returns summary with per-movement scores (0-10), errors, coaching feedback.
        """
        self._stop_requested = False
        sport = (sport or "unknown").lower().strip()
        use_auto_sport = sport == "auto"
        t_start = time.perf_counter()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            from backend.exceptions import VideoSourceError
            raise VideoSourceError(f"Cannot open video source: {source}", source=str(source))

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        overlay_name = f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = (output_path or str(self.output_dir / overlay_name)) if not skip_overlay else None
        # Write to temp first so we don't leave a 0-byte or corrupt file at out_path if writer fails
        overlay_write_path: Optional[str] = None  # temp path while writing
        writer = None  # Lazy init when we know preprocessed frame size

        pose_estimator = (
            HybridPoseEstimator() if USE_HYBRID_POSE else PoseEstimator()
        )
        landmark_smoother = LandmarkSmoother(alpha=POSE_SMOOTHING_ALPHA)
        movement_recognizer = MovementRecognizer()
        object_tracker = ObjectTracker()
        overlay = VideoOverlay(pose_estimator)
        rec_engine = RecommendationEngine()
        evaluator = MovementEvaluator(fps=fps)

        # Video preprocessing: optimize for speed (short videos get fast mode regardless of resolution)
        tf = total_frames or 0
        # Fast mode: <90 sec OR high-res OR FAST_PROCESSING env → motion-based skip, no stabilization
        use_fast_mode = FAST_PROCESSING_MODE or tf <= 2700 or tf > 900 or h > (VIDEO_TARGET_HEIGHT or 0)
        preprocessor = VideoPreprocessor(PreprocessOptions(
            target_height=VIDEO_TARGET_HEIGHT if (h > (VIDEO_TARGET_HEIGHT or 0)) else 0,
            frame_skip=1,  # We handle skip via dynamic_skip
            enable_stabilization=bool(USE_VIDEO_STABILIZATION) and not FAST_PROCESSING_MODE,
            enable_crop=False,
        ))
        key_frame_detector = KeyFrameDetector()
        prev_features_for_kf: Optional[FrameFeatures] = None
        obj_detection_interval = max(1, int(OBJECT_DETECTION_INTERVAL or 1))
        live_callback_interval = max(1, int(LIVE_CALLBACK_INTERVAL or 1))
        if tf <= 900:
            obj_detection_interval = max(obj_detection_interval, 10)  # YOLO every 10 frames for short clips
            live_callback_interval = max(live_callback_interval, 3)   # Fewer SSE callbacks for short clips
        last_objs: List[Any] = []

        # Per-movement score aggregation: movement_id -> list of frame scores (0-100)
        movement_scores: Dict[str, List[float]] = defaultdict(list)
        all_errors: List[str] = []
        all_evaluations: List[Dict] = []
        all_objects: List[Dict] = []

        frame_idx = 0
        movement = "unknown"
        inferred_sport = "unknown"
        inferred_sport_conf = 0.0
        sport_votes: Dict[str, int] = {}  # Temporal smoothing: sport votes over recent frames
        frame_errors_last_30: List[str] = []  # For error logging every 30 frames

        # Frame skip: higher = faster (short videos get aggressive skip)
        if FAST_PROCESSING_MODE:
            dynamic_skip = max(FRAME_SKIP_FAST, 3)  # Always use frame skip in fast mode
        elif tf > 3600:
            dynamic_skip = 5  # >2 min
        elif tf > 1800:
            dynamic_skip = 4  # >1 min
        elif tf > 900:
            dynamic_skip = max(FRAME_SKIP_FAST, 3)  # >30 sec: use fast skip
        else:
            dynamic_skip = FRAME_SKIP  # Short clips: every 3rd frame by default
        try:
            while cap.isOpened() and not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                if dynamic_skip > 1 and frame_idx > 0 and frame_idx % dynamic_skip != 0:
                    frame_idx += 1
                    if frame_idx % 60 == 0 and self.on_progress:
                        try:
                            self.on_progress(frame_idx, total_frames or frame_idx, "Processing")
                        except (TypeError, ValueError, RuntimeError) as e:
                            logger.debug("Progress callback failed: %s", e)
                    continue
                t0 = time.perf_counter()
                # Preprocess: resize to 720p, stabilize (for long/high-res videos)
                frame = preprocessor.process(frame, frame_idx)
                results, landmarks = None, {}

                try:
                    results, landmarks_raw = pose_estimator.process_frame(frame)
                    landmarks = landmark_smoother.smooth(landmarks_raw)
                except Exception as e:
                    frame_errors_last_30.append(f"pose:{str(e)[:50]}")
                    logger.warning("Pose estimation failed frame %d: %s", frame_idx, e)
                    landmarks = {}

                try:
                    movement, _ = movement_recognizer.recognize(landmarks)
                except Exception as e:
                    logger.debug("Movement recognition failed: %s", e)

                objs = []
                if frame_idx % obj_detection_interval == 0:
                    try:
                        objs = object_tracker.detect_objects(frame, frame_idx)  # Uses preprocessed frame
                        last_objs = objs
                    except Exception as e:
                        logger.debug("Object detection failed frame %d: %s", frame_idx, e)
                        last_objs = objs if objs else last_objs
                else:
                    objs = last_objs
                obj_labels = [str(getattr(o, "label", "")) for o in objs]
                # Auto sport inference (with hysteresis: require confidence >= 0.5)
                inf_sport, inf_conf = infer_sport(movement, obj_labels)
                if inf_conf >= 0.5:
                    sport_votes[inf_sport] = sport_votes.get(inf_sport, 0) + 1
                    if sport_votes.get(inf_sport, 0) >= 3:  # Sticky: 3+ votes
                        inferred_sport = inf_sport
                        inferred_sport_conf = inf_conf
                # Effective sport and object filter (before use)
                effective_sport = sport
                if use_auto_sport:
                    effective_sport = inferred_sport if inferred_sport_conf >= 0.5 else "unknown"
                relevant_labels = get_relevant_object_labels(effective_sport)
                objs_filtered = [o for o in objs if str(getattr(o, "label", "")).lower() in {l.lower() for l in relevant_labels}] if relevant_labels else objs
                for o in objs_filtered:
                    bbox = getattr(o, "bbox", [])
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        try:
                            bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                        except (TypeError, ValueError):
                            bbox = []
                    else:
                        bbox = []
                    all_objects.append({
                        "label": str(getattr(o, "label", "")),
                        "frame_idx": frame_idx,
                        "confidence": float(getattr(o, "confidence", 0)),
                        "bbox": bbox,
                    })

                # Key-frame detection (Landing, Strike, Jump, Throw) for high-detail analysis
                key_frame_event = None
                if landmarks:
                    features = extract_frame_features(landmarks, prev_features_for_kf, fps)
                    prev_features_for_kf = features
                    key_frame_event = key_frame_detector.detect(frame_idx, features)

                    # Dynamic frame skip: high motion -> analyze more frames (skip 2), low motion -> skip 4-5
                    if use_fast_mode and USE_DYNAMIC_FRAME_SKIP and features.angular_velocity:
                        motion = sum(abs(v) for v in features.angular_velocity.values())
                        dynamic_skip = 2 if motion > 50 else (3 if motion > 20 else max(FRAME_SKIP_FAST, 4))

                    # Hybrid: refine key frames with Heavy model for better accuracy
                    if key_frame_event and USE_HYBRID_POSE and hasattr(pose_estimator, "upgrade_to_heavy"):
                        try:
                            _, landmarks_heavy = pose_estimator.upgrade_to_heavy(frame)
                            if landmarks_heavy:
                                landmarks = landmark_smoother.smooth(landmarks_heavy)
                        except Exception as e:
                            logger.debug("Heavy upgrade failed frame %d: %s", frame_idx, e)

                # Map generic movement to sport-specific
                sport_movement = get_movement_by_generic(effective_sport, movement)
                if sport_movement:
                    movement_id = sport_movement.get("id", movement)
                else:
                    movement_id = movement if movement != "unknown" else "general"

                # Evaluation uses effective_sport (inferred when auto mode) + equipment for interaction metrics
                try:
                    eval_result = evaluator.evaluate_frame(
                        landmarks, frame_idx, effective_sport, movement=movement, objects=objs_filtered
                    )
                    score_100 = eval_result.overall_score
                    movement_scores[movement_id].append(score_100)
                    all_errors.extend(eval_result.errors or [])
                    joint_dicts = [
                        {
                            "name": str(j.name),
                            "score": float(j.score),
                            "is_correct": bool(j.is_correct),
                            "feedback": str(j.feedback),
                        }
                        for j in (eval_result.joint_scores or [])
                    ]
                    all_evaluations.append({
                        "frame_idx": frame_idx,
                        "overall_score": score_100,
                        "movement": movement_id,
                        "errors": list(eval_result.errors or []),
                        "joint_scores": joint_dicts,
                        "skill_metrics": getattr(eval_result, "skill_metrics", None) or {},
                        "injury_risk_warnings": getattr(eval_result, "injury_risk_warnings", None) or [],
                        "injury_risk_score": getattr(eval_result, "injury_risk_score", 0),
                        "features_for_training": getattr(eval_result, "features_for_training", None) or {},
                        "key_frame_event": key_frame_event.event_type if key_frame_event else None,
                    })
                    eval_result_for_overlay = eval_result
                except Exception as e:
                    logger.warning("Evaluation failed frame %d: %s", frame_idx, e)
                    all_evaluations.append({
                        "frame_idx": frame_idx,
                        "overall_score": 0.0,
                        "movement": movement_id,
                        "errors": ["Evaluation failed"],
                        "joint_scores": [],
                        "skill_metrics": {},
                        "key_frame_event": None,
                    })
                    eval_result_for_overlay = MovementEvaluation(
                        frame_idx=frame_idx, overall_score=0.0, is_correct=False,
                        errors=["Evaluation failed"], sport=sport, movement=movement
                    )
                    frame_errors_last_30.append("evaluation:failed")

                rec_names = []
                try:
                    recs = rec_engine.get_recommendations(
                        eval_result_for_overlay.errors or [],
                        eval_result_for_overlay.joint_scores or [],
                        sport,
                    )
                    rec_names = [r.name for r in recs]
                except (AttributeError, TypeError, KeyError) as e:
                    logger.debug("Recommendations failed frame %d: %s", frame_idx, e)

                frame_out = frame
                if not skip_overlay and out_path:
                    if writer is None:
                        # Write to temp file first; move to out_path only when we have a valid file (avoids 0-sec/corrupt)
                        if overlay_write_path is None:
                            fd, overlay_write_path = tempfile.mkstemp(suffix=".mp4", prefix="overlay_", dir=str(self.output_dir))
                            os.close(fd)
                        # Output fps = input fps / skip so duration matches (avoid 0-sec or too-short video)
                        out_fps = max(1.0, fps / dynamic_skip)
                        writer = _create_video_writer(
                            overlay_write_path, out_fps, frame.shape[1], frame.shape[0]
                        )
                        if writer is None:
                            writer = cv2.VideoWriter(
                                overlay_write_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                out_fps,
                                (frame.shape[1], frame.shape[0]),
                            )
                            if not writer.isOpened():
                                logger.error("Fallback mp4v writer failed to open; overlay video will be empty")
                                writer = None
                    if writer is not None and writer.isOpened():
                        try:
                            frame_out = overlay.draw_overlay(
                                frame,
                                sport=effective_sport,
                                movement=movement_id,
                                score=eval_result_for_overlay.overall_score,
                                errors=eval_result_for_overlay.errors or [],
                                recommendations=rec_names,
                                objects=objs_filtered,
                                frame_idx=frame_idx,
                                processing_time_ms=(time.perf_counter() - t0) * 1000,
                                draw_skeleton=True,
                                results=results,
                                total_frames=total_frames or 0,
                                joint_risk_levels=getattr(eval_result_for_overlay, "joint_risk_levels", None) or {},
                                injury_risk_score=getattr(eval_result_for_overlay, "injury_risk_score", None),
                            )
                            to_write = np.ascontiguousarray(frame_out)
                            writer.write(to_write)
                        except (cv2.error, TypeError, ValueError) as e:
                            logger.warning("Overlay draw failed frame %d: %s", frame_idx, e)
                            writer.write(np.ascontiguousarray(frame))

                if self.on_frame and not skip_overlay and frame_idx % live_callback_interval == 0:
                    try:
                        _, jpeg = cv2.imencode(".jpg", frame_out)
                        if jpeg is not None:
                            errs = eval_result_for_overlay.errors or []
                            strengths = [
                                f"{safe_get(j, 'name', '').replace('_', ' ').title()} alignment"
                                for j in (eval_result_for_overlay.joint_scores or [])
                                if safe_get(j, "score", 0) >= 80
                            ]
                            obj_labels = [getattr(o, "label", "") for o in objs_filtered][:5]
                            fb = ""
                            if errs:
                                obj_labels_str = [getattr(o, "label", "") for o in objs_filtered]
                                fb = get_coaching_feedback_with_equipment(effective_sport, errs[0], obj_labels_str)
                            elif eval_result_for_overlay.overall_score < 70:
                                fb = "Focus on form and alignment"
                            else:
                                fb = "Good form"
                            self.on_frame(
                                frame_idx,
                                total_frames or (frame_idx + 1),
                                effective_sport,
                                _score_100_to_10(eval_result_for_overlay.overall_score),
                                base64.b64encode(jpeg.tobytes()).decode("ascii"),
                                movement_id,
                                errors=errs,
                                feedback=fb,
                                strengths=strengths,
                                objects=obj_labels,
                            )
                    except (TypeError, ValueError, RuntimeError, OSError) as e:
                        logger.debug("Frame callback failed frame %d: %s", frame_idx, e)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    logger.info("Processed %d frames", frame_idx)
                    if frame_errors_last_30:
                        logger.warning(
                            "Frame errors in last 30 frames (at %d): %s",
                            frame_idx,
                            ", ".join(frame_errors_last_30[-10:]),
                        )
                        frame_errors_last_30.clear()
                    if self.on_progress:
                        try:
                            self.on_progress(frame_idx, total_frames or frame_idx, "Processing")
                        except (TypeError, ValueError, RuntimeError) as e:
                            logger.debug("Progress callback failed: %s", e)

        finally:
            cap.release()
            if writer:
                writer.release()
            pose_estimator.close()
            # Move temp overlay to final path only if we have a valid file (avoids 0-sec or unplayable)
            if not skip_overlay and out_path and overlay_write_path and Path(overlay_write_path).is_file():
                size = Path(overlay_write_path).stat().st_size
                if size > 0:
                    try:
                        shutil.move(overlay_write_path, out_path)
                        _reencode_to_h264_for_ios(out_path)
                    except (OSError, IOError) as e:
                        logger.warning("Failed to move overlay to %s: %s", out_path, e)
                        try:
                            Path(overlay_write_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                        out_path = None
                else:
                    logger.warning("Overlay video was 0 bytes; not saving")
                    try:
                        Path(overlay_write_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                    out_path = None
            elif overlay_write_path and Path(overlay_write_path).exists():
                try:
                    Path(overlay_write_path).unlink(missing_ok=True)
                except OSError:
                    pass

        # Final sport for report: inferred when auto, else user-selected
        final_sport = inferred_sport if (use_auto_sport and inferred_sport_conf >= 0.5) else sport
        if final_sport == "auto":
            final_sport = inferred_sport if inferred_sport_conf >= 0.5 else "unknown"

        # Build per-movement summary (0-10 scale)
        movements_analyzed: List[Dict] = []
        for mov in get_technical_movements(final_sport):
            mov_id = mov.get("id", "")
            scores = movement_scores.get(mov_id, [])
            if not scores and mov.get("generic"):
                # Check generic key
                for k, v in movement_scores.items():
                    if k == mov.get("generic"):
                        scores = v
                        break
            if scores:
                avg_100 = float(np.mean(scores))
                score_10 = _score_100_to_10(avg_100)
                strengths_mov = ["Good form", "Solid technique"] if score_10 >= 7 else []
                weaknesses_mov = [] if score_10 >= 7 else ["Needs technique refinement"]
                movements_analyzed.append({
                    "id": mov_id,
                    "name_en": mov.get("name_en", mov_id),
                    "name": mov.get("name_en", mov_id),
                    "score": score_10,
                    "frames_count": len(scores),
                    "feedback": "Good form - continue practice" if score_10 >= 7 else "Focus on improvement - see recommendations",
                    "strengths": strengths_mov,
                    "weaknesses": weaknesses_mov,
                    "improvement_note": "" if score_10 >= 7 else "Practice slow, controlled repetitions with focus on joint alignment.",
                })
        # Add any detected movements not in technical list
        for mov_id, scores in movement_scores.items():
            if not any(m["id"] == mov_id for m in movements_analyzed):
                avg_100 = float(np.mean(scores))
                score_10 = _score_100_to_10(avg_100)
                movements_analyzed.append({
                    "id": mov_id,
                    "name_en": mov_id.replace("_", " ").title(),
                    "name": mov_id.replace("_", " ").title(),
                    "score": score_10,
                    "frames_count": len(scores),
                    "feedback": "Detected movement" if score_10 >= 5 else "Needs improvement",
                    "strengths": [],
                    "weaknesses": [],
                    "improvement_note": "",
                })

        unique_errors = list(dict.fromkeys(all_errors))[:10]
        # Aggregate injury-risk warnings from evaluations (poor mechanics -> injury risk)
        all_injury_warnings: List[str] = []
        for e in all_evaluations:
            for w in (e.get("injury_risk_warnings") or []):
                if w and w not in all_injury_warnings:
                    all_injury_warnings.append(w)
        unique_injury_warnings = all_injury_warnings[:5]
        object_labels_seen = list(dict.fromkeys(o.get("label", "") for o in all_objects if o.get("label")))
        equipment_warnings = get_equipment_validation_warnings(final_sport, object_labels_seen)
        unique_errors = equipment_warnings + unique_errors
        joint_scores_agg = all_evaluations[-1].get("joint_scores", []) if all_evaluations else []
        avg_score = (
            sum(e["overall_score"] for e in all_evaluations) / len(all_evaluations)
            if all_evaluations else 0
        )
        strengths_observed = [
            safe_get(j, "name", "").replace("_", " ").title() + " alignment"
            for j in (joint_scores_agg or [])
            if safe_get(j, "score", 0) >= 80
        ]
        if not strengths_observed:
            strengths_observed = ["Overall form" if avg_score >= 70 else "Stable posture"]
        final_recs = rec_engine.get_recommendations(
            unique_errors, joint_scores_agg, final_sport
        )
        development_plan = rec_engine.get_development_plan(final_sport, avg_score, unique_errors)

        # Coaching feedback for each error (equipment-aware when objects detected)
        # Deduplicate by feedback text — same tip for knee valgus (left/right) shown once
        # Sources verified: ERROR_OFFICIAL_SOURCES in backend.sources (internal audit only)
        coaching_feedback: List[Dict] = []
        seen_feedback: set = set()
        for err in unique_errors:
            if "expected" in err.lower() and "not detected" in err.lower():
                if err not in seen_feedback:
                    seen_feedback.add(err)
                    coaching_feedback.append({"error": err, "feedback": err})
                continue
            err_lower = err.lower()
            tip = None
            for key in get_sport_profile(final_sport).get("coaching_tips", {}):
                if key in err_lower or err_lower in key:
                    tip = get_coaching_feedback_with_equipment(final_sport, key, object_labels_seen)
                    break
            if not tip:
                tip = get_coaching_feedback_with_equipment(final_sport, err, object_labels_seen)
            # Deduplicate: same feedback for multiple errors (e.g. left/right knee) → show once
            tip_key = (tip or "").strip()
            if tip_key and tip_key not in seen_feedback:
                seen_feedback.add(tip_key)
                coaching_feedback.append({"error": err, "feedback": tip})

        sport_profile = get_sport_profile(final_sport)

        # Collect injury-risk warnings from frame evaluations (poor mechanics detection)
        injury_warnings_seen: set = set()
        for ev in all_evaluations:
            for w in ev.get("injury_risk_warnings", []) or []:
                injury_warnings_seen.add(w)
        injury_risk_warnings = list(injury_warnings_seen)[:5]

        # Build injury_risk_with_corrections: each warning + explanation + how to fix
        injury_risk_with_corrections: List[Dict[str, Any]] = []
        for err_key, warning in INJURY_RISK_MAP.items():
            if warning in injury_warnings_seen:
                advice, injuries = get_coaching_for_error(err_key)
                injury_risk_with_corrections.append({
                    "warning": warning,
                    "correction": advice,
                    "possible_injuries": injuries[:3],
                })

        # Inferred sport from movement+objects (for UI hint)
        final_inferred = inferred_sport if inferred_sport_conf >= 0.5 else None
        # Aggregate injury risk score and confidence (from last evaluation)
        last_eval = all_evaluations[-1] if all_evaluations else {}
        injury_risk_score_agg = last_eval.get("injury_risk_score") or 0.0
        confidence_agg = last_eval.get("confidence") or 1.0
        possible_injuries_agg = list(dict.fromkeys(
            i for e in all_evaluations for i in (e.get("possible_injuries") or [])
        ))[:5]
        if not injury_risk_score_agg and all_evaluations:
            scores = [e.get("injury_risk_score") for e in all_evaluations if e.get("injury_risk_score")]
            injury_risk_score_agg = float(np.mean(scores)) if scores else 0.0

        processing_time_sec = round(time.perf_counter() - t_start, 2)
        logger.info(
            "Analysis complete: sport=%s frames=%d time=%.2fs fps=%.1f",
            final_sport, frame_idx, processing_time_sec,
            frame_idx / processing_time_sec if processing_time_sec > 0 else 0,
        )
        summary = {
            "sport": final_sport,
            "sport_name": sport_profile.get("name_en", sport_profile.get("name", final_sport)),
            "sport_name_en": sport_profile.get("name_en", final_sport),
            "sport_was_auto": use_auto_sport,
            "injury_risk_warnings": injury_risk_warnings,
            "injury_risk_with_corrections": injury_risk_with_corrections,
            "injury_risk_score": round(injury_risk_score_agg, 1),
            "possible_injuries": possible_injuries_agg,
            "confidence": round(confidence_agg, 2),
            "inferred_sport": final_inferred,
            "inferred_sport_confidence": round(inferred_sport_conf, 2) if final_inferred else None,
            "movements_analyzed": movements_analyzed,
            "overall_score": _score_100_to_10(avg_score),
            "overall_score_100": round(avg_score, 1),
            "total_frames": frame_idx,
            "video_width": w,
            "video_height": h,
            "video_fps": fps,
            "errors": unique_errors,
            "strengths": strengths_observed[:8],
            "coaching_feedback": coaching_feedback,
            "recommendations": [
                {
                    "name": str(r.name),
                    "description": str(r.description),
                    "target_joint": str(r.target_joint),
                    "reps_sets": str(r.reps_sets),
                    "sport_focus": str(getattr(r, "sport_focus", "")),
                }
                for r in final_recs
            ],
            "joint_scores": [
                {**js, "score": round(min(10.0, max(0.0, safe_get(js, "score", 0) / 10.0)), 1)}
                for js in (joint_scores_agg or [])
            ],
            "object_tracking": all_objects[:50],
            "development_plan": development_plan,
            "frame_evaluations": all_evaluations,
            "output_video_path": (valid_out := _valid_output_path(out_path)),
            "output_filename": Path(valid_out).name if valid_out else None,
            "sources": get_sources_for_sport(final_sport),
            "processing_time_sec": processing_time_sec,
        }
        return summary

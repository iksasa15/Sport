"""Pose estimation using MediaPipe Tasks API (PoseLandmarker).

Professional landmark extraction for sports movement analysis.
Exports: PoseEstimator, LANDMARK_NAMES, POSE_CONNECTIONS.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import POSE_MODEL_VARIANT

logger = logging.getLogger("sport_analysis.pose")

# MediaPipe Tasks API (PoseLandmarker). Requires mediapipe>=0.10.14.
try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.core import image as mp_image
    HAS_TASKS = True
except ImportError as e:
    logger.debug("MediaPipe tasks import failed: %s", e)
    if "runtime_version" in str(e) or "protobuf" in str(e).lower():
        logger.warning(
            "MediaPipe failed due to protobuf. Upgrade: pip install 'protobuf>=6.31.1,<7'"
        )
    HAS_TASKS = False

# Pose landmark names (33 landmarks in MediaPipe)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Connection pairs for drawing skeleton (MediaPipe pose connections)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # face
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),  # mouth
    (11, 12),  # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # right arm
    (11, 23), (12, 24),  # torso
    (23, 24),  # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
]

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _ensure_model(variant: str = "heavy") -> Path:
    """Download pose landmarker model if not present. variant: 'heavy' (accurate) or 'lite' (fast)."""
    import tempfile
    import urllib.request
    variant = (variant or "heavy").lower()
    if variant not in MODEL_URLS:
        variant = "heavy"
    url = MODEL_URLS[variant]
    models_dir = Path(tempfile.gettempdir()) / "sport_analysis_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"pose_landmarker_{variant}.task"
    if not model_path.exists():
        logger.info("Downloading pose landmarker model (%s)...", variant)
        urllib.request.urlretrieve(url, model_path)
    return model_path


class PoseEstimator:
    """MediaPipe PoseLandmarker for sports movement analysis. Uses Heavy model for better accuracy."""

    def __init__(self, model_variant: Optional[str] = None):
        if not HAS_TASKS:
            raise RuntimeError(
                "MediaPipe Tasks API not available. Install or upgrade: pip install --upgrade 'mediapipe>=0.10.14'"
            )
        variant = model_variant or POSE_MODEL_VARIANT
        model_path = _ensure_model(variant)
        base_options = BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[object], Dict[str, Tuple[float, float, float]]]:
        """
        Process a single frame and return landmarks.
        timestamp_ms: optional; if provided, use it and do not advance internal timestamp (for hybrid Lite+Heavy same-frame).
        Returns (result, landmarks_dict) where landmarks_dict maps name -> (x, y, z).
        """
        ts = timestamp_ms if timestamp_ms is not None else self._frame_timestamp_ms
        if timestamp_ms is None:
            self._frame_timestamp_ms += 33  # ~30fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, ts)

        landmarks = {}
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = result.pose_landmarks[0]
            for i, lm in enumerate(lm_list):
                if i < len(LANDMARK_NAMES):
                    name = LANDMARK_NAMES[i]
                    landmarks[name] = (lm.x, lm.y, lm.z)

        return result, landmarks

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """Draw skeleton and landmarks - professional styling (thick lines, clear joints)."""
        if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            line_color = (0, 220, 100)  # Green (BGR)
            joint_color = (0, 255, 200)  # Cyan-green
            if draw_connections:
                for i, j in POSE_CONNECTIONS:
                    if i < len(pts) and j < len(pts):
                        cv2.line(frame, pts[i], pts[j], line_color, 3)
            for p in pts:
                cv2.circle(frame, p, 5, joint_color, -1)
                cv2.circle(frame, p, 5, (255, 255, 255), 1)
        return frame

    def close(self):
        """Release resources."""
        if hasattr(self.landmarker, "close"):
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

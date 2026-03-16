"""
Sport-specific movement evaluation.

Evaluates each movement against ideal biomechanical standards (no user-selected skill level).
All athletes are evaluated against the same sport-specific ideal angles and criteria.

- Ideal joint angles (movement-specific, then generic fallback)
- Sport-specific error detection (knee valgus, hip extension, etc.)
- Weighted scoring for key joints of that sport
- Coaching feedback from profile.coaching_tips
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.analysis.sport_profiles import (
    get_sport_profile,
    get_ideal_angle_fallback,
    get_sport_exercises,
    get_development_plan,
)
from backend.analysis.exercises import GENERIC_EXERCISES_RAW
from backend.sources import get_sources_for_sport, format_source_short
from backend.analysis.features import extract_frame_features, FrameFeatures
from backend.analysis.biomechanics import (
    check_angle_safety,
    compute_injury_risk_score,
    get_coaching_for_error,
    ERROR_COACHING_MAP,
)
from backend.utils import safe_get

logger = logging.getLogger("sport_analysis.evaluator")


@dataclass
class JointScore:
    """Score for a single joint or joint group."""

    name: str
    score: float  # 0-100
    is_correct: bool
    feedback: str
    risk_level: str = "safe"  # safe | moderate | high (for overlay coloring)


# Error -> injury-risk warning mapping (biomechanics-based)
INJURY_RISK_MAP: Dict[str, str] = {
    "knee_valgus": "Potential ACL/MCL stress - knee collapsing inward during load",
    "poor_hip_extension": "Lower back compensation risk - reduced hip drive",
    "ankle_instability": "Ankle sprain risk - improve single-leg balance",
    "unstable_landing": "Knee/ankle injury risk - practice soft landings with bent knees",
    "shoulder_imbalance": "Rotator cuff strain risk - improve shoulder symmetry",
    "limited_rotation": "Spine compensation risk - drive from hips and core",
    "elbow_alignment": "Elbow strain - keep elbow in line with wrist",
    "elbow_drop": "Body exposure - maintain guard position",
    "core_instability": "Lower back stress - engage core during movement",
}


@dataclass
class MovementEvaluation:
    """Evaluation result for a frame or movement."""

    frame_idx: int
    overall_score: float  # 0-100
    is_correct: bool
    joint_scores: List[JointScore] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    sport: str = "unknown"
    movement: str = "unknown"
    # Skill metrics: speed, power, balance, accuracy, timing (0-10 scale)
    skill_metrics: Dict[str, float] = field(default_factory=dict)
    # Injury-risk warnings when poor mechanics detected
    injury_risk_warnings: List[str] = field(default_factory=list)
    # Injury Risk Score 0-100 (100 = highest risk)
    injury_risk_score: float = 0.0
    # Joint name -> risk level for overlay (red/yellow): "safe" | "moderate" | "high"
    joint_risk_levels: Dict[str, str] = field(default_factory=dict)
    # Possible injuries from detected errors
    possible_injuries: List[str] = field(default_factory=list)
    # Confidence 0-1 for movement detection
    confidence: float = 1.0
    # Compact features for training data (angles, symmetry, stability)
    features_for_training: Optional[Dict[str, Any]] = None


def _get_ideal(sport: str, joint_type: str, movement: str = "") -> Tuple[float, float]:
    """
    Get ideal joint angle range (degrees) for sport + joint + movement.
    Tries: movement-specific key, generic joint key, sport fallback, defaults.
    Returns:
        Tuple of (ideal_min, ideal_max) degrees for scoring.
    """
    sport = sport.lower() if sport else "unknown"
    profile = get_sport_profile(sport)
    ideal_angles = profile.get("ideal_angles", {})
    movement = (movement or "").lower()

    # Try movement-specific keys first (e.g., knee_kick, elbow_swing)
    if movement:
        move_key = f"{joint_type}_{movement}"
        if move_key in ideal_angles:
            v = ideal_angles[move_key]
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                return float(v[0]), float(v[1])
        for key, val in ideal_angles.items():
            if f"{joint_type}" in key and movement in key and isinstance(val, (tuple, list)) and len(val) >= 2:
                return float(val[0]), float(val[1])
    # Generic sport profile keys
    if joint_type in ideal_angles:
        v = ideal_angles[joint_type]
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            return float(v[0]), float(v[1])
    # Fallback from sport_profiles
    fb = get_ideal_angle_fallback(sport, joint_type)
    if fb:
        return float(fb[0]), float(fb[1])
    # Defaults
    if "knee" in joint_type:
        return (150, 175)
    if "elbow" in joint_type:
        return (80, 100)
    if "shoulder" in joint_type or "hip" in joint_type:
        return (160, 185)
    return (150, 175)


class MovementEvaluator:
    """Evaluate movement quality with sport-specific criteria."""

    def __init__(self, fps: float = 30.0):
        self.history: List[Dict] = []
        self._prev_features: Optional[FrameFeatures] = None
        self.fps = fps

    def _angle_between_points(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> float:
        """Compute angle at p2 (degrees)."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2) or 1e-6
        mag2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2) or 1e-6
        cos_a = np.clip(dot / (mag1 * mag2), -1, 1)
        return float(np.degrees(np.arccos(cos_a)))

    def _get_landmark(
        self, landmarks: Dict, name: str
    ) -> Optional[Tuple[float, float]]:
        """Get (x, y) for landmark."""
        if name in landmarks:
            return landmarks[name][0], landmarks[name][1]
        return None

    def _score_angle(self, angle: float, ideal_min: float, ideal_max: float, joint_name: str) -> float:
        """Score 0-100 based on deviation from ideal range."""
        mid = (ideal_min + ideal_max) / 2
        if ideal_min <= angle <= ideal_max:
            return 100.0
        dev = min(abs(angle - ideal_min), abs(angle - ideal_max))
        return max(0, 100 - dev * 2)

    def _equipment_interaction_score(
        self,
        landmarks: Dict,
        objects: List[Any],
        movement: str,
        sport: str,
    ) -> Tuple[float, List[str]]:
        """Equipment interaction bonus: proximity of limbs to ball/racket. Returns (bonus 0-5, feedbacks)."""
        bonus = 0.0
        feedbacks: List[str] = []
        if not objects or not landmarks:
            return bonus, feedbacks
        mov = (movement or "").lower()
        for obj in objects[:3]:  # Top 3 objects
            label = (getattr(obj, "label", "") or "").lower()
            bbox = getattr(obj, "bbox", (0, 0, 0, 0))
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                cx = float(bbox[0]) + float(bbox[2]) / 2
                cy = float(bbox[1]) + float(bbox[3]) / 2
            else:
                continue
            dist = 1.0
            limb = None
            if "ball" in label or "sports_ball" in label:
                if mov in ("kick", "sprint") or sport in ("football", "soccer"):
                    for name in ["left_ankle", "right_ankle"]:
                        p = self._get_landmark(landmarks, name)
                        if p:
                            d = ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
                            if d < dist:
                                dist, limb = d, name
                elif mov in ("throw", "swing", "jump") or sport in ("basketball", "volleyball"):
                    for name in ["left_wrist", "right_wrist"]:
                        p = self._get_landmark(landmarks, name)
                        if p:
                            d = ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
                            if d < dist:
                                dist, limb = d, name
            elif "racket" in label or "bat" in label:
                for name in ["left_wrist", "right_wrist"]:
                    p = self._get_landmark(landmarks, name)
                    if p:
                        d = ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
                        if d < dist:
                            dist, limb = d, name
            if limb and dist < 0.15:
                bonus += min(2.0, (0.15 - dist) * 20)
                feedbacks.append("Good equipment interaction - limb near target")
            elif limb and dist > 0.25 and mov in ("kick", "throw", "swing"):
                feedbacks.append("Keep equipment in frame for better interaction analysis")
        return min(5.0, bonus), feedbacks

    def evaluate_frame(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        frame_idx: int = 0,
        sport: str = "unknown",
        movement: str = "unknown",
        objects: Optional[List[Any]] = None,
    ) -> MovementEvaluation:
        """
        Evaluate posture with sport-specific criteria.
        Sport and movement determine which joints matter most.
        objects: optional TrackedObject list for equipment interaction scoring.
        """
        joint_scores: List[JointScore] = []
        errors: List[str] = []
        scores_list: List[float] = []
        profile = get_sport_profile(sport)
        key_joints = profile.get("key_joints", ["knees", "hips", "shoulders"])
        tech_movements = profile.get("technical_movements", [])
        key_movements = {m.get("generic", "").lower() for m in tech_movements if m.get("generic")}
        critical_errors = set(e.lower() for e in profile.get("critical_errors", []))
        is_key_movement = (movement or "").lower() in key_movements

        if not landmarks:
            return MovementEvaluation(
                frame_idx=frame_idx,
                overall_score=0.0,
                is_correct=False,
                errors=["No body detected - ensure full body is visible in frame"],
                sport=sport,
                movement=movement,
            )

        # Feature engineering: angles, symmetry, stability
        features = extract_frame_features(landmarks, self._prev_features, self.fps)
        self._prev_features = features

        # Biomechanics: check safe ranges and populate joint_risk_levels
        joint_risk_levels: Dict[str, str] = {}
        for joint, angle in {**features.knee_angles, **features.hip_angles, **features.ankle_angles,
                            **features.shoulder_angles, **features.elbow_angles}.items():
            risk, _ = check_angle_safety(angle, joint, sport, movement)
            if risk != "safe":
                joint_risk_levels[joint] = risk

        l_hip = self._get_landmark(landmarks, "left_hip")
        l_knee = self._get_landmark(landmarks, "left_knee")
        l_ankle = self._get_landmark(landmarks, "left_ankle")
        r_hip = self._get_landmark(landmarks, "right_hip")
        r_knee = self._get_landmark(landmarks, "right_knee")
        r_ankle = self._get_landmark(landmarks, "right_ankle")
        l_sh = self._get_landmark(landmarks, "left_shoulder")
        r_sh = self._get_landmark(landmarks, "right_shoulder")
        nose = self._get_landmark(landmarks, "nose")

        # Knee - sport- and movement-specific ideal
        ideal_knee = _get_ideal(sport, "knee", movement)
        for side, hip, knee, ankle in [("left", l_hip, l_knee, l_ankle), ("right", r_hip, r_knee, r_ankle)]:
            if hip and knee and ankle:
                angle = self._angle_between_points(hip, knee, ankle)
                score = self._score_angle(angle, ideal_knee[0], ideal_knee[1], "knee")
                hip_ankle_mid_x = (hip[0] + ankle[0]) / 2
                valgus_left = side == "left" and knee[0] < hip_ankle_mid_x - 0.02
                valgus_right = side == "right" and knee[0] > hip_ankle_mid_x + 0.02
                if valgus_left or valgus_right:
                    score = min(score, 50)
                    errors.append(f"Knee valgus ({side}) - keep knees aligned over toes")
                    joint_risk_levels[f"{side}_knee"] = "high"
                joint_scores.append(
                    JointScore(
                        name=f"{side}_knee",
                        score=float(score),
                        is_correct=bool(score >= 70),
                        feedback=f"Knee angle: {angle:.0f}°" + (" ✓" if score >= 70 else " - needs improvement"),
                        risk_level=joint_risk_levels.get(f"{side}_knee", "safe"),
                    )
                )
                scores_list.append(score)

        # Shoulders
        if l_sh and r_sh:
            dy = abs(l_sh[1] - r_sh[1])
            score = max(0, 100 - dy * 500)
            joint_scores.append(
                JointScore(
                    name="shoulders",
                    score=float(score),
                    is_correct=bool(score >= 70),
                    feedback="Shoulders level ✓" if score >= 70 else "Shoulder imbalance",
                )
            )
            scores_list.append(score)
            if score < 70:
                errors.append("Shoulder imbalance")

        # Hips
        if l_hip and r_hip:
            dy = abs(l_hip[1] - r_hip[1])
            score = max(0, 100 - dy * 500)
            joint_scores.append(
                JointScore(
                    name="hips",
                    score=float(score),
                    is_correct=bool(score >= 70),
                    feedback="Hips level ✓" if score >= 70 else "Hip imbalance",
                )
            )
            scores_list.append(score)

        # Hip extension (sport-specific for running, football, weightlifting)
        if "poor_hip_extension" in critical_errors or sport in ("running", "football", "weightlifting"):
            if nose and l_hip and l_knee:
                hip_angle = self._angle_between_points(nose, l_hip, l_knee)
                ideal_hip = _get_ideal(sport, "hip", movement)
                if hip_angle < ideal_hip[0] and movement in ("squat", "lunge", "sprint", "kick"):
                    errors.append("Poor hip extension")
                    scores_list.append(60)

        # Ankles
        if l_ankle and r_ankle:
            dy = abs(l_ankle[1] - r_ankle[1])
            score = max(0, 100 - dy * 400)
            joint_scores.append(
                JointScore(
                    name="ankles",
                    score=float(score),
                    is_correct=bool(score >= 65),
                    feedback="Ankle stability ✓" if score >= 65 else "Improve ankle stability",
                )
            )
            scores_list.append(score)
            if score < 65:
                errors.append("Unstable posture")

        # Elbow - sport- and movement-specific
        ideal_elbow = _get_ideal(sport, "elbow", movement)
        for side, sh, elb, wr in [
            ("left", l_sh, self._get_landmark(landmarks, "left_elbow"), self._get_landmark(landmarks, "left_wrist")),
            ("right", r_sh, self._get_landmark(landmarks, "right_elbow"), self._get_landmark(landmarks, "right_wrist")),
        ]:
            if sh and elb and wr:
                angle = self._angle_between_points(sh, elb, wr)
                score = self._score_angle(angle, ideal_elbow[0], ideal_elbow[1], "elbow")
                joint_scores.append(
                    JointScore(
                        name=f"{side}_elbow",
                        score=float(score),
                        is_correct=bool(score >= 60),
                        feedback=f"Elbow angle: {angle:.0f}°",
                    )
                )
                scores_list.append(score)

        # Rotation (tennis, boxing, baseball)
        if l_sh and r_sh and l_hip and r_hip and "limited_rotation" in critical_errors:
            shoulder_width = abs(l_sh[0] - r_sh[0])
            hip_width = abs(l_hip[0] - r_hip[0])
            twist = abs(shoulder_width - hip_width)
            if shoulder_width > 0.01:
                rotation_score = max(0, 100 - twist / shoulder_width * 200)
                if rotation_score < 60 and movement in ("swing", "throw", "rotation", "punch"):
                    errors.append("Limited rotation - drive hips and shoulders together")
                scores_list.append(min(100, rotation_score))

        # Weight scores: key joints for this sport get 1.5x, key movement gets stricter threshold
        weights = []
        joint_names_used = [j.name for j in joint_scores]
        for j in joint_scores:
            w = 1.0
            for kj in key_joints:
                if kj in j.name or j.name in kj:
                    w = 1.5
                    break
            weights.append(w)
        if weights and len(weights) == len(scores_list):
            overall = float(np.average(scores_list, weights=weights))
        else:
            overall = float(np.mean(scores_list)) if scores_list else 0.0
        # Equipment interaction bonus (0-5) when ball/racket detected and limb proximity good
        eq_bonus, eq_feedbacks = self._equipment_interaction_score(
            landmarks, objects or [], movement, sport
        )
        overall = min(100.0, overall + eq_bonus)
        for fb in eq_feedbacks:
            if "Good equipment" in fb and fb not in errors:
                errors.append(fb)
            elif "Keep equipment" in fb and fb not in errors:
                errors.append(fb)

        # Skill metrics: speed, power, balance, accuracy, timing, alignment (0-10 scale)
        balance_score = 0.0
        if l_sh and r_sh and l_hip and r_hip:
            sh_dy = abs(l_sh[1] - r_sh[1])
            hip_dy = abs(l_hip[1] - r_hip[1])
            balance_score = max(0, 100 - (sh_dy + hip_dy) * 400)
        accuracy_score = min(100, 70 + eq_bonus * 6) if objects else 70.0
        knee_align = next((j.score for j in joint_scores if "knee" in j.name.lower()), 70)
        skill_metrics = {
            "balance": round(balance_score / 10.0, 1),
            "accuracy": round(accuracy_score / 10.0, 1),
            "alignment": round(knee_align / 10.0, 1),
            "timing": round(min(10, overall / 12), 1),  # Derived from form score
            "power": round(min(10, overall / 11), 1),  # Placeholder - range of motion proxy
        }
        threshold = 65 if is_key_movement else 70
        is_correct = bool(overall >= threshold and len(errors) <= 2)

        # Derive injury-risk warnings from errors (for poor mechanics detection)
        injury_warnings: List[str] = []
        err_lower = " ".join(errors).lower()
        for err_key, warning in INJURY_RISK_MAP.items():
            if err_key.replace("_", " ") in err_lower or err_key in err_lower:
                if warning not in injury_warnings:
                    injury_warnings.append(warning)

        # Add knee_angle_unsafe when outside safe range
        for jname, ang in features.knee_angles.items():
            risk, _ = check_angle_safety(ang, jname, sport, movement)
            if risk == "high":
                errors.append("Knee angle outside safe range (landing/load)")
                joint_risk_levels[jname] = "high"

        # Injury Risk Score 0-100
        injury_risk_score = compute_injury_risk_score(errors, joint_risk_levels)

        # Possible injuries from error-to-coaching map
        possible_injuries: List[str] = []
        for err in errors:
            _, inj_list = get_coaching_for_error(err)
            for i in inj_list:
                if i not in possible_injuries:
                    possible_injuries.append(i)

        # Confidence: higher when symmetry and stability are good
        confidence = (features.left_right_symmetry + features.stability_score) / 2.0

        # Compact features for training/continuous improvement
        features_for_training = {
            "knee_angles": dict(features.knee_angles),
            "hip_angles": dict(features.hip_angles),
            "ankle_angles": dict(features.ankle_angles),
            "shoulder_angles": dict(features.shoulder_angles),
            "elbow_angles": dict(features.elbow_angles),
            "symmetry": round(features.left_right_symmetry, 3),
            "stability": round(features.stability_score, 3),
        }

        return MovementEvaluation(
            frame_idx=frame_idx,
            overall_score=round(overall, 1),
            is_correct=is_correct,
            joint_scores=joint_scores,
            errors=list(dict.fromkeys(errors))[:6],
            sport=sport,
            movement=movement,
            skill_metrics=skill_metrics,
            injury_risk_warnings=injury_warnings[:5],
            injury_risk_score=round(injury_risk_score, 1),
            joint_risk_levels=joint_risk_levels,
            possible_injuries=possible_injuries[:5],
            confidence=round(confidence, 2),
            features_for_training=features_for_training,
        )


# --- Recommendations (merged from recommendations.py) ---


@dataclass
class CorrectiveExercise:
    """A corrective exercise recommendation with official source citation."""

    name: str
    description: str
    target_joint: str
    reps_sets: str
    difficulty: str
    sport_focus: str = ""
    source: str = ""  # Official source citation (e.g. "NASM (National Academy of Sports Medicine)")


# Built from backend.analysis.exercises.GENERIC_EXERCISES_RAW
_GENERIC_EXERCISE_SOURCE = "NASM Corrective Exercise (National Academy of Sports Medicine)"

GENERIC_EXERCISES: Dict[str, List[CorrectiveExercise]] = {
    k: [
        CorrectiveExercise(
            ex["name"],
            ex["description"],
            ex["target_joint"],
            ex["reps_sets"],
            ex["difficulty"],
            "",
            ex.get("source", _GENERIC_EXERCISE_SOURCE),
        )
        for ex in raw_list
    ]
    for k, raw_list in GENERIC_EXERCISES_RAW.items()
}


def _exercise_from_profile(ex: dict, sport: str) -> CorrectiveExercise:
    """
    Build CorrectiveExercise from sport profile exercise dict.
    Uses name, reason/description, target joint; source from sport federation.
    """
    srcs = get_sources_for_sport(sport)
    src_str = format_source_short(srcs[0]) if srcs else _GENERIC_EXERCISE_SOURCE
    return CorrectiveExercise(
        name=ex.get("name", ""),
        description=ex.get("reason", ex.get("description", "")),
        target_joint=ex.get("target", ""),
        reps_sets="3x12",
        difficulty="beginner",
        sport_focus=sport,
        source=src_str,
    )


class RecommendationEngine:
    """Generate sport-specific recommendations and development plans."""

    def get_recommendations(
        self,
        errors: List[str],
        joint_scores: List[Union[dict, Any]],
        sport: str = "unknown",
    ) -> List[CorrectiveExercise]:
        """Get sport-specific corrective exercises."""
        recommendations: List[CorrectiveExercise] = []
        seen = set()
        sport = (sport or "unknown").lower().strip()
        profile_exercises = get_sport_exercises(sport, errors or [])
        for ex in profile_exercises:
            rec = _exercise_from_profile(ex, sport)
            if rec.name and rec.name not in seen:
                seen.add(rec.name)
                recommendations.append(rec)
        for err in (errors or []):
            err_lower = str(err).lower()
            for keyword, exercises in GENERIC_EXERCISES.items():
                if keyword in err_lower:
                    for ex in exercises[:1]:
                        if ex.name not in seen:
                            seen.add(ex.name)
                            recommendations.append(ex)
                    break
        for js in (joint_scores or []):
            name = safe_get(js, "name", "")
            score = safe_get(js, "score", 100)
            if isinstance(score, (int, float)) and score < 70:
                target = name.split("_")[-1] if "_" in str(name) else str(name)
                for keyword, exercises in GENERIC_EXERCISES.items():
                    if keyword in target.lower():
                        for ex in exercises[:1]:
                            if ex.name not in seen:
                                seen.add(ex.name)
                                recommendations.append(ex)
                        break
        return recommendations[:5]

    def get_development_plan(self, sport: str, score: float, errors: List[str]) -> List[str]:
        """Get sport-specific professional development roadmap."""
        return get_development_plan(sport, score, errors or [])

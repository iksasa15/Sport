"""
Sport-specific evaluation criteria, technical movements, and coaching profiles.

Each sport has:
- technical_movements: all analyzable movements with mapping to detection signals
- ideal_angles: joint angles per movement
- critical_errors: sport-specific errors to detect
- coaching_tips: feedback templates per error
"""

from typing import Dict, List, Optional, Tuple

from backend.analysis.exercises import get_exercises_for_sport

# Generic movements detected by MovementRecognizer -> sport-specific mapping
GENERIC_MOVEMENTS = ["kick", "jump", "sprint", "punch", "swing", "throw", "squat", "lunge", "rotation", "static"]

# Sport-specific relevant objects (ignore floor, sticks, weights not related to sport)
SPORT_RELEVANT_OBJECTS: Dict[str, set] = {
    "football": {"ball", "sports_ball", "orange_basketball", "yellow_tennis", "white_ball", "green_ball"},
    "soccer": {"ball", "sports_ball", "orange_basketball", "yellow_tennis", "white_ball", "green_ball"},
    "basketball": {"ball", "sports_ball", "orange_basketball"},
    "tennis": {"racket", "tennis_racket", "ball", "sports_ball", "yellow_tennis"},
    "golf": {"ball", "sports_ball", "green_ball"},
    "baseball": {"bat", "baseball_bat", "ball", "sports_ball", "white_ball"},
    "volleyball": {"ball", "sports_ball"},
    "weightlifting": {"barbell"},
    "hockey": {"stick"},
    "boxing": set(),  # no equipment
    "yoga": set(),
    "running": set(),
    "swimming": set(),
    "general_fitness": {"barbell"},
    "unknown": {"ball", "sports_ball", "racket", "bat", "barbell", "stick"},
}

SPORT_PROFILES: Dict[str, dict] = {
    "football": {
        "name": "كرة القدم",
        "name_en": "Football (Soccer)",
        "technical_movements": [
            {"id": "ball_striking", "name_en": "Ball Striking / Shooting", "name_ar": "ضرب وتصويب الكرة", "generic": "kick"},
            {"id": "dribbling", "name_en": "Dribbling", "name_ar": "المراوغة", "generic": "sprint"},
            {"id": "passing", "name_en": "Passing", "name_ar": "التمرير", "generic": "kick"},
            {"id": "shooting", "name_en": "Shooting", "name_ar": "التصويب", "generic": "kick"},
            {"id": "sprinting", "name_en": "Sprinting", "name_ar": "الجري السريع", "generic": "sprint"},
            {"id": "cutting", "name_en": "Cutting", "name_ar": "القطع والتغيير", "generic": "lunge"},
            {"id": "juggling", "name_en": "Juggling", "name_ar": "الخداع", "generic": "kick"},
            {"id": "jump_header", "name_en": "Jump Header", "name_ar": "الضربة الرأسية", "generic": "jump"},
        ],
        "key_joints": ["left_knee", "right_knee", "left_hip", "right_hip", "ankles"],
        "ideal_angles": {
            "knee_kick": (140, 180), "knee_sprint": (155, 175), "hip_kick": (80, 120),
            "knee_jump": (140, 170), "hip_lunge": (80, 120),
        },
        "critical_errors": ["knee_valgus", "poor_hip_extension", "ankle_instability"],
        "coaching_tips": {
            "knee_valgus": "Keep knees aligned over toes during striking and landing.",
            "poor_hip_extension": "Drive through the hip for power in kicks and sprints.",
            "ankle_instability": "Strengthen ankles with single-leg balance drills.",
        },
        "development_plan": [
            "Phase 1: Ball control and warm-up basics",
            "Phase 2: Passing accuracy at various distances",
            "Phase 3: Shooting - focus on hip and knee angles",
            "Phase 4: Agility and speed training",
            "Phase 5: Tactics and team play",
        ],
        "exercises": [],  # من backend.analysis.exercises
    },
    "basketball": {
        "name": "كرة السلة",
        "name_en": "Basketball",
        "technical_movements": [
            {"id": "shooting", "name_en": "Shooting", "name_ar": "التصويب", "generic": "throw"},
            {"id": "passing", "name_en": "Passing", "name_ar": "التمرير", "generic": "throw"},
            {"id": "dribbling", "name_en": "Dribbling", "name_ar": "المراوغة", "generic": "sprint"},
            {"id": "layup", "name_en": "Layup", "name_ar": "السلة السهلة", "generic": "jump"},
            {"id": "rebound", "name_en": "Rebound", "name_ar": "الارتداد", "generic": "jump"},
            {"id": "defense", "name_en": "Defense", "name_ar": "الدفاع", "generic": "lunge"},
            {"id": "footwork", "name_en": "Footwork", "name_ar": "حركة الأقدام", "generic": "lunge"},
            {"id": "jump_shot", "name_en": "Jump Shot", "name_ar": "التصويب بالقفز", "generic": "jump"},
        ],
        "key_joints": ["knees", "ankles", "shoulders", "elbows"],
        "ideal_angles": {
            "knee_jump": (140, 170), "elbow_shoot": (80, 100),
            "knee_landing": (140, 170), "knee_lunge": (70, 120),
        },
        "critical_errors": ["knee_valgus", "unstable_landing", "shoulder_imbalance"],
        "coaching_tips": {
            "knee_valgus": "Land with knees over toes; avoid inward collapse.",
            "unstable_landing": "Practice soft landings with bent knees.",
            "shoulder_imbalance": "Keep shooting elbow aligned; follow through.",
        },
        "development_plan": [
            "Phase 1: Ball handling and warm-up",
            "Phase 2: Shooting form - arm and hand positioning",
            "Phase 3: Jump and safe landing mechanics",
            "Phase 4: Dribbling and agility",
            "Phase 5: Team play and competition",
        ],
        "exercises": [],
    },
    "tennis": {
        "name": "التنس",
        "name_en": "Tennis",
        "technical_movements": [
            {"id": "serve", "name_en": "Serve", "name_ar": "الإرسال", "generic": "throw"},
            {"id": "forehand", "name_en": "Forehand", "name_ar": "الضربة الأمامية", "generic": "swing"},
            {"id": "backhand", "name_en": "Backhand", "name_ar": "الضربة الخلفية", "generic": "swing"},
            {"id": "volley", "name_en": "Volley", "name_ar": "الضربة الهوائية", "generic": "swing"},
            {"id": "movement", "name_en": "Court Movement", "name_ar": "التحرك في الملعب", "generic": "sprint"},
        ],
        "key_joints": ["shoulders", "elbows", "hips", "core"],
        "ideal_angles": {
            "elbow_swing": (90, 150), "shoulder_swing": (60, 120),
            "hip_rotation": (30, 90), "elbow_throw": (90, 140),
        },
        "critical_errors": ["shoulder_imbalance", "limited_rotation", "elbow_alignment"],
        "coaching_tips": {
            "shoulder_imbalance": "Rotate shoulders through the stroke.",
            "limited_rotation": "Drive from hips and core; unit turn.",
            "elbow_alignment": "Keep elbow in front; avoid dropping.",
        },
        "development_plan": [
            "Phase 1: Grip and ready position",
            "Phase 2: Groundstrokes - forehand and backhand",
            "Phase 3: Timing and shoulder-hip rotation",
            "Phase 4: Serve and volley",
            "Phase 5: Tactics and court positioning",
        ],
        "exercises": [],
    },
    "gymnastics": {
        "name": "الجمباز",
        "name_en": "Gymnastics",
        "technical_movements": [
            {"id": "floor_routine", "name_en": "Floor Routine", "name_ar": "تمرين الأرض", "generic": "rotation"},
            {"id": "beam", "name_en": "Beam", "name_ar": "عارضة التوازن", "generic": "static"},
            {"id": "vault", "name_en": "Vault", "name_ar": "القفز على طاولة القفز", "generic": "jump"},
            {"id": "tumbling", "name_en": "Tumbling", "name_ar": "الدحرجات", "generic": "rotation"},
        ],
        "key_joints": ["shoulders", "hips", "core", "ankles"],
        "ideal_angles": {},
        "critical_errors": ["shoulder_imbalance", "unstable_landing", "core_instability"],
        "coaching_tips": {
            "shoulder_imbalance": "Maintain shoulder alignment in holds.",
            "unstable_landing": "Stick landings with soft knees and core engaged.",
            "core_instability": "Hollow body and plank progressions.",
        },
        "development_plan": [
            "Phase 1: Flexibility and basics",
            "Phase 2: Strength and conditioning",
            "Phase 3: Skill progressions",
            "Phase 4: Routine development",
            "Phase 5: Competition readiness",
        ],
        "exercises": [],
    },
    "martial_arts": {
        "name": "الفنون القتالية",
        "name_en": "Martial Arts",
        "technical_movements": [
            {"id": "punches", "name_en": "Strikes", "name_ar": "الضربات", "generic": "punch"},
            {"id": "kicks", "name_en": "Kicks", "name_ar": "الركلات", "generic": "kick"},
            {"id": "defense", "name_en": "Defense", "name_ar": "الدفاع", "generic": "lunge"},
            {"id": "footwork", "name_en": "Footwork", "name_ar": "حركة الأقدام", "generic": "lunge"},
        ],
        "key_joints": ["shoulders", "elbows", "hips", "core"],
        "ideal_angles": {"elbow_punch": (150, 180), "hip_rotation": (20, 60)},
        "critical_errors": ["shoulder_imbalance", "elbow_drop", "limited_rotation"],
        "coaching_tips": {
            "shoulder_imbalance": "Keep guard up; retract strikes.",
            "elbow_drop": "Elbows in; protect body.",
            "limited_rotation": "Rotate hips for power.",
        },
        "development_plan": [
            "Phase 1: Stance and basics",
            "Phase 2: Strikes and kicks",
            "Phase 3: Defense and timing",
            "Phase 4: Combinations",
            "Phase 5: Sparring",
        ],
        "exercises": [],
    },
    "boxing": {
        "name": "الملاكمة",
        "name_en": "Boxing",
        "technical_movements": [
            {"id": "punches", "name_en": "Punches (Jab, Cross, Hook)", "name_ar": "اللكمات", "generic": "punch"},
            {"id": "defense", "name_en": "Defense (Block, Slip)", "name_ar": "الدفاع", "generic": "lunge"},
            {"id": "footwork", "name_en": "Footwork", "name_ar": "حركة الأقدام", "generic": "lunge"},
            {"id": "rotation", "name_en": "Rotation / Power Transfer", "name_ar": "الدوران", "generic": "rotation"},
        ],
        "key_joints": ["shoulders", "elbows", "hips", "core"],
        "ideal_angles": {
            "elbow_punch": (150, 180), "hip_rotation": (20, 60),
            "shoulder_punch": (0, 15),
        },
        "critical_errors": ["shoulder_imbalance", "elbow_drop", "limited_rotation"],
        "coaching_tips": {
            "shoulder_imbalance": "Keep guard up; retract punches.",
            "elbow_drop": "Elbows in; protect body.",
            "limited_rotation": "Rotate hips into punches for power.",
        },
        "development_plan": [
            "Phase 1: Stance and basic movement",
            "Phase 2: Punches - jab, cross, hook",
            "Phase 3: Timing and defense",
            "Phase 4: Combinations and footwork",
            "Phase 5: Sparring and competition",
        ],
        "exercises": [],
    },
    "weightlifting": {
        "name": "رفع الأثقال",
        "name_en": "Weightlifting / Strength",
        "technical_movements": [
            {"id": "snatch", "name_en": "Snatch", "name_ar": "الخطف", "generic": "throw"},
            {"id": "clean_jerk", "name_en": "Clean & Jerk", "name_ar": "النتر والاندفاع", "generic": "throw"},
            {"id": "squat", "name_en": "Squat", "name_ar": "القرفصاء", "generic": "squat"},
            {"id": "deadlift", "name_en": "Deadlift", "name_ar": "الرفعة المميتة", "generic": "squat"},
        ],
        "key_joints": ["knees", "hips", "shoulders", "core"],
        "ideal_angles": {
            "knee_squat": (70, 110), "hip_squat": (60, 100),
            "knee_lunge": (70, 120), "hip_hinge": (60, 100),
        },
        "critical_errors": ["knee_valgus", "poor_hip_extension", "shoulder_imbalance"],
        "coaching_tips": {
            "knee_valgus": "Push knees out over toes in squat.",
            "poor_hip_extension": "Drive hips through at lockout.",
            "shoulder_imbalance": "Keep barbell path vertical; symmetric pull.",
        },
        "development_plan": [
            "Phase 1: Mobility - bodyweight squat and hinge",
            "Phase 2: Bar path and spinal alignment",
            "Phase 3: Progressive load",
            "Phase 4: Strength program (e.g. 5x5)",
            "Phase 5: Max weights and competition",
        ],
        "exercises": [],
    },
    "running": {
        "name": "الجري",
        "name_en": "Running / Track",
        "technical_movements": [
            {"id": "sprint_start", "name_en": "Sprint Start", "name_ar": "انطلاقة العدو", "generic": "squat"},
            {"id": "acceleration", "name_en": "Acceleration", "name_ar": "التسارع", "generic": "sprint"},
            {"id": "stride", "name_en": "Stride / Form", "name_ar": "الخطوة", "generic": "sprint"},
            {"id": "finish", "name_en": "Finish / Lean", "name_ar": "الوصول", "generic": "sprint"},
        ],
        "key_joints": ["knees", "hips", "ankles"],
        "ideal_angles": {
            "knee_sprint": (160, 180), "hip_sprint": (170, 200),
        },
        "critical_errors": ["knee_valgus", "poor_hip_extension", "unstable_posture"],
        "coaching_tips": {
            "knee_valgus": "Knees drive forward, not inward.",
            "poor_hip_extension": "Full hip extension at toe-off.",
            "unstable_posture": "Upright posture; slight lean from ankles.",
        },
        "development_plan": [
            "Phase 1: Walk-to-run progression",
            "Phase 2: Intervals and endurance",
            "Phase 3: Distance and pace",
            "Phase 4: Speed and economy",
            "Phase 5: Racing",
        ],
        "exercises": [],
    },
    "yoga": {
        "name": "اليوغا",
        "name_en": "Yoga / Flexibility",
        "technical_movements": [
            {"id": "poses", "name_en": "Poses", "name_ar": "الوضعيات", "generic": "static"},
            {"id": "balance", "name_en": "Balance", "name_ar": "التوازن", "generic": "static"},
            {"id": "alignment", "name_en": "Alignment", "name_ar": "المحاذاة", "generic": "static"},
        ],
        "key_joints": ["shoulders", "hips", "spine"],
        "ideal_angles": {},
        "critical_errors": ["shoulder_imbalance", "core_instability", "limited_flexibility"],
        "coaching_tips": {
            "shoulder_imbalance": "Stack joints; relax shoulders.",
            "core_instability": "Engage core gently; avoid overarching.",
            "limited_flexibility": "Breathe into stretches; no forcing.",
        },
        "development_plan": [
            "Phase 1: Breath and basic poses",
            "Phase 2: Alignment and holding",
            "Phase 3: Balance poses",
            "Phase 4: Flows and sequences",
            "Phase 5: Advanced practice",
        ],
        "exercises": [],
    },
    "golf": {
        "name": "الجولف",
        "name_en": "Golf",
        "technical_movements": [
            {"id": "full_swing", "name_en": "Full Swing", "name_ar": "الضربة الكاملة", "generic": "swing"},
            {"id": "chip", "name_en": "Chip / Pitch", "name_ar": "الضربة القصيرة", "generic": "swing"},
            {"id": "putt", "name_en": "Putting", "name_ar": "الضغط", "generic": "static"},
        ],
        "key_joints": ["shoulders", "hips", "elbows", "core"],
        "ideal_angles": {"elbow_swing": (90, 150), "hip_swing": (80, 120), "hip_rotation": (30, 90)},
        "critical_errors": ["limited_rotation", "shoulder_imbalance"],
        "coaching_tips": {
            "limited_rotation": "Turn shoulders 90°; coil and release.",
            "shoulder_imbalance": "Level shoulders; spine angle consistent.",
        },
        "development_plan": [
            "Phase 1: Grip and setup",
            "Phase 2: Backswing",
            "Phase 3: Downswing and weight shift",
            "Phase 4: Follow-through",
            "Phase 5: Accuracy and distance",
        ],
        "exercises": [],
    },
    "baseball": {
        "name": "البيسبول",
        "name_en": "Baseball",
        "technical_movements": [
            {"id": "throwing", "name_en": "Throwing", "name_ar": "الرمي", "generic": "throw"},
            {"id": "batting", "name_en": "Batting", "name_ar": "الضرب", "generic": "swing"},
        ],
        "key_joints": ["shoulders", "elbows", "hips", "core"],
        "ideal_angles": {"elbow_throw": (80, 140), "elbow_swing": (80, 120), "shoulder_throw": (60, 120)},
        "critical_errors": ["limited_rotation", "shoulder_imbalance"],
        "coaching_tips": {
            "limited_rotation": "Hip-shoulder separation for velocity.",
            "shoulder_imbalance": "Rotator cuff strengthening; follow-through.",
        },
        "development_plan": [
            "Phase 1: Grip and stance",
            "Phase 2: Throwing mechanics",
            "Phase 3: Batting - timing and contact",
            "Phase 4: Advanced throwing",
            "Phase 5: Game play",
        ],
        "exercises": [],
    },
    "volleyball": {
        "name": "الكرة الطائرة",
        "name_en": "Volleyball",
        "technical_movements": [
            {"id": "serve", "name_en": "Serve", "name_ar": "الإرسال", "generic": "throw"},
            {"id": "spike", "name_en": "Spike", "name_ar": "الضرب", "generic": "jump"},
            {"id": "pass", "name_en": "Pass / Receive", "name_ar": "الاستقبال", "generic": "squat"},
            {"id": "block", "name_en": "Block", "name_ar": "الحائط", "generic": "jump"},
        ],
        "key_joints": ["knees", "shoulders", "elbows"],
        "ideal_angles": {"knee_jump": (140, 170), "elbow_jump": (80, 120)},
        "critical_errors": ["knee_valgus", "unstable_landing"],
        "coaching_tips": {
            "knee_valgus": "Knees over toes on jump and land.",
            "unstable_landing": "Land soft; absorb with legs.",
        },
        "development_plan": [
            "Phase 1: Serve and receive",
            "Phase 2: Overhead attack",
            "Phase 3: Jump and spike timing",
            "Phase 4: Safe landing",
            "Phase 5: Tactics",
        ],
        "exercises": [],
    },
    "hockey": {
        "name": "الهوكي",
        "name_en": "Hockey",
        "technical_movements": [
            {"id": "skating", "name_en": "Skating", "name_ar": "التزحلق", "generic": "sprint"},
            {"id": "stick_handling", "name_en": "Stick Handling", "name_ar": "التعامل مع العصا", "generic": "swing"},
            {"id": "shooting", "name_en": "Shooting", "name_ar": "التسديد", "generic": "swing"},
        ],
        "key_joints": ["hips", "knees", "ankles"],
        "ideal_angles": {},
        "critical_errors": ["knee_valgus", "limited_rotation"],
        "coaching_tips": {},
        "development_plan": ["Phase 1-5: Skating fundamentals to advanced play"],
        "exercises": [],
    },
    "swimming": {
        "name": "السباحة",
        "name_en": "Swimming",
        "technical_movements": [
            {"id": "stroke", "name_en": "Stroke", "name_ar": "ضربة اليد", "generic": "swing"},
            {"id": "kick", "name_en": "Kick", "name_ar": "ضربة الرجل", "generic": "kick"},
            {"id": "breathing", "name_en": "Breathing / Rotation", "name_ar": "التنفس والدوران", "generic": "rotation"},
        ],
        "key_joints": ["shoulders", "hips", "core"],
        "ideal_angles": {},
        "critical_errors": ["shoulder_imbalance", "limited_rotation"],
        "coaching_tips": {},
        "development_plan": ["Phase 1-5: Technique to competition"],
        "exercises": [],
    },
    "general_fitness": {
        "name": "اللياقة العامة",
        "name_en": "General Fitness",
        "technical_movements": [
            {"id": "squat", "name_en": "Squat", "name_ar": "القرفصاء", "generic": "squat"},
            {"id": "lunge", "name_en": "Lunge", "name_ar": "الاندفاع", "generic": "lunge"},
            {"id": "jump", "name_en": "Jump", "name_ar": "القفز", "generic": "jump"},
            {"id": "sprint", "name_en": "Sprint", "name_ar": "الجري السريع", "generic": "sprint"},
            {"id": "rotation", "name_en": "Rotation", "name_ar": "الدوران", "generic": "rotation"},
        ],
        "key_joints": ["knees", "hips", "shoulders", "core"],
        "ideal_angles": {"knee_squat": (70, 110), "knee_lunge": (80, 120)},
        "critical_errors": ["knee_valgus", "poor_posture", "imbalance"],
        "coaching_tips": {
            "knee_valgus": "Keep knees over toes during squats and lunges.",
            "poor_posture": "Maintain neutral spine; engage core.",
            "imbalance": "Work both sides equally for symmetry.",
        },
        "development_plan": [
            "Phase 1: Mobility and warm-up",
            "Phase 2: Strength foundations (squat, lunge)",
            "Phase 3: Power (jumps, sprints)",
            "Phase 4: Integration and conditioning",
        ],
        "exercises": [],
    },
    "unknown": {
        "name": "غير محدد",
        "name_en": "General",
        "technical_movements": [],
        "key_joints": ["knees", "hips", "shoulders"],
        "ideal_angles": {},
        "critical_errors": [],
        "coaching_tips": {},
        "development_plan": [
            "Select a sport for movement-specific analysis",
            "Focus on general posture and movement quality",
        ],
        "exercises": [],
    },
}

# Fallback generic angles per sport (used when movement-specific not in profile.ideal_angles)
GENERIC_SPORT_ANGLES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "football": {"knee": (155, 175), "hip": (165, 185), "elbow": (85, 95)},
    "tennis": {"knee": (140, 170), "elbow": (90, 150), "shoulder": (80, 120)},
    "weightlifting": {"knee": (70, 110), "hip": (60, 100), "back": (40, 70)},
    "boxing": {"elbow": (150, 180), "shoulder": (0, 15)},
    "basketball": {"knee": (140, 170), "elbow": (80, 100)},
    "running": {"knee": (160, 180), "hip": (170, 200)},
    "golf": {"knee": (150, 170), "hip": (80, 120), "elbow": (90, 150)},
    "baseball": {"elbow": (80, 140), "shoulder": (60, 120)},
    "volleyball": {"knee": (140, 170), "elbow": (80, 120)},
    "soccer": {"knee": (155, 175), "hip": (165, 185), "elbow": (85, 95)},
    "track": {"knee": (155, 175), "hip": (90, 130)},
    "general_fitness": {"knee": (70, 120), "hip": (80, 120), "shoulder": (80, 120)},
    "hockey": {"knee": (140, 170), "hip": (80, 120)},
    "swimming": {"shoulder": (60, 120), "hip": (80, 120)},
}


def get_relevant_object_labels(sport: str) -> set:
    """Return object labels relevant to sport. Irrelevant objects (floor, wrong equipment) are ignored."""
    key = (sport or "").lower().strip()
    return SPORT_RELEVANT_OBJECTS.get(key, SPORT_RELEVANT_OBJECTS["unknown"])


def get_ideal_angle_fallback(sport: str, joint_type: str) -> Optional[Tuple[float, float]]:
    """Get generic fallback angle range for sport+joint. Used by evaluator when movement-specific not found."""
    key = (sport or "").lower().strip()
    angles = GENERIC_SPORT_ANGLES.get(key) or GENERIC_SPORT_ANGLES.get("football")
    return angles.get(joint_type) if angles else None


def get_sport_profile(sport: str) -> dict:
    """Get profile for sport. Uses modular analyzer when available, else legacy SPORT_PROFILES."""
    key = sport.lower().strip() if sport else "unknown"
    if key == "soccer":
        key = "football"  # soccer uses football profile
    try:
        from backend.sports.registry import get_analyzer
        analyzer = get_analyzer(key)
        if analyzer:
            profile = analyzer.get_profile()
            profile["exercises"] = get_exercises_for_sport(key)
            return profile
    except ImportError:
        pass
    profile = SPORT_PROFILES.get(key, SPORT_PROFILES["unknown"]).copy()
    profile["exercises"] = get_exercises_for_sport(key)
    return profile


def get_technical_movements(sport: str) -> List[dict]:
    """Get all technical movements for a sport."""
    profile = get_sport_profile(sport)
    return list(profile.get("technical_movements", []))


def get_movement_by_generic(sport: str, generic: str) -> Optional[dict]:
    """
    Map generic movement (kick, jump, sprint, etc.) to sport-specific movement.
    Returns first matching tech movement dict or None.
    """
    for m in get_technical_movements(sport):
        if m.get("generic") == generic:
            return m
    return None


def get_development_plan(sport: str, score: float, errors: List[str]) -> List[str]:
    """Generate sport-specific professional development plan. score: 0-100 (internal)."""
    profile = get_sport_profile(sport)
    plan = list(profile.get("development_plan", []))
    # Skill level assessment (score 0-100: 40≈4/10, 70≈7/10)
    if score < 40:
        plan.insert(0, "Skill level: Beginner - focus on fundamentals and consistent practice.")
    elif score < 70:
        plan.insert(0, "Skill level: Intermediate - refine technique and address specific errors.")
    else:
        plan.insert(0, "Skill level: Advanced - fine-tune details and optimize performance.")
    if errors:
        plan.insert(1, f"Priority focus: Address '{errors[0]}' in your next training session.")
    return plan[:8]


def get_sport_exercises(sport: str, errors: List[str]) -> List[dict]:
    """Get sport-specific exercises, prioritizing error-related ones."""
    profile = get_sport_profile(sport)
    exercises = list(profile.get("exercises", []))
    error_lower = " ".join(errors).lower()
    priority = [e for e in exercises if e.get("target", "").lower() in error_lower]
    rest = [e for e in exercises if e not in priority]
    return (priority + rest)[:5]


def get_coaching_feedback(sport: str, error: str) -> str:
    """Get coaching tip for an error. Matches by keyword or returns first applicable tip."""
    profile = get_sport_profile(sport)
    tips = profile.get("coaching_tips", {})
    err_lower = (error or "").lower()
    # Direct key match
    if err_lower in tips:
        return tips[err_lower]
    # Keyword match (knee, hip, shoulder, etc.)
    for key, tip in tips.items():
        if key.replace("_", " ") in err_lower or any(
            k in err_lower for k in key.split("_")
        ):
            return tip
    # Default: first tip or generic
    return list(tips.values())[0] if tips else "Focus on correct form and gradual improvement. Practice with slow, controlled movements."


# Equipment-aware coaching enhancements: (object_type) -> extra tip when equipment detected
EQUIPMENT_COACHING_TIPS: Dict[str, str] = {
    "ball": "Focus on contact point with the ball for better accuracy.",
    "sports_ball": "Keep the ball in frame - analyze your strike and follow-through.",
    "racket": "Extend racket follow-through after impact for power and control.",
    "tennis_racket": "Racket path and grip position at contact are key.",
    "bat": "Keep bat level through the strike zone - avoid dropping the hands.",
    "baseball_bat": "Follow through toward target; hip rotation drives power.",
    "barbell": "Barbell path should stay over mid-foot; control the descent.",
    "stick": "Stick blade position at contact affects shot direction.",
}


def get_coaching_feedback_with_equipment(sport: str, error: str, object_labels: List[str]) -> str:
    """Get coaching tip enhanced with equipment-specific advice when equipment is detected."""
    base_tip = get_coaching_feedback(sport, error)
    if not object_labels:
        return base_tip
    relevant = get_relevant_object_labels(sport)
    obj_set = {o.lower().replace(" ", "_") for o in object_labels}
    for obj in obj_set:
        if obj not in relevant:
            continue
        extra = EQUIPMENT_COACHING_TIPS.get(obj)
        if not extra and "ball" in obj:
            extra = EQUIPMENT_COACHING_TIPS.get("sports_ball")
        if not extra and "racket" in obj:
            extra = EQUIPMENT_COACHING_TIPS.get("racket")
        if not extra and "bat" in obj:
            extra = EQUIPMENT_COACHING_TIPS.get("bat")
        if extra:
            return f"{base_tip} {extra}"
    return base_tip


def get_equipment_validation_warnings(sport: str, object_labels: List[str]) -> List[str]:
    """Return warnings when expected equipment for the sport is not detected."""
    expected = SPORT_RELEVANT_OBJECTS.get((sport or "").lower().strip(), set())
    if not expected:
        return []
    obj_set = {o.lower().replace(" ", "_") for o in object_labels}
    detected = bool(expected & obj_set)
    if detected:
        return []
    ball_types = {"ball", "sports_ball", "orange_basketball", "yellow_tennis", "white_ball", "green_ball"}
    if expected & ball_types:
        primary = "ball"
    elif "racket" in expected or "tennis_racket" in expected:
        primary = "racket"
    elif "bat" in expected or "baseball_bat" in expected:
        primary = "bat"
    elif "barbell" in expected:
        primary = "barbell"
    elif "stick" in expected:
        primary = "stick"
    else:
        primary = "equipment"
    return [f"Expected {primary} not detected - ensure full frame includes equipment for better analysis and tips."]

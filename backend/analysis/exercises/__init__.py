"""
تمارين تصحيحية: كل تمرين في ملف بايثون مستقل.
Corrective exercises: each exercise in its own Python file.

الاستخدام: استورد get_exercises_for_sport أو SPORT_EXERCISES أو GENERIC_EXERCISES_RAW
Usage: import get_exercises_for_sport, SPORT_EXERCISES, GENERIC_EXERCISES_RAW
"""

from typing import Dict, List

from .sports import football, basketball, tennis, gymnastics, martial_arts, boxing
from .sports import weightlifting, running, yoga, golf, baseball, volleyball
from .sports import hockey, swimming, general_fitness, unknown
from .generic import GENERIC_EXERCISES_RAW

# تجميع تمارين كل رياضة من المجلدات (كل رياضة لها مجلد، كل تمرين له ملف)
_SPORT_MODULES = {
    "football": football,
    "basketball": basketball,
    "tennis": tennis,
    "gymnastics": gymnastics,
    "martial_arts": martial_arts,
    "boxing": boxing,
    "weightlifting": weightlifting,
    "running": running,
    "yoga": yoga,
    "golf": golf,
    "baseball": baseball,
    "volleyball": volleyball,
    "hockey": hockey,
    "swimming": swimming,
    "general_fitness": general_fitness,
    "unknown": unknown,
}

SPORT_EXERCISES: Dict[str, List[dict]] = {
    key: list(mod.EXERCISES) for key, mod in _SPORT_MODULES.items()
}
SPORT_EXERCISES["soccer"] = SPORT_EXERCISES["football"]
SPORT_EXERCISES["track"] = SPORT_EXERCISES["running"]


def get_exercises_for_sport(sport: str) -> List[dict]:
    """ترجع قائمة التمارين لرياضة معينة. sport بالإنجليزي (مثل football, basketball)."""
    key = (sport or "unknown").lower().strip()
    if key == "soccer":
        key = "football"
    return list(SPORT_EXERCISES.get(key, SPORT_EXERCISES["unknown"]))


__all__ = ["SPORT_EXERCISES", "GENERIC_EXERCISES_RAW", "get_exercises_for_sport"]

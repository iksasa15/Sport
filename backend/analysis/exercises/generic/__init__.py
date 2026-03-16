# Generic corrective exercises (by joint). Each exercise in its own .py file.
from . import wall_sit, clam_shells, band_pull_apart, hip_bridge, calf_raises, plank

_SOURCE = "NASM Corrective Exercise (National Academy of Sports Medicine)"

def _raw(ex):
    return {
        "name": ex["name"],
        "description": ex["description"],
        "target_joint": ex["target_joint"],
        "reps_sets": ex["reps_sets"],
        "difficulty": ex["difficulty"],
        "source": _SOURCE,
    }

GENERIC_EXERCISES_RAW = {
    "knee": [_raw(wall_sit.EXERCISE), _raw(clam_shells.EXERCISE)],
    "shoulder": [_raw(band_pull_apart.EXERCISE)],
    "hip": [_raw(hip_bridge.EXERCISE)],
    "ankle": [_raw(calf_raises.EXERCISE)],
    "core": [_raw(plank.EXERCISE)],
}

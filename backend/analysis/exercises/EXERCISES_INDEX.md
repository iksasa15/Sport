# فهرس ملفات التمارين — للربط

كل سطر = ملف بايثون واحد، مع اسم التمرين والهدف والسبب (أو الوصف للملفات العامة).

---

## 1. تمارين حسب الرياضة (`sports/`)

### football (كرة القدم)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/football/clam_shells.py` | Clam shells | hip | Glute strength for kick control |
| `sports/football/single_leg_balance.py` | Single-leg balance | ankle | Plant foot stability |
| `sports/football/hip_flexor_stretch.py` | Hip flexor stretch | hip | Range of motion for kicking |

### basketball (كرة السلة)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/basketball/squat_jump.py` | Squat jump | knee | Jump power and landing |
| `sports/basketball/single_leg_stance.py` | Single-leg stance | ankle | Agility stability |
| `sports/basketball/wall_sit.py` | Wall sit | knee | Knee protection on landing |

### tennis (التنس)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/tennis/band_pull_apart.py` | Band pull-apart | shoulder | Posterior shoulder strength |
| `sports/tennis/seated_torso_twist.py` | Seated torso twist | core | Rotation for strokes |
| `sports/tennis/rotator_cuff_exercises.py` | Rotator cuff exercises | shoulder | Injury prevention |

### gymnastics (الجمباز)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/gymnastics/hollow_hold.py` | Hollow hold | core | Core control |
| `sports/gymnastics/shoulder_mobility.py` | Shoulder mobility | shoulder | Overhead positions |
| `sports/gymnastics/single_leg_balance.py` | Single-leg balance | ankle | Beam stability |

### martial_arts (الفنون القتالية)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/martial_arts/band_pull_apart.py` | Band pull-apart | shoulder | Shoulder balance |
| `sports/martial_arts/hip_flexor_stretch.py` | Hip flexor stretch | hip | Kick range |

### boxing (الملاكمة)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/boxing/band_pull_apart.py` | Band pull-apart | shoulder | Shoulder balance |
| `sports/boxing/push_ups.py` | Push-ups | core | Push power and guard |
| `sports/boxing/torso_twist.py` | Torso twist | core | Rotation for power |

### weightlifting (رفع الأثقال)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/weightlifting/wall_sit.py` | Wall sit | knee | Knee tracking |
| `sports/weightlifting/hip_bridge.py` | Hip bridge | hip | Hip drive for squat |
| `sports/weightlifting/plank.py` | Plank | core | Core bracing |

### running (الجري)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/running/hip_flexor_stretch.py` | Hip flexor stretch | hip | Range for stride |
| `sports/running/calf_raises.py` | Calf raises | ankle | Ankle power |
| `sports/running/glute_bridges.py` | Glute bridges | hip | Hip drive |

### yoga (اليوغا)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/yoga/cat_cow.py` | Cat-cow | spine | Spinal mobility |
| `sports/yoga/downward_dog.py` | Downward dog | shoulder | Shoulder and hamstring |
| `sports/yoga/tree_pose.py` | Tree pose | balance | Balance and focus |

### golf (الجولف)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/golf/seated_torso_twist.py` | Seated torso twist | core | Rotation for swing |
| `sports/golf/hip_rotation_drill.py` | Hip rotation drill | hip | Power from hip |

### baseball (البيسبول)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/baseball/rotator_cuff.py` | Rotator cuff | shoulder | Throw injury prevention |
| `sports/baseball/core_rotation.py` | Core rotation | core | Bat and throw power |

### volleyball (الكرة الطائرة)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/volleyball/squat_jump.py` | Squat jump | knee | Jump power |
| `sports/volleyball/single_leg_landing.py` | Single-leg landing | ankle | Safe landing |

### hockey (الهوكي)
| الملف | التمارين |
|-------|----------|
| `sports/hockey/__init__.py` | لا تمارين (قائمة فارغة) |

### swimming (السباحة)
| الملف | التمارين |
|-------|----------|
| `sports/swimming/__init__.py` | لا تمارين (قائمة فارغة) |

### general_fitness (اللياقة العامة)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/general_fitness/plank.py` | Plank | core | Core stability |
| `sports/general_fitness/squats.py` | Squats | leg | Leg strength |
| `sports/general_fitness/glute_bridge.py` | Glute bridge | hip | Hip stability |

### unknown (عام)
| الملف | التمرين | الهدف | السبب |
|-------|---------|-------|--------|
| `sports/unknown/plank.py` | Plank | core | Core stability |
| `sports/unknown/squats.py` | Squats | leg | Leg strength |

---

## 2. تمارين عامة حسب المفصل (`generic/`)

تُستخدم عندما لا يوجد تمرين رياضة محدد للمفصل. المصدر: NASM Corrective Exercise.

| الملف | التمرين | المفصل | الوصف | الجرعات |
|-------|---------|--------|--------|---------|
| `generic/wall_sit.py` | Wall sit | knee | Wall sit for quad strength | 3x30s |
| `generic/clam_shells.py` | Clam shells | knee | Hip rotation for glutes | 3x15 each |
| `generic/band_pull_apart.py` | Band pull-apart | shoulder | Posterior shoulder strength | 3x15 |
| `generic/hip_bridge.py` | Hip bridge | hip | Glute strength | 3x15 |
| `generic/calf_raises.py` | Calf raises | ankle | Ankle stability | 3x15 |
| `generic/plank.py` | Plank | core | Core stability | 3x30s |

---

## 3. الربط من الكود

- **رياضة معينة:**  
  `get_exercises_for_sport("football")` ← ترجع قائمة تمارين كرة القدم من الملفات أعلاه.

- **كل تمارين رياضة:**  
  `SPORT_EXERCISES["basketball"]` ← قائمة التمارين من `sports/basketball/*.py`.

- **تمارين عامة (مفصل):**  
  `GENERIC_EXERCISES_RAW["knee"]` ← تمارين الركبة من `generic/wall_sit.py` و `generic/clam_shells.py`.

المسار من جذر المشروع: `backend/analysis/exercises/` (وهذا الملف فيه: `EXERCISES_INDEX.md`).

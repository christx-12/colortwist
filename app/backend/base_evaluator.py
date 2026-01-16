#für skalierung 

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

@dataclass
class ExerciseConfig:
    name: str
    min_angle: float          # Ziel-Winkel (tiefster Punkt)
    max_angle_top: float      # Start-Winkel (oben)
    min_depth_ratio: float
    min_good_reps_ratio: float = 0.6

@dataclass
class RepResult:
    rep_id: int
    is_good: bool
    score: float
    start_time: float
    end_time: float

class BaseExerciseEvaluator:
    def __init__(self, config: ExerciseConfig):
        self.config = config
        self._in_rep = False
        self._rep_id = 0
        self.rep_history: List[RepResult] = []
        self._reset_temp_metrics()

    def _reset_temp_metrics(self):
        self._rep_start_time = 0.0
        self._min_angle = 999.0
        self._start_y = None
        self._min_y = None

    def _angle(self, a, b, c):
        bax, bay = a[0]-b[0], a[1]-b[1]
        bcx, bcy = c[0]-b[0], c[1]-b[1]
        dot = bax*bcx + bay*bcy
        mag_ba = math.sqrt(bax**2 + bay**2)
        mag_bc = math.sqrt(bcx**2 + bcy**2)
        if mag_ba == 0 or mag_bc == 0: return 0.0
        return math.degrees(math.acos(max(min(dot/(mag_ba*mag_bc), 1.0), -1.0)))

    def evaluate_frame(self, keypoints: Dict, timestamp: float = None) -> Dict:
        if timestamp is None: timestamp = time.time()
        metrics = self.get_exercise_metrics(keypoints)
        
        if not metrics:
            return {"state": "searching", "angle": None, "rep_finished": False}

        angle, current_y = metrics["angle"], metrics["y"]
        if self._start_y is None: self._start_y = current_y
        self._min_angle = min(self._min_angle, angle)
        self._min_y = min(self._min_y, current_y) if self._min_y is not None else current_y

        at_top = angle > self.config.max_angle_top - 10
        at_bottom = angle < self.config.min_angle + 20

        rep_finished = False
        last_rep = None

        if not self._in_rep and at_top:
            self._in_rep = True
            self._rep_start_time = timestamp
            self._min_angle = angle
            self._start_y = current_y

        elif self._in_rep and at_top and (timestamp - self._rep_start_time) > 0.5:
            # Erfolgskriterien
            depth_ratio = abs(self._start_y - self._min_y) / max(self._start_y, 1e-6)
            depth_ok = depth_ratio >= self.config.min_depth_ratio
            angle_ok = self._min_angle <= self.config.min_angle
            
            score = (float(depth_ok) + float(angle_ok)) / 2
            self._rep_id += 1
            last_rep = RepResult(self._rep_id, score >= self.config.min_good_reps_ratio, score, self._rep_start_time, timestamp)
            self.rep_history.append(last_rep)
            
            self._in_rep = False
            self._reset_temp_metrics()
            rep_finished = True

        return {"state": "in_rep" if self._in_rep else "idle", "angle": angle, "rep_finished": rep_finished, "last_rep": last_rep}

    def get_exercise_metrics(self, keypoints: Dict):
        raise NotImplementedError("Subclasses must implement this")

# --- Spezifische Übungen ---

class SquatEvaluator(BaseExerciseEvaluator):
    def get_exercise_metrics(self, keypoints):
        h, k, a = keypoints.get("hip"), keypoints.get("knee"), keypoints.get("ankle")
        if not (h and k and a): return None
        return {"angle": self._angle(h, k, a), "y": h[1]}

class PushupEvaluator(BaseExerciseEvaluator):
    def get_exercise_metrics(self, keypoints):
        s, e, w = keypoints.get("shoulder"), keypoints.get("elbow"), keypoints.get("wrist")
        if not (s and e and w): return None
        return {"angle": self._angle(s, e, w), "y": s[1]}
    

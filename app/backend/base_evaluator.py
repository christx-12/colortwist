# app/backend/base_evaluator.py

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import time



@dataclass
class ExerciseConfig:
    name: str = "Squat"
    min_angle: float = 80.0           # Tiefe: kleiner = tiefer
    max_angle_top: float = 170.0      # fast durchgestreckt
    min_depth_ratio: float = 0.15     # minimaler vertikaler Weg
    min_good_reps_ratio: float = 0.6  # ab diesem Anteil gilt Rep als "gut"
    min_rep_duration: float = 0.4     # Sekunden
    max_rep_duration: float = 5.0     # Sekunden

##Container, der das Ergebnis einer einzigen Wiederholung speichert
@dataclass
class RepResult:
    rep_id: int
    is_good: bool
    score: float
    reason: Optional[str]
    start_time: float
    end_time: float

#abstrakte klasse mir Wissen über Bewegungsabläufe, aber kein Wissen über spezifische Körperteile.
class BaseEvaluator:
    """Basis-Klasse mit gemeinsamer Logik für alle Übungen."""
    
    #interne Status initialisiert.
    def __init__(self, config: ExerciseConfig):
        self.config = config
        
        # Tracking-Status
        self._in_rep: bool = False #ist user in wegegung
        self._rep_start_time: float = 0.0
        self._rep_id: int = 0
        
        # Pro-Rep-Metriken
        self._max_angle: float = 0.0
        self._min_angle: float = 999.0
        self._start_ref_y: Optional[float] = None
        self._min_ref_y: Optional[float] = None
        
        # Anti-Overcounting Features
        self._prev_angle: Optional[float] = None
        self._prev_ref_y: Optional[float] = None
        self._top_frames_counter: int = 0
        self._bottom_visited: bool = False  # Wichtig: muss unten gewesen sein
        
        # Ergebnisse
        self.rep_history: List[RepResult] = []

    #mathematische kerfunktion: 
    # Logik: Sie berechnet zwei Vektoren und nutzt das Skalarprodukt, um den Winkel dazwischen zu bestimmen.
    @staticmethod
    def _angle(a: Tuple[float, float],
               b: Tuple[float, float],
               c: Tuple[float, float]) -> float:
        """Winkel ABC in Grad (b = Scheitelpunkt)."""
        bax, bay = a[0] - b[0], a[1] - b[1]
        bcx, bcy = c[0] - b[0], c[1] - b[1]
        
        dot = bax * bcx + bay * bcy
        #normailsierung = Winkel unabhängig von der Bildgröße oder Entfernung zur Kamera.
        mag_ba = math.sqrt(bax ** 2 + bay ** 2)
        mag_bc = math.sqrt(bcx ** 2 + bcy ** 2)
        
        if mag_ba == 0 or mag_bc == 0:
            return 0.0
        
        cos_angle = max(min(dot / (mag_ba * mag_bc), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def _get_relevant_points(self, keypoints: Dict) -> Tuple[Optional[Tuple], ...]:
        """Override in Subclass: Gibt (point_a, point_b, point_c, ref_y_point) zurück."""
        raise NotImplementedError

    # pro Video-Frame
    def evaluate_frame(self, keypoints: Dict, timestamp: Optional[float] = None) -> Dict:
        if timestamp is None:
            timestamp = time.time()
        
        points = self._get_relevant_points(keypoints)
        if None in points[:3]:  # Mindestens 3 Punkte für Winkel nötig
            self._prev_angle = None
            self._prev_ref_y = None
            self._top_frames_counter = 0
            return {
                "state": "no_detection",
                "angle": None,
                "rep_started": False,
                "rep_finished": False,
                "last_rep": None,
            }
        
        pt_a, pt_b, pt_c, ref_point = points
        angle = self._angle(pt_a, pt_b, pt_c)
        ref_y = ref_point[1] if ref_point else pt_b[1]
        
        # Velocity & Movement Detection
        moving_down = False
        if self._prev_angle is not None and self._prev_ref_y is not None:
            angle_delta = angle - self._prev_angle
            ref_y_delta = ref_y - self._prev_ref_y
            # Winkel wird kleiner (beugen) UND Referenzpunkt sinkt
            moving_down = (angle_delta < -3) and (ref_y_delta > 3)
        
        self._prev_angle = angle
        self._prev_ref_y = ref_y
        
        # Referenzhöhe initialisieren
        if self._start_ref_y is None:
            self._start_ref_y = ref_y
        
        # Pro-Rep Metriken updaten
        self._max_angle = max(self._max_angle, angle)
        self._min_angle = min(self._min_angle, angle)
        if self._min_ref_y is None:
            self._min_ref_y = ref_y
        else:
            self._min_ref_y = min(self._min_ref_y, ref_y)
        
        rep_started = False
        rep_finished = False
        last_rep: Optional[RepResult] = None
        
        # Zustandslogik mit Hysteresis
        threshold_top = self.config.max_angle_top - 10
        threshold_bottom = self.config.min_angle + 20
        
        at_top_raw = angle > threshold_top
        at_bottom = angle < threshold_bottom
        
        # Hysteresis: 3 Frames für stabiles at_top
        if at_top_raw:
            self._top_frames_counter += 1
        else:
            self._top_frames_counter = 0
        
        at_top = self._top_frames_counter >= 3
        
        # Bottom-Check für echte Reps
        if at_bottom:
            self._bottom_visited = True
        
        # Rep-Start: Nur bei bestätigtem Top + Abwärtsbewegung
        if not self._in_rep and at_top and moving_down:
            self._in_rep = True
            self._rep_start_time = timestamp
            self._max_angle = angle
            self._min_angle = angle
            self._start_ref_y = ref_y
            self._min_ref_y = ref_y
            self._top_frames_counter = 0
            self._bottom_visited = False
            rep_started = True
        
        # Rep-Ende: Top erreicht + genug Bewegung + war unten
        if self._in_rep:
            if self._min_ref_y is not None and ref_y < self._min_ref_y:
                self._min_ref_y = ref_y
            
            min_angle_change = 40  # Mindestens 40° Änderung
            time_elapsed = timestamp - self._rep_start_time
            angle_change = self._max_angle - self._min_angle
            
            if (at_top and 
                time_elapsed > 0.3 and 
                angle_change > min_angle_change and
                self._bottom_visited):
                
                rep_duration = time_elapsed
                depth = (self._start_ref_y - self._min_ref_y) if self._start_ref_y and self._min_ref_y else 0.0
                depth_ratio = depth / max(self._start_ref_y, 1e-6) if self._start_ref_y else 0.0
                
                # Scoring
                depth_ok = depth_ratio >= self.config.min_depth_ratio
                angle_ok = self._min_angle <= self.config.min_angle
                duration_ok = self.config.min_rep_duration <= rep_duration <= self.config.max_rep_duration
                
                score = sum([depth_ok, angle_ok, duration_ok]) / 3
                is_good = score >= self.config.min_good_reps_ratio
                
                reason = None
                if not depth_ok:
                    reason = "Nicht tief genug"
                elif not angle_ok:
                    reason = "Winkel zu groß"
                elif not duration_ok:
                    reason = "Zu schnell" if rep_duration < self.config.min_rep_duration else "Zu langsam"
                
                self._rep_id += 1
                last_rep = RepResult(
                    rep_id=self._rep_id,
                    is_good=is_good,
                    score=score,
                    reason=reason,
                    start_time=self._rep_start_time,
                    end_time=timestamp,
                )
                self.rep_history.append(last_rep)
                
                # Reset
                self._in_rep = False
                self._rep_start_time = 0.0
                self._max_angle = 0.0
                self._min_angle = 999.0
                self._start_ref_y = None
                self._min_ref_y = None
                self._bottom_visited = False
                rep_finished = True
        
        return {
            "state": "in_rep" if self._in_rep else "idle",
            "angle": angle,
            "rep_started": rep_started,
            "rep_finished": rep_finished,
            "last_rep": last_rep,
        }

    def summary(self) -> Dict:
        total = len(self.rep_history)
        good = sum(1 for r in self.rep_history if r.is_good)
        return {
            "total_reps": total,
            "good_reps": good,
            "bad_reps": total - good,
            "avg_score": round(sum(r.score for r in self.rep_history) / total, 2) if total else 0.0,
        }

    def reset(self):
        self.__init__(self.config)


class SquatEvaluator(BaseEvaluator):
    """Squat: Winkel = hip-knee-ankle, Referenz = hip_y"""
    
    def _get_relevant_points(self, keypoints: Dict):
        hip = keypoints.get("hip")
        knee = keypoints.get("knee")
        ankle = keypoints.get("ankle")
        return (hip, knee, ankle, hip)


class PushupEvaluator(BaseEvaluator):
    """Pushup: Winkel = shoulder-elbow-wrist, Referenz = shoulder_y"""
    
    def _get_relevant_points(self, keypoints: Dict):
        shoulder = keypoints.get("shoulder")
        elbow = keypoints.get("elbow")
        wrist = keypoints.get("wrist")
        return (shoulder, elbow, wrist, shoulder)

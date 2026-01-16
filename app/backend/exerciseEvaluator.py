from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import time

@dataclass
class ExerciseConfig:
    name: str = "squat"
    min_knee_angle: float = 80.0       # Tiefe: kleiner = tiefer Squat
    max_knee_angle_top: float = 170.0  # fast durchgestreckt im Stand
    min_depth_ratio: float = 0.15      # minimaler vertikaler Weg Hüfte
    min_good_reps_ratio: float = 0.6   # ab diesem Anteil gilt Rep als "gut"
    min_rep_duration: float = 0.4      # Sekunden (Filter gegen Noise)
    max_rep_duration: float = 5.0      # Sekunden

@dataclass
class RepResult:
    rep_id: int
    is_good: bool
    score: float             # 0–1
    reason: Optional[str]
    start_time: float
    end_time: float

class ExerciseEvaluator:
    """
    Bewertung von Squats (oder ähnlichen Lower-Body-Übungen) anhand von Keypoints.
    Erwartet 2D-Keypoints (x, y) im Bildkoordinatensystem, z.B. YOLO-Pose oder MediaPipe.
    """
    
    def __init__(self, config: ExerciseConfig = ExerciseConfig()):
        self.config = config
        
        # Tracking-Status
        self._in_rep: bool = False
        self._rep_start_time: float = 0.0
        self._rep_bottom_time: float = 0.0
        self._rep_id: int = 0
        
        # Pro-Rep-Metriken
        self._max_knee_angle: float = 0.0
        self._min_knee_angle: float = 999.0
        self._start_hip_y: Optional[float] = None
        self._min_hip_y: Optional[float] = None
        
        # Velocity-Tracking (neue Anti-Overcounting-Features)
        self._prev_knee_angle: Optional[float] = None
        self._prev_hip_y: Optional[float] = None
        self._top_frames_counter: int = 0  # Hysteresis
        
        # Ergebnisse
        self.rep_history: List[RepResult] = []

    # --------------- Hilfsfunktionen ---------------
    
    @staticmethod
    def _angle(a: Tuple[float, float],
               b: Tuple[float, float],
               c: Tuple[float, float]) -> float:
        """
        Winkel ABC in Grad berechnen.
        a, b, c sind (x, y) Koordinaten, wobei b der Scheitelpunkt ist.
        """
        bax = a[0] - b[0]
        bay = a[1] - b[1]
        bcx = c[0] - b[0]
        bcy = c[1] - b[1]
        
        dot = bax * bcx + bay * bcy
        mag_ba = math.sqrt(bax ** 2 + bay ** 2)
        mag_bc = math.sqrt(bcx ** 2 + bcy ** 2)
        
        if mag_ba == 0 or mag_bc == 0:
            return 0.0
            
        cos_angle = max(min(dot / (mag_ba * mag_bc), 1.0), -1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    
    @staticmethod
    def _get_point(keypoints: Dict[str, Tuple[float, float]],
                   name: str) -> Optional[Tuple[float, float]]:
        return keypoints.get(name)

    # --------------- API: Frame-weise Bewertung ---------------
    
    def evaluate_frame(self,
                       keypoints: Dict[str, Tuple[float, float]],
                       timestamp: Optional[float] = None) -> Dict:
        """
        Wird pro Frame aufgerufen.
        keypoints: dict mit mindestens: 'hip', 'knee', 'ankle' (z.B. linke Seite).
        timestamp: Sekunden (time.time()), falls None → now.
        """
        if timestamp is None:
            timestamp = time.time()
        
        hip = self._get_point(keypoints, "hip")
        knee = self._get_point(keypoints, "knee")
        ankle = self._get_point(keypoints, "ankle")
        
        if not hip or not knee or not ankle:
            # Reset velocity tracking bei no detection
            self._prev_knee_angle = None
            self._prev_hip_y = None
            self._top_frames_counter = 0
            return {
                "state": "no_detection",
                "current_knee_angle": None,
                "current_hip_y": None,
                "rep_started": False,
                "rep_finished": False,
                "last_rep": None,
            }
        
        # Knie-Winkel berechnen: hip - knee - ankle
        knee_angle = self._angle(hip, knee, ankle)
        hip_y = hip[1]  # y im Bild: größer = tiefer im Bild
        
        # Velocity & Movement Detection (FIX für Overcounting)
        moving_down = False
        if self._prev_knee_angle is not None and self._prev_hip_y is not None:
            angle_delta = knee_angle - self._prev_knee_angle
            hip_delta = hip_y - self._prev_hip_y
            moving_down = (angle_delta < -5) and (hip_delta > 5)  # Knie beugen + Hüfte sinkt
        
        self._prev_knee_angle = knee_angle
        self._prev_hip_y = hip_y
        
        # Start-/Referenzhöhe initialisieren
        if self._start_hip_y is None:
            self._start_hip_y = hip_y
        
        # Update pro Rep Metriken
        self._max_knee_angle = max(self._max_knee_angle, knee_angle)
        self._min_knee_angle = min(self._min_knee_angle, knee_angle)
        if self._min_hip_y is None:
            self._min_hip_y = hip_y
        else:
            self._min_hip_y = min(self._min_hip_y, hip_y)
        
        rep_started = False
        rep_finished = False
        last_rep: Optional[RepResult] = None
        
        # Verbesserte Zustandslogik mit Hysteresis & Velocity
        knee_threshold_top = self.config.max_knee_angle_top - 10  # 160°
        at_top_raw = knee_angle > knee_threshold_top
        
        # Hysteresis: 3 aufeinanderfolgende top-Frames nötig
        if at_top_raw:
            self._top_frames_counter += 1
        else:
            self._top_frames_counter = 0
        
        at_top = self._top_frames_counter >= 3
        
        at_bottom = knee_angle < self.config.min_knee_angle + 20
        
        if not self._in_rep and at_top and moving_down:
            # Neuer Rep beginnt NUR bei bestätigtem Abwärtstrend
            self._in_rep = True
            self._rep_start_time = timestamp
            self._rep_bottom_time = timestamp
            self._max_knee_angle = knee_angle
            self._min_knee_angle = knee_angle
            self._start_hip_y = hip_y
            self._min_hip_y = hip_y
            self._top_frames_counter = 0  # Reset
            rep_started = True
        
        if self._in_rep:
            # Tiefster Punkt tracken
            if hip_y < self._min_hip_y:
                self._min_hip_y = hip_y
                self._rep_bottom_time = timestamp
            
            # Rep endet NUR bei voller Rückkehr + ausreichender Bewegung
            if (at_top and 
                (timestamp - self._rep_start_time) > 0.3 and  # Etwas längere Mindestdauer
                (self._max_knee_angle - self._min_knee_angle) > 40):  # Signifikante Bewegung
        
                rep_duration = timestamp - self._rep_start_time
                depth = (self._start_hip_y - self._min_hip_y) if self._start_hip_y is not None else 0.0
                depth_ratio = depth / max(self._start_hip_y, 1e-6)
                
                # Score anhand Knie-Winkel & Tiefe
                depth_ok = depth_ratio >= self.config.min_depth_ratio
                angle_ok = self._min_knee_angle <= self.config.min_knee_angle
                
                # Zeitfilter
                duration_ok = (self.config.min_rep_duration 
                              <= rep_duration 
                              <= self.config.max_rep_duration)
                
                score_components = [
                    1.0 if depth_ok else 0.0,
                    1.0 if angle_ok else 0.0,
                    1.0 if duration_ok else 0.0
                ]
                score = sum(score_components) / len(score_components)
                
                is_good = score >= self.config.min_good_reps_ratio
                
                reason = None
                if not depth_ok:
                    reason = "Not deep enough"
                elif not angle_ok:
                    reason = "Knee angle too large (not enough bend)"
                elif not duration_ok:
                    if rep_duration < self.config.min_rep_duration:
                        reason = "Too fast"
                    else:
                        reason = "Too slow"
                
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
                
                # Reset für nächste Wiederholung
                self._in_rep = False
                self._rep_start_time = 0.0
                self._rep_bottom_time = 0.0
                self._max_knee_angle = 0.0
                self._min_knee_angle = 999.0
                self._start_hip_y = None
                self._min_hip_y = None
                rep_finished = True
        
        return {
            "state": "in_rep" if self._in_rep else "idle",
            "current_knee_angle": knee_angle,
            "current_hip_y": hip_y,
            "rep_started": rep_started,
            "rep_finished": rep_finished,
            "last_rep": last_rep,
            "debug": {  # Für Testing
                "at_top_raw": at_top_raw,
                "at_top": at_top,
                "moving_down": moving_down,
                "top_frames": self._top_frames_counter,
                "angle_delta": knee_angle - (self._prev_knee_angle or 0)
            }
        }

    # --------------- Komfort-APIs ---------------
    
    def summary(self) -> Dict:
        """Gibt Gesamtstatistik zurück."""
        total = len(self.rep_history)
        good = sum(1 for r in self.rep_history if r.is_good)
        bad = total - good
        avg_score = sum(r.score for r in self.rep_history) / total if total > 0 else 0.0
        
        return {
            "total_reps": total,
            "good_reps": good,
            "bad_reps": bad,
            "avg_score": round(avg_score, 2),
            "history": self.rep_history,
        }
    
    def reset(self):
        """Komplett zurücksetzen."""
        self.__init__(self.config)

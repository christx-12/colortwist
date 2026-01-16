import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys
import os

# Pfad-Fix, damit er 'app' findet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import deiner spezifischen Pushup-Logik
try:
    from app.backend.pushupEvaluator import PushupEvaluator, ExerciseConfig
except ImportError:
    from backend.pushupEvaluator import PushupEvaluator, ExerciseConfig

class PushupEvaluatorApp:
    def __init__(self, model_path="yolo11n-pose.pt"):
        print(f"Lade YOLO-Modell: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialisierung mit Pushup-Konfiguration
        self.evaluator = PushupEvaluator(ExerciseConfig(
            name="pushup",
            min_knee_angle=90.0,      # Wird in deiner Klasse als Ellbogen-Winkel genutzt
            max_knee_angle_top=160.0,
            min_depth_ratio=0.1
        ))
        
        self.cap = cv2.VideoCapture(0)
        # Höhere Auflösung für bessere Erkennung
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def get_keypoints_from_yolo(self, results) -> dict:
        """Extrahiert Schulter, Ellbogen und Handgelenk."""
        if not results or len(results) == 0 or results[0].keypoints is None:
            return {}
        
        try:
            # Holen der xy-Koordinaten
            kpts = results[0].keypoints.xy
            if kpts.shape[1] < 11: # Nicht genug Keypoints gefunden
                return {}
            
            # Erste gefundene Person
            person_kpts = kpts[0].cpu().numpy()
            
            def safe_get(idx_l, idx_r):
                # Prüfen, ob Punkt existiert und nicht (0,0) ist
                if person_kpts[idx_l].any(): return tuple(person_kpts[idx_l])
                if person_kpts[idx_r].any(): return tuple(person_kpts[idx_r])
                return None

            # WICHTIG: Namen müssen exakt mit PushupEvaluator übereinstimmen!
            keypoint_dict = {
                "shoulder": safe_get(5, 6),
                "elbow": safe_get(7, 8),
                "wrist": safe_get(9, 10)
            }
            
            # Nur zurückgeben, wenn alle 3 Punkte da sind
            if all(keypoint_dict.values()):
                return keypoint_dict
        except Exception as e:
            print(f"Fehler bei Keypoints: {e}")
        
        return {}

    def draw_skeleton(self, frame, keypoints):
        """Zeichnet das Arm-Skelett."""
        if not keypoints:
            return frame
        
        # Punkte zeichnen
        pts = {}
        for name, coord in keypoints.items():
            pos = (int(coord[0]), int(coord[1]))
            pts[name] = pos
            cv2.circle(frame, pos, 10, (0, 255, 255), -1)
            cv2.putText(frame, name, (pos[0]+10, pos[1]), self.font, 0.5, (255,255,255), 1)

        # Linien zeichnen: Schulter -> Ellbogen -> Handgelenk
        if "shoulder" in pts and "elbow" in pts:
            cv2.line(frame, pts["shoulder"], pts["elbow"], (0, 255, 0), 4)
        if "elbow" in pts and "wrist" in pts:
            cv2.line(frame, pts["elbow"], pts["wrist"], (0, 255, 0), 4)
            
        return frame

    def run(self):
        print("Pushup-Analyse gestartet... (Drücke 'q' zum Beenden)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. YOLO Vorhersage
            results = self.model.predict(frame, verbose=False, conf=0.5)
            
            # 2. Keypoints extrahieren
            keypoints = self.get_keypoints_from_yolo(results)
            
            # 3. Evaluieren
            info = self.evaluator.evaluate_frame(keypoints, time.time())
            
            # 4. Zeichnen
            frame = self.draw_skeleton(frame, keypoints)
            
            # UI Overlay (wie beim Squat-Projekt)
            state = info.get("state", "searching")
            summary = self.evaluator.summary()
            
            # Schwarzer Hintergrund für Text
            cv2.rectangle(frame, (10, 10), (450, 150), (0,0,0), -1)
            cv2.putText(frame, f"MODUS: PUSHUP ({state})", (20, 40), self.font, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"REPS: {summary['total_reps']}", (20, 80), self.font, 1.2, (0, 255, 0), 3)
            
            if info.get("rep_finished"):
                last = info["last_rep"]
                color = (0, 255, 0) if last.is_good else (0, 0, 255)
                cv2.putText(frame, "GUTE REP!" if last.is_good else "ZU FLACH!", (20, 130), self.font, 0.8, color, 2)

            cv2.imshow("Pushup Evaluator", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PushupEvaluatorApp()
    app.run()
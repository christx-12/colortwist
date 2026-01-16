"""
exercise_evaluator_app.py - KOMPLETTE FEHLERSICHERE VERSION
Speichere als neue Datei und starte direkt!
"""

import cv2
import numpy as np
from ultralytics import YOLO
from app.backend.exerciseEvaluator import ExerciseEvaluator, ExerciseConfig
import time


class ExerciseEvaluatorApp:
    def __init__(self, model_path="yolo11n-pose.pt"):
        print(f"Lade YOLO-Modell: {model_path}")
        self.model = YOLO(model_path)
        self.evaluator = ExerciseEvaluator(ExerciseConfig(
            min_knee_angle=75.0,      # Grad für gute Tiefe
            min_depth_ratio=0.6,     # Hüftsenkung
            min_rep_duration=0.4,     # Min. Rep-Zeit
        ))
        
        # Kamera
        self.cap = cv2.VideoCapture(3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            raise Exception("Kamera nicht verfügbar!")
        
        # Styling
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_stats = False

    def get_keypoints_from_yolo(self, results) -> dict:
        """Sichere Keypoint-Extraktion aus YOLO."""
        if not results or len(results) == 0:
            return {}
        
        try:
            kpts_tensor = results[0].keypoints.xy
            if kpts_tensor is None or kpts_tensor.shape[0] == 0:
                return {}
            
            kpts = kpts_tensor[0].cpu().numpy()
            
            # COCO Keypoints (links bevorzugt, rechts Fallback)
            def safe_extract(idx_left, idx_right):
                if len(kpts) > idx_left and not np.isnan(kpts[idx_left]).any():
                    return tuple(kpts[idx_left])
                if len(kpts) > idx_right and not np.isnan(kpts[idx_right]).any():
                    return tuple(kpts[idx_right])
                return None
            
            hip = safe_extract(11, 8)    # left_hip, right_hip
            knee = safe_extract(13, 10)  # left_knee, right_knee
            ankle = safe_extract(15, 12) # left_ankle, right_ankle
            
            if hip and knee and ankle:
                return {"hip": hip, "knee": knee, "ankle": ankle}
                
        except Exception as e:
            print(f"Keypoint-Error: {e}")
        
        return {}

    def draw_pose_skeleton(self, frame, keypoints):
        """Robustes Skelett-Zeichnen."""
        if not keypoints:
            return frame
        
        # Punkte sammeln (int-Koordinaten)
        points = {}
        labels = []
        for name, pt in keypoints.items():
            if pt and len(pt) == 2:
                try:
                    x, y = int(pt[0]), int(pt[1])
                    points[name] = (x, y)
                    labels.append((name.upper(), x, y))
                except:
                    continue
        
        # Linien zeichnen (hip→knee→ankle)
        line_pairs = [("hip", "knee"), ("knee", "ankle")]
        for start, end in line_pairs:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (0, 255, 255), 4)
        
        # Punkte + Labels
        for label, x, y in labels:
            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
            cv2.putText(frame, label, (x+15, y-5), self.font, 0.5, (255, 255, 255), 1)
        
        return frame

    def create_stats_window(self, info, summary):
        """Stats-Dashboard."""
        h, w = 420, 380
        stats_img = np.zeros((h, w, 3), dtype=np.uint8)
        stats_img.fill(25)  # Dunkel
        
        y = 30
        cv2.rectangle(stats_img, (10, 10), (410, 370), (0, 200, 0), 2)
        
        cv2.putText(stats_img, "=== SQUAT ANALYSE ===", (20, y),
                   self.font, 0.8, (0, 255, 0), 2)
        y += 40
        
        # Live-Status
        state = info.get("state", "?")
        angle = info.get("current_knee_angle")
        angle_text = f"{angle:.1f}°" if angle else "N/A"
        
        cv2.putText(stats_img, f"Status: {state.upper()}", (20, y),
                   self.font, 0.7, (255, 255, 255), 2)
        y += 30
        cv2.putText(stats_img, f"Kniewinkel: {angle_text}", (20, y),
                   self.font, 0.7, (0, 255, 255), 2)
        y += 35
        
        # Letzte Rep
        if info.get("last_rep"):
            rep = info["last_rep"]
            color = (0, 255, 0) if rep.is_good else (0, 0, 255)
            status = "✅ GUT" if rep.is_good else "❌ SCHLECHT"
            cv2.putText(stats_img, f"Rep {rep.rep_id}: {status}", (20, y),
                       self.font, 0.8, color, 2)
            y += 25
            cv2.putText(stats_img, f"Score: {rep.score:.1%}", (20, y),
                       self.font, 0.7, color, 2)
            if rep.reason:
                cv2.putText(stats_img, f"Fehler: {rep.reason}", (20, y+25),
                           self.font, 0.5, (255, 165, 0), 1)
        
        # Gesamt-Stats
        y = 280
        cv2.putText(stats_img, "GESAMT:", (20, y), self.font, 0.7, (255, 255, 0), 2)
        y += 25
        cv2.putText(stats_img, f"✓ Gut: {summary['good_reps']}", (25, y),
                   self.font, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(stats_img, f"✗ Schlecht: {summary['bad_reps']}", (25, y),
                   self.font, 0.7, (0, 0, 255), 2)
        y += 25
        cv2.putText(stats_img, f"Durchschnitt: {summary['avg_score']:.1%}", (20, y),
                   self.font, 0.7, (255, 255, 0), 2)
        
        return stats_img

    def run(self):
        print("✅ Bereit! Steh vor Kamera & mach Squats!")
        print("'s'=Stats, 'r'=Reset, 'q'=Beenden")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera-Fehler!")
                break
            
            # YOLO
            results = self.model.predict(frame, verbose=False, conf=0.4)
            keypoints = self.get_keypoints_from_yolo(results)
            
            # Evaluieren
            info = {"state": "no_pose"}
            if keypoints:
                info = self.evaluator.evaluate_frame(keypoints, time.time())
            
            # Skelett
            frame = self.draw_pose_skeleton(frame, keypoints)
            
            # Haupt-Overlay
            state = info.get("state", "no_pose")
            angle = info.get("current_knee_angle")
            angle_text = f"{angle:.0f}°" if angle is not None else "N/A"
            
            # Status-Farbe
            if state == "idle":
                color = (50, 200, 50)
            elif state == "in_rep":
                color = (255, 255, 0)
            else:
                color = (100, 100, 100)
            
            cv2.rectangle(frame, (15, 15), (480, 140), color, -1)
            cv2.rectangle(frame, (15, 15), (480, 140), (255,255,255), 2)
            
            cv2.putText(frame, "🏋️ SQUAT TRAINER", (25, 45), self.font, 1.3, (0,0,0), 4)
            cv2.putText(frame, f"Status: {state.upper()}", (25, 75), self.font, 0.9, (255,255,255), 3)
            cv2.putText(frame, f"Winkel: {angle_text}", (25, 110), self.font, 0.9, (0,0,0), 3)
            
            # Live-Rep-Feedback
            if info.get("rep_finished") and info.get("last_rep"):
                rep = info["last_rep"]
                status = "✅ GUTE REP!" if rep.is_good else "❌ SCHLECHTE REP!"
                rep_color = (0, 255, 0) if rep.is_good else (0, 0, 255)
                cv2.putText(frame, status, (520, 80), self.font, 1.2, rep_color, 3)
                cv2.putText(frame, f"Score: {rep.score:.0%}", (520, 120), self.font, 1.0, rep_color, 3)
            
            # Pose-Status unten
            pose_ok = bool(keypoints)
            cv2.putText(frame, "POSE OK ✓" if pose_ok else "NO POSE ✗", 
                       (25, frame.shape[0]-40), self.font, 0.8, 
                       (0, 255, 0) if pose_ok else (0, 0, 255), 3)
            
            # Steuerung
            cv2.putText(frame, "'S'=Stats 'R'=Reset 'Q'=Quit", 
                       (25, frame.shape[0]-10), self.font, 0.6, (200,200,200), 2)
            
            cv2.imshow("Squat Trainer", frame)
            
            # Stats-Fenster
            if self.show_stats:
                stats_img = self.create_stats_window(info, self.evaluator.summary())
                cv2.imshow("📊 STATS", stats_img)
            
            # Tasten
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Stats: {'AN' if self.show_stats else 'AUS'}")
            elif key == ord('r'):
                self.evaluator.rep_history.clear()
                print("📊 Zurückgesetzt!")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        final = self.evaluator.summary()
        print("\n🏆 FINALE ERGEBNISSE:")
        print(f"  Gute Reps: {final['good_reps']}")
        print(f"  Schlechte: {final['bad_reps']}")
        print(f"  Durchschnitt: {final['avg_score']:.1%}")


if __name__ == "__main__":
    try:
        app = ExerciseEvaluatorApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Beendet.")
    except Exception as e:
        print(f"❌ Fehler: {e}")
       

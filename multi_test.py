# main.py (ExerciseApp)

import cv2
import numpy as np
from ultralytics import YOLO
from app.backend.base_evaluator import SquatEvaluator, PushupEvaluator, ExerciseConfig


class ExerciseApp:
    def __init__(self):
        print("Initialisiere Modell...")
        self.model = YOLO("yolo11n-pose.pt")
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.current_mode = "squat"
        self.set_mode("squat")

    def set_mode(self, mode):
        self.current_mode = mode
        if mode == "squat":
            config = ExerciseConfig(name="Squat", min_angle=85, max_angle_top=165, min_depth_ratio=0.1)
            self.evaluator = SquatEvaluator(config)
        else:
            config = ExerciseConfig(name="Pushup", min_angle=95, max_angle_top=155, min_depth_ratio=0.05)
            self.evaluator = PushupEvaluator(config)
        print(f"Modus gewechselt zu: {mode.upper()}")

    def get_all_keypoints(self, results):
        if not results or not results[0].keypoints or len(results[0].keypoints.xy) == 0:
            return {}
        
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        def safe_pt(idx_l, idx_r):
            if idx_l < len(kpts) and kpts[idx_l].any():
                return tuple(kpts[idx_l])
            if idx_r < len(kpts) and kpts[idx_r].any():
                return tuple(kpts[idx_r])
            return None

        return {
            "shoulder": safe_pt(5, 6),
            "elbow": safe_pt(7, 8),
            "wrist": safe_pt(9, 10),
            "hip": safe_pt(11, 12),
            "knee": safe_pt(13, 14),
            "ankle": safe_pt(15, 16)
        }

    def draw_exercise_skeleton(self, frame, keypoints):
        if self.current_mode == "squat":
            pairs = [("hip", "knee"), ("knee", "ankle")]
            color = (0, 255, 255)
        else:
            pairs = [("shoulder", "elbow"), ("elbow", "wrist")]
            color = (255, 255, 0)

        for start_key, end_key in pairs:
            pt1 = keypoints.get(start_key)
            pt2 = keypoints.get(end_key)
            
            if pt1 and pt2:
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))
                cv2.line(frame, p1, p2, color, 4)
                cv2.circle(frame, p1, 8, (255, 255, 255), -1)
                cv2.circle(frame, p2, 8, (255, 255, 255), -1)
                cv2.putText(frame, start_key, (p1[0]+10, p1[1]), self.font, 0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        print("Programm läuft. '1'=Squat, '2'=Pushup, 'r'=Reset, 'q'=Quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            results = self.model.predict(frame, verbose=False, conf=0.5)
            keypoints = self.get_all_keypoints(results)
            
            if keypoints:
                frame = self.draw_exercise_skeleton(frame, keypoints)
            
            info = self.evaluator.evaluate_frame(keypoints)
            
            # UI Overlay
            cv2.rectangle(frame, (10, 10), (420, 180), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (420, 180), (255, 255, 255), 1)
            
            cv2.putText(frame, f"AKTIV: {self.evaluator.config.name}", (20, 40), self.font, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"REPS: {len(self.evaluator.rep_history)}", (20, 90), self.font, 1.2, (0, 255, 0), 3)
            
            state = info.get("state", "searching")
            angle = info.get("angle")
            angle_str = f"{int(angle)} GRAD" if angle else "SUCHE..."
            
            # Farbcodierung für Status
            state_color = (0, 255, 255) if state == "in_rep" else (255, 255, 0)
            cv2.putText(frame, f"STATUS: {state.upper()} | {angle_str}", (20, 140), self.font, 0.6, state_color, 1)
            
            # Letzte Rep Feedback
            if info.get("rep_finished") and info.get("last_rep"):
                rep = info["last_rep"]
                feedback = "GUT!" if rep.is_good else f"({rep.reason})"
                fb_color = (0, 255, 0) if rep.is_good else (0, 0, 255)
                cv2.putText(frame, feedback, (20, 170), self.font, 0.6, fb_color, 2)

            cv2.imshow("Multi-Exercise Trainer", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.set_mode("squat")
            elif key == ord('2'):
                self.set_mode("pushup")
            elif key == ord('r'):
                self.evaluator.reset()
                print("Counter zurückgesetzt!")

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Endsummary
        summary = self.evaluator.summary()
        print(f"\n=== ZUSAMMENFASSUNG ===")
        print(f"Gesamt: {summary['total_reps']} | Gut: {summary['good_reps']} | Schlecht: {summary['bad_reps']}")


if __name__ == "__main__":
    ExerciseApp().run()

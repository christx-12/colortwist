from flask import Flask, render_template_string, Response, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import time
from backend.exercise_evaluator import ExerciseEvaluator, ExerciseConfig
 
app = Flask(__name__)
 
# Globale State (einfach)
evaluator = ExerciseEvaluator()
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)
 
 
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Squat Trainer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; background: #1a1a1a; color: white; font-family: -apple-system, Arial; }
        #container { display: flex; height: 100vh; }
        #video { flex: 2; background: black; }
        #video img { width: 100%; height: 100%; object-fit: cover; }
        #stats { flex: 1; padding: 20px; background: rgba(0,0,0,0.9); }
        #status, #angle, #reps { font-size: 24px; margin: 15px 0; }
        .good { color: #4CAF50; }
        .bad { color: #f44336; }
        .info { color: #ffeb3b; }
        button { background: #2196F3; color: white; border: none; padding: 12px 24px;
                 margin: 10px 5px; border-radius: 6px; cursor: pointer; font-size: 16px; }
        button:hover { background: #1976D2; }
    </style>
</head>
<body>
    <div id="container">
        <div id="video">
            <img id="stream" src="/video" alt="Live Stream">
        </div>
        <div id="stats">
            <h2>🏋️ SQUAT ANALYSE</h2>
            <div id="status" class="info">Warte auf Pose...</div>
            <div id="angle">Kniewinkel: --°</div>
            <div id="reps">Reps: 0/0 (0%)</div>
            <div id="last_rep"></div>
            <button onclick="reset()">🔄 Reset Stats</button>
            <button onclick="toggleFull()">📱 Vollbild</button>
        </div>
    </div>
 
    <script>
        let statsInterval;
       
        function updateStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').textContent = 'Status: ' + data.state;
                    document.getElementById('status').className = data.state === 'in_rep' ? 'info' :
                                                                 data.state === 'idle' ? 'good' : '';
                   
                    const angleEl = document.getElementById('angle');
                    angleEl.textContent = 'Kniewinkel: ' + (data.angle || '--') + '°';
                   
                    const repsEl = document.getElementById('reps');
                    repsEl.textContent = `Reps: ${data.good}/${data.total} (${data.avg}%)`;
                   
                    const lastRepEl = document.getElementById('last_rep');
                    if (data.last_rep) {
                        const status = data.last_rep.good ? '✅ GUT' : '❌ SCHLECHT';
                        const color = data.last_rep.good ? 'good' : 'bad';
                        lastRepEl.innerHTML = `<div class="${color}">Letzte Rep: ${status} (${data.last_rep.score}%)</div>`;
                    }
                });
        }
       
        // 10fps Updates
        statsInterval = setInterval(updateStats, 100);
       
        function reset() {
            fetch('/reset');
        }
       
        function toggleFull() {
            const vid = document.getElementById('stream');
            if (vid.requestFullscreen) vid.requestFullscreen();
        }
       
        // Cleanup
        window.onbeforeunload = () => clearInterval(statsInterval);
    </script>
</body>
</html>
'''
 
 
@app.route('/video')
def video():
    def gen_frames():
        global evaluator, model, cap
       
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
           
            # YOLO + Evaluator (schnell!)
            results = model.predict(frame, verbose=False, conf=0.4)
            keypoints = get_keypoints_from_yolo(results)
           
            info = {"state": "no_pose"}
            if keypoints:
                info = evaluator.evaluate_frame(keypoints, time.time())
           
            # Overlay
            frame = draw_overlay(frame, info, keypoints)
           
            # JPEG → Stream
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
   
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/api/stats')
def api_stats():
    summary = evaluator.summary()
    return jsonify({
        "state": "idle",  # Vereinfacht
        "angle": None,
        "good": summary["good_reps"],
        "total": summary["total_reps"],
        "avg": f"{summary['avg_score']:.0%}",
        "last_rep": summary["history"][-1] if summary["history"] else None
    })
 
 
@app.route('/reset')
def reset():
    evaluator.rep_history.clear()
    return "OK"
 
 
def get_keypoints_from_yolo(results):
    """Mini-Version aus vorher."""
    if not results or len(results) == 0:
        return {}
    try:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        hip = kpts[11] if len(kpts) > 11 else None
        knee = kpts[13] if len(kpts) > 13 else None  
        ankle = kpts[15] if len(kpts) > 15 else None
        if hip is not None and knee is not None and ankle is not None:
            return {"hip": tuple(hip), "knee": tuple(knee), "ankle": tuple(ankle)}
    except:
        pass
    return {}
 
 
def draw_overlay(frame, info, keypoints):
    """Minimales Overlay."""
    angle = info.get("current_knee_angle", 0)
   
    # Status-Box
    cv2.rectangle(frame, (10, 10), (300, 80), (50, 200, 50), -1)
    cv2.putText(frame, f"SQUAT: {angle:.0f}°", (20, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
   
    # Skelett
    if keypoints:
        pts = [keypoints["hip"], keypoints["knee"], keypoints["ankle"]]
        for i in range(len(pts)-1):
            p1 = (int(pts[i][0]), int(pts[i][1]))
            p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
            cv2.line(frame, p1, p2, (0, 255, 0), 3)
   
    return frame
 
 
if __name__ == '__main__':
    print("🚀 Squat Trainer WebApp startet...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        cap.release()




# # run.py (alles in einer Datei)
# from flask import Flask, render_template, request, jsonify
# from backend.pose_detector import PoseDetector

# app = Flask(__name__)
# detector = PoseDetector()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/pose', methods=['POST'])
# def detect_pose():
#     # YOLO Logic hier
#     return jsonify({'action': 'jump'})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
# run.py (alles in einer Datei)
from flask import Flask, render_template, request, jsonify
from backend.pose_detector import PoseDetector

app = Flask(__name__)
detector = PoseDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pose', methods=['POST'])
def detect_pose():
    # YOLO Logic hier
    return jsonify({'action': 'jump'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    

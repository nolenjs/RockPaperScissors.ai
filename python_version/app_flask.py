from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2 as cv
import copy
import csv
from collections import Counter, deque
import numpy as np
import mediapipe as mp
import base64
import json
import time
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from draw_landmarks import draw_landmarks
from hand_utils import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    pre_process_point_history,
    draw_bounding_rect,
    draw_info_text,
    draw_info,
    draw_point_history
)

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Load models
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize FPS calculator
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Initialize history
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# Map gestures to Rock, Paper, Scissors
gesture_map = {
    'Open': 'Paper',
    'Close': 'Rock',
    'Pointer': 'Scissors'
}

def process_frame():
    last_gesture_time = time.time()
    gesture_interval = 2  # Time interval in seconds between gesture recognitions
    gesture_countdown = ['Rock', 'Paper', 'Scissors', 'Shoot']  # Countdown sequence
    countdown_index = 0
    consistent_gesture = None

    while True:
        fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_sign = keypoint_classifier_labels[hand_sign_id]

                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], point_history_classifier_labels[most_common_fg_id[0][0]])

                current_time = time.time()
                if current_time - last_gesture_time >= gesture_interval:
                    if countdown_index < len(gesture_countdown):
                        gesture = gesture_countdown[countdown_index]
                        countdown_index += 1
                        last_gesture_time = current_time
                    else:
                        gesture = gesture_map.get(hand_sign, None)
                        if gesture:
                            consistent_gesture = gesture

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, 0, -1)

        # Encode the image as JPEG
        ret, jpeg = cv.imencode('.jpg', debug_image)
        frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        if consistent_gesture:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.encode() + b'\r\n\r\n' +
                   b'--frame\r\n'
                   b'Content-Type: application/json\r\n\r\n' + json.dumps({"gesture": consistent_gesture}).encode() + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.encode() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

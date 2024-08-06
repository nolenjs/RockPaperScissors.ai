from flask import Flask, render_template, Response
import cv2 as cv
import copy, time
from collections import deque, Counter
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from draw_landmarks import draw_landmarks
from app import (
    calc_bounding_rect,
    calc_landmark_list,
    draw_bounding_rect,
    draw_info
)
import os
from mediapipe.tasks.python import vision
from flask_socketio import SocketIO, emit
import threading
import random

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

# Load the custom model for gesture recognition
model_path = os.path.abspath("rock_paper_scissors.task")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

recognizer = vision.GestureRecognizer.create_from_model_path(model_path)

# Initialize FPS calculator
cvFpsCalc = CvFpsCalc(buffer_len=10)

last_gesture = "None"

def get_last_gesture():
    return last_gesture

# Main processing function
def process_frame():
    global last_gesture
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Run gesture recognition using the custom model
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                recognition_result = recognizer.recognize(mp_image)
                
                if recognition_result.gestures:
                    top_gesture = recognition_result.gestures[0][0]
                    last_gesture = top_gesture.category_name  # Update last_gesture
                    # print(f"Gesture recognized: {top_gesture.category_name} ({top_gesture.score})")
                    # Draw gesture result on the image
                    cv.putText(debug_image, f"Gesture: {top_gesture.category_name}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
        # else:
        #     point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, 0, -1)

        # Encode the image as JPEG
        ret, jpeg = cv.imencode('.jpg', debug_image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('start_game')
def start_game():
    gestures = ['rock', 'paper', 'scissors']
    for i in range(len(gestures)):
        emit('countdown', gestures[i])
        time.sleep(0.75)

    emit('countdown', 'Shoot!')
    time.sleep(0.25)

    success, frame = cap.read()
    last = get_last_gesture()
    if success:
        emit('gesture', f'{last}')
        computer_gesture = random.choice(gestures)
    
    # Determine the winner
    if last == computer_gesture:
        result = "It's a tie!"
    elif (last == 'rock' and computer_gesture == 'scissors') or \
         (last == 'paper' and computer_gesture == 'rock') or \
         (last == 'scissors' and computer_gesture == 'paper'):
        result = "You win!"
    else:
        result = "Computer wins!"

    # Emit the result
    socketio.emit('result', f'{result} (Computer chose {computer_gesture})')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
import cv2
import mediapipe as mp
import time
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class TimerState:
    def __init__(self):
        self.timer_running = False
        self.start_time = 0
        self.elapsed_time = 0
        self.timer_stopped = False
        self.previous_gesture = None
        self.last_gesture_time = 0
        self.gesture_duration_threshold = 0.2

# Function to initialize and configure the gesture recognizer
def initialize_recognizer():
    script_dir = os.getcwd()
    model_path = os.path.join(script_dir, 'gesture_recognizer.task')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    recognizer = vision.GestureRecognizer.create_from_options(options)
    return recognizer

# Function to process video and return elapsed time
def process_velocity(video_path, distance):
    timer_state = TimerState()
    recognizer = initialize_recognizer()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    
    elapsed_time = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Process frame to recognize gestures
            result = recognizer.recognize(mp_image)
            if result and hasattr(result, 'gestures') and result.gestures:
                gesture = result.gestures[0][0].category_name
                if gesture == "Open_Palm" and not timer_state.timer_running:
                    timer_state.timer_running = True
                    timer_state.start_time = time.time()

                elif gesture == "Closed_Fist" and timer_state.timer_running:
                    timer_state.timer_running = False
                    timer_state.elapsed_time = time.time() - timer_state.start_time
                    break
            elif timer_state.timer_running:
                timer_state.elapsed_time = time.time() - timer_state.start_time

        elapsed_time = timer_state.elapsed_time
    finally:
        cap.release()

    velocity = calculate_velocity(elapsed_time, distance)
    return elapsed_time, velocity

# Function to calculate velocity
def calculate_velocity(time, distance):
    return round(distance / time, 2) if time > 0 else 0

import cv2
import os
import mediapipe as mp
import numpy as np
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


BG_COLOR = (192, 192, 192) # gray
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
        cap = cv2.VideoCapture('video/rope.mp4')
        frame_count = 0

        while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                        break

                image_height, image_width, _ = image.shape
                start_time = time.time()
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                end_time = time.time()
                print(f'cost time {end_time-start_time}')
        cap.release()

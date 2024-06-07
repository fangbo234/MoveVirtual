import cv2
import mediapipe as mp
import numpy as np
import time


class PoseCheck:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    BG_COLOR = (192, 192, 192)
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks= True,
        smooth_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    def drawSkelon(self,image,results):
        if(results.face_landmarks):
            PoseCheck.mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                PoseCheck.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=PoseCheck.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        if(results.pose_landmarks):
            PoseCheck.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                PoseCheck.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=PoseCheck.mp_drawing_styles.
                get_default_pose_landmarks_style())


    def check(self, image):
        start_time = time.time()
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        # start_time = time.time()
        # results = holistic.process(image)
        # results = PoseCheck.holistic.process(image)
        results = PoseCheck.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print(results)
        # end_time = time.time()
        # print(f'pose check cost {end_time - start_time} seconds')
        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".

        #condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        #bg_image = np.zeros(image.shape, dtype=np.uint8)
        #bg_image[:] = PoseCheck.BG_COLOR
        #annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose, left and right hands, and face landmarks on the image.
        end_time = time.time()
        print(f'post check cost {end_time - start_time}')
        return results

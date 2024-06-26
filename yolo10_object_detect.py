import time

from ultralytics import YOLOv10
import supervision as sv
import cv2


class Yolov10ObjectCheck:
    MODEL_PATH = 'yolov/yolov10n.pt'
    #IMAGE_PATH = 'img/6.png'
    model = YOLOv10(MODEL_PATH)
    #image = cv2.imread(IMAGE_PATH)
    #results = model(source=image, conf=0.5, verbose=False)[0]
    #detections = sv.Detections.from_ultralytics(results)
    #box_annotator = sv.BoxAnnotator()
    category_dict = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    def check(self,image):
        start_time = time.time()
        results = Yolov10ObjectCheck.model(source=image, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        end_time = time.time()
        print(f'object check cost {end_time-start_time}')
        return detections.xyxy,detections.class_id

    # labels = [f"{category_dict[class_id]} {confidence:.2f}"for class_id, confidence in zip(detections.class_id, detections.confidence)]
    # annotated_image = box_annotator.annotate(image.copy(), detections=detections, labels=labels)
    # cv2.imshow('image',annotated_image)
    # cv2.imwrite('annotated_dog.jpeg', annotated_image)
    # cv2.waitKey(0)

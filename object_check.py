import cv2
import numpy as np

class ObjectCheck:
    net = cv2.dnn.readNet("yolov/yolov3.weights", "yolov/yolov3.cfg")

    def check(self,path):
        image = cv2.imread(path)
        # Get image dimensions
        (height, width) = image.shape[:2]
        # Define the neural network input
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        ObjectCheck.net.setInput(blob)

        # Perform forward propagation
        output_layer_name =  ObjectCheck.net.getUnconnectedOutLayersNames()
        output_layers =  ObjectCheck.net.forward(output_layer_name)
        boxes = []
        confidences = []
        classIDs = []
        images=[]
        # Loop over the output layers
        for output in output_layers:
            # Loop over the detections
            for detection in output:
                # Extract the class ID and confidence of the current detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only keep detections with a high confidence
                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    # Object detected
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(class_id)

                    # Add the detection to the list of people
                    # people.append((x, y, w, h))
        # print(f'get boxs {len(boxes)}')
        # Draw bounding boxes around the people
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        # print(f'after remove repeat boxs {len(idxs)}')
        if len(idxs) > 0:
            # 迭代每个边界框
            for i in idxs.flatten():
                # 提取边界框的坐标
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                images.append(image[y:y+h,x:x+w])
        return images

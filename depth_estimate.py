import time

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2


class DepthEstimate:

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large",local_files_only=True)
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    # img  cv2 Image
    def check(self,img):
        start_time = time.time()
        image =  Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # prepare image for the model
        inputs = DepthEstimate.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = DepthEstimate.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        numpy_image = np.array(depth)
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        end_time = time.time()
        print(f'depth estimate cost {end_time - start_time}')

        return cv2_image


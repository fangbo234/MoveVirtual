import cv2
from pose_check import PoseCheck
from yolo10_object_detect import Yolov10ObjectCheck
from depth_estimate import DepthEstimate
import webview
import numpy as np
import pyvista as pv


def create_window():
    webview.create_window("My First App", "http://localhost:5000", width=800, height=600)


def test():
    # objCheck = ObjectCheck()
    image = cv2.imread('img/5.png')
    obj_check = Yolov10ObjectCheck()
    pose_check = PoseCheck()
    depthEstimate = DepthEstimate()
    xyxy, class_id = obj_check.check(image)
    depth_image = depthEstimate.check(image)
    print(f'max of depth {np.max(depth_image)}')
    print(f'min of depth {np.min(depth_image)}')
    index = 0
    for box in xyxy:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        image_rect = image[y1:y2, x1:x2]
        image_rect_depth=depth_image[y1:y2, x1:x2]
        # print(np.mean(image_rect_depth))
        if (class_id[index] == 0):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        skelon = pose_check.check(image_rect)
        drawSkelon(skelon.pose_landmarks.landmark)
        bg_image = np.zeros(image_rect.shape, dtype=np.uint8)
        bg_image[:] = PoseCheck.BG_COLOR
        pose_check.drawSkelon(image_rect, skelon)
        index = index + 1
    cv2.imshow('obj_img', image)
    cv2.waitKey(0)
    # cv2.imshow('depth_img',depth_image)

def drawLine(plotter,lanmark_list,a,b):
    line1 = pv.Line(pointa=(lanmark_list[a].x, lanmark_list[a].y, lanmark_list[a].z/4),
                    pointb=(lanmark_list[b].x, lanmark_list[b].y, lanmark_list[b].z/4))
    plotter.add_mesh(line1, color='red', line_width=2)

def drawSkelon(lanmark_list):
    plotter = pv.Plotter()
    drawLine(plotter,lanmark_list,11,12)
    drawLine(plotter, lanmark_list, 12, 24)
    drawLine(plotter, lanmark_list, 24, 23)
    drawLine(plotter, lanmark_list, 11, 23)
    drawLine(plotter, lanmark_list, 24, 26)
    drawLine(plotter, lanmark_list, 26, 28)
    drawLine(plotter, lanmark_list, 28, 32)
    drawLine(plotter, lanmark_list, 32, 30)
    drawLine(plotter, lanmark_list, 28, 30)
    drawLine(plotter, lanmark_list, 23, 25)
    drawLine(plotter, lanmark_list, 25, 27)
    drawLine(plotter, lanmark_list, 27, 29)
    drawLine(plotter, lanmark_list, 29, 31)
    drawLine(plotter, lanmark_list, 27, 31)
    drawLine(plotter, lanmark_list, 12, 14)
    drawLine(plotter, lanmark_list, 14, 16)
    drawLine(plotter, lanmark_list, 16, 22)
    drawLine(plotter, lanmark_list, 16, 18)
    drawLine(plotter, lanmark_list, 18, 20)
    drawLine(plotter, lanmark_list, 16, 20)
    drawLine(plotter, lanmark_list, 11, 13)
    drawLine(plotter, lanmark_list, 13, 15)
    drawLine(plotter, lanmark_list, 15, 21)
    drawLine(plotter, lanmark_list, 15, 17)
    drawLine(plotter, lanmark_list, 17, 19)
    drawLine(plotter, lanmark_list, 19, 15)

    drawLine(plotter, lanmark_list, 9, 10)
    drawLine(plotter, lanmark_list, 0, 5)
    drawLine(plotter, lanmark_list, 5, 8)
    drawLine(plotter, lanmark_list, 0, 2)
    drawLine(plotter, lanmark_list, 2, 7)
    plotter.show_grid()
    plotter.show()
def test3d():
    # 创建两点之间的线
    line = pv.Line(pointa=(0, 0, 0), pointb=(1, 1, -1))
    # 创建一个简单的画布，并添加线
    plotter = pv.Plotter()
    plotter.add_mesh(line, color='red', line_width=5)
    plotter.show_grid()
    plotter.show()  # 显示图形


if __name__ == '__main__':
    # create_window()
    # webview.start()
    test()

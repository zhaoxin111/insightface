from mtcnn_detector import MtcnnDetector
import os
import cv2
import mxnet as mx
import numpy as np

ctx = mx.gpu(2)
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.6,0.7,0.8])

img = cv2.imread('/home/zhaoxin/workspace/face/insightface/sample-images/t1.jpg')
# TODO 检测结果有较大偏差！！！待解决
total_boxes, points = detector.detect_face(img, det_type=0)

for box in total_boxes:
    box = box.astype(np.int)
    color = (0,0,255)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
cv2.imwrite('/home/zhaoxin/workspace/face/insightface/sample-images/t1_mtcnn.jpg', img)
print()





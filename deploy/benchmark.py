import face_embedding
import argparse
import cv2
import numpy as np
import datetime

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-arcface-ms1m-refine-v2/model,0', help='path to load model.')
parser.add_argument('--gpu', default=6, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_embedding.FaceModel(args)
#img = cv2.imread('/raid5data/dplearn/lfw/Jude_Law/Jude_Law_0001.jpg')
img = cv2.imread('../sample-images/t1.jpg')

time_now = datetime.datetime.now()
for i in range(3000):
  f1 = model.get_feature(img)
time_now2 = datetime.datetime.now()
diff = time_now2 - time_now
print(diff.total_seconds()/3000)

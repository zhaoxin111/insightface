import mxnet as mx
import sklearn
from glob import glob
import numpy as np
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

device=0
model_path = '/home/zhaoxin/workspace/face/insightface/models/model,0'
image_shape = [112,112]
batch_size = 1

def cos(vector1,vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

vec = model_path.split(',')
assert len(vec) > 1
prefix = vec[0]
epoch = int(vec[1])
print('loading', prefix, epoch)
ctx = mx.gpu(device)

# load the model and ckpt
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
model.bind(data_shapes=[('data', (batch_size, 3, image_shape[0], image_shape[1]))])
model.set_params(arg_params, aux_params)

def get_embeddings(img_paths):
    embeddings = []
    files = glob(img_paths)
    files = sorted(files)
    print(files)
    for img_path in files:
        img = cv2.imread(img_path)
        img = cv2.resize(img,tuple(image_shape))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)

        data = mx.nd.array(img)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        # print(model.get_outputs()[0].shape)
        # print(model.get_outputs()[0])
        embedding = model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        # embedding = embedding.flatten()
        embeddings.append(embedding)
    return np.array(embeddings)

if __name__ == "__main__":
    img_paths = '/home/zhaoxin/workspace/face/insightface/RetinaFace/man/*.jpg'
    embeddings = get_embeddings(img_paths)
    dists = [[cos(e1,e2) for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists))
    # sim = cosine_similarity(embeddings)
    # print(sim)
    



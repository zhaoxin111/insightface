import mxnet as mx
import sklearn
from glob import glob
import numpy as np
import cv2
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

device=0
model_path = '/home/zhaoxin/workspace/face/insightface/models/model-r100-arcface-ms1m-refine-v2/model,0'
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
# sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
# all_layers = sym.get_internals()
# sym = all_layers['fc1_output']
# model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
# model.bind(data_shapes=[('data', (batch_size, 3, image_shape[0], image_shape[1]))])
# model.set_params(arg_params, aux_params)

def get_embeddings(files):
    embeddings = []
    
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

def sh():
    data = pd.read_excel('/home/zhaoxin/workspace/face/insightface/recognition/sim.xlsx')
    data = np.asarray(data)[:,1:]
    data = np.triu(data, 1)
    data = data[:31,31:]
    for line in range(len(data)):
        print(data[line,:].max(),line,np.argmax(data[line,:]))

if __name__ == "__main__":
    # img_paths = '/home/zhaoxin/workspace/face/insightface/RetinaFace/man/*.jpg'
    # img_paths = '/home/zhaoxin/workspace/face/insightface/recognition/test_imgs/*/*.jpg'
    # files = glob(img_paths)
    # files = sorted(files)
    # embeddings = get_embeddings(files)
    # dists = [[cos(e1,e2) for e2 in embeddings] for e1 in embeddings]
    # names = ['_'.join(img.split('/')[-2:]) for img in files]
    # df = pd.DataFrame(dists,columns=names,index=names)
    # print(df)

    # df.to_excel('/home/zhaoxin/workspace/face/insightface/recognition/sim.xlsx')
    sh()



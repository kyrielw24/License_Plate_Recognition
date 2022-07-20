# 载入模型
from src.keras_utils import load_model

from src.data_generator import DataGenerator
from src.sampler import augment_sample, labels2output_map

from tensorflow import keras

import numpy as np

from glob import glob
from os.path import isfile, splitext
import cv2

from src.keras_utils import save_model
import keras.optimizers

from src.loss import loss

# 请在下方修改模型路径
wpod_net_path = 'D:\\TEMP_Work\license_work\\alpr-unconstrained\\data\\lp-detector\\wpod-net_final_2'
wpod_net = load_model(wpod_net_path)

# 添加输入输出层


# 请在下方修改输入层维度
input_dim = 208

input_shape = (input_dim, input_dim, 3)

inputs = keras.layers.Input(shape=input_shape)
outputs = wpod_net(inputs)

output_shape = list(outputs.shape[1:])
output_dim = output_shape[1]
model_stride = input_dim / output_dim

assert input_dim % output_dim == 0, \
    'The output resolution must be divisible by the input resolution'

assert model_stride == 2 ** 4, \
    'Make sure your model generates a feature map with resolution ' \
    '16x smaller than the input'


# 标签类


class Shape():

    def __init__(self, pts=np.zeros((2, 0)), max_sides=4, text=''):
        self.pts = pts
        self.max_sides = max_sides
        self.text = text

    def isValid(self):
        return self.pts.shape[1] > 2

    def write(self, fp):
        fp.write('%d,' % self.pts.shape[1])
        ptsarray = self.pts.flatten()
        fp.write(''.join([('%f,' % value) for value in ptsarray]))
        fp.write('%s,' % self.text)
        fp.write('\n')

    def read(self, line):
        data = line.strip().split(',')
        ss = int(data[0])
        values = data[1:(ss * 2 + 1)]
        text = data[(ss * 2 + 1)] if len(data) >= (ss * 2 + 2) else ''
        self.pts = np.array([float(value) for value in values]).reshape((2, ss))
        self.text = text


def readShapes(path, obj_type=Shape):
    shapes = []
    with open(path) as fp:
        for line in fp:
            shape = obj_type()
            shape.read(line)
            shapes.append(shape)
    return shapes


def writeShapes(path, shapes):
    if len(shapes):
        with open(path, 'w') as fp:
            for shape in shapes:
                if shape.isValid():
                    shape.write(fp)


# 载入数据集


# 请在下方修改训练集路径
train_dir = 'D:\\TEMP_Work\\License_Plate_Recognition\\samples\\train'

extensions = ['jpg', 'jpeg', 'png']
Files = []
for ext in extensions:
    Files += glob('%s\*.%s' % (train_dir, ext))

Data = []
for file in Files:
    labfile = splitext(file)[0] + '.txt'
    if isfile(labfile):
        L = readShapes(labfile)
        I = cv2.imread(file)
        Data.append([I, L[0]])

print('%d images with labels found' % len(Data))


# 生成dataLoader


def process_data_item(data_item, dim, model_stride):
    XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim)
    YY = labels2output_map(llp, pts, dim, model_stride)
    return XX, YY


dg = DataGenerator(data=Data,
                   process_data_item_func=lambda x: process_data_item(x, input_dim, model_stride),
                   xshape=input_shape,
                   yshape=(output_shape[0], output_shape[1], output_shape[2] + 1),
                   nthreads=2,
                   pool_size=1000,
                   min_nsamples=100)
dg.start()

# 训练


# 在此处修改batch_size
batch_size = 16
iterations = 25
outdir = 'D:\\TEMP_Work\license_work\\alpr-unconstrained\\data\\lp-detector'

Xtrain = np.empty((batch_size, input_dim, input_dim, 3), dtype='single')
Ytrain = np.empty((batch_size, int(input_dim / model_stride), int(input_dim / model_stride), 2 * 4 + 1))

model_path_backup = '%s\%s_backup' % (outdir, 'wpod-net')
model_path_final = '%s\%s_final' % (outdir, 'wpod-net')

opt = keras.optimizers.Adam(learning_rate=0.001)
wpod_net.compile(opt, loss)

for it in range(iterations):

    print('Iter. %d (of %d)' % (it + 1, iterations))

    Xtrain, Ytrain = dg.get_batch(batch_size)
    train_loss = wpod_net.train_on_batch(Xtrain, Ytrain)

    print('\tLoss: %f' % train_loss)

    # Save model every 1000 iterations
    if (it + 1) % 1000 == 0:
        print('Saving model (%s)' % model_path_backup)
        save_model(wpod_net, model_path_backup)

print('Stopping data generator')
dg.stop()

print('Saving model (%s)' % model_path_final)
save_model(wpod_net, model_path_final)

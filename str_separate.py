# 导入 OpenCV 模块
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import tensorflow as tf
import h5py

PROVINCE_START = 1000
SZ = 32  # 训练图片长宽
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]

chi = ["川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津",
       "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕",
       "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏",
       "浙"]

enu = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z']

model = cv2.ml.SVM_load("lib/svm.dat")
modelchinese = cv2.ml.SVM_load("lib/svmchinese.dat")

resmodelchs = tf.keras.models.load_model("lib/resnet50_chs_no_noise.h5")
resmodelenu = tf.keras.models.load_model("lib/resnet50_enu_no_noise.h5")


# 构造LeNet5模型
class LeNet5(tf.keras.Model):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(120, activation='tanh')
        self.f2 = tf.keras.layers.Dense(84, activation='tanh')
        self.f3 = tf.keras.layers.Dense(31, activation='softmax')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


new_modelchs = LeNet5()
new_modelchs.load_weights('lib/lenet5_chs_weight')
new_modelenu = LeNet5()
new_modelenu.load_weights('lib/lenet5_ens_weight')


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyWindow(name)


def plt_show(image):
    b, g, r = cv2.split(image)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


def plt_show_g(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 分离车牌字符
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])

    return part_cards


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def separate_and_predict(IMAGE_PATH):
    # # 指定要加载的图片
    # IMAGE_PATH = "D:\\TEMP_Work\\license_work\\alpr-unconstrained\\samples\\input\\1_lp.png"

    # 读入原图片对象
    source_image = cv2.imread(IMAGE_PATH)
    # plt_show(source_image)

    # 高斯去噪、灰度化
    G_image = cv2.GaussianBlur(source_image, (3, 3), 0)
    gray = cv2.cvtColor(G_image, cv2.COLOR_BGR2GRAY)
    # plt_show_g(gray)

    # 二值化
    ret, gray_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    plt_show_g(gray_img)

    w = gray_img.shape[0]
    h = gray_img.shape[1]
    black_point = 0
    white_point = 0
    print(w, h)
    for i in range(w):
        for j in range(h):
            if gray_img[i][j] == 0:
                black_point = black_point + 1
            else:
                white_point = white_point + 1

    # 黑字转白字
    if white_point > black_point:
        for i in range(w):
            for j in range(h):
                gray_img[i][j] = 255 - gray_img[i][j]

    plt_show_g(gray_img)

    # 寻找波峰 波谷
    x_histogram = np.sum(gray_img, axis=1)
    x_min = np.min(x_histogram)
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    x_threshold = (x_min + x_average) / 2

    wave_peaks = find_waves(x_threshold, x_histogram)

    # 认为水平方向，最大的波峰为车牌区域
    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    gray_img = gray_img[wave[0]:wave[1]]
    no_img = gray[wave[0]:wave[1]]
    # plt_show_g(gray_img)
    # plt_show_g(no_img)

    # 查找垂直直方图波峰
    row_num, col_num = gray_img.shape[:2]
    # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
    gray_img = gray_img[1:row_num - 1]

    y_histogram = np.sum(gray_img, axis=0)
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
    wave_peaks = find_waves(y_threshold, y_histogram)

    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    max_wave_dis = wave[1] - wave[0]
    # 判断是否是左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
        wave_peaks.pop(0)

    # 组合分离汉字
    cur_dis = 0
    for wi, wave in enumerate(wave_peaks):
        if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
            break
        else:
            cur_dis += wave[1] - wave[0]
    if wi > 0:
        wave = (wave_peaks[0][0], wave_peaks[wi][1])
        wave_peaks = wave_peaks[wi + 1:]
        wave_peaks.insert(0, wave)
    point = wave_peaks[2]
    point_img = gray_img[:, point[0]:point[1]]
    if np.mean(point_img) < 255 / 5:
        wave_peaks.pop(2)

    part_cards = seperate_card(gray_img, wave_peaks)

    L = len(part_cards)
    predict_result = []
    predict_result1 = []
    predict_str = ""
    predict_str1 = ""
    pic = []

    for pi, part_card in enumerate(part_cards):

        # 可能是固定车牌的铆钉
        if np.mean(part_card) < 255 / 5:
            # print("a point")
            continue

        pic.append(part_card)
        part_card_old = part_card

        # w = abs(part_card.shape[1] - SZ) // 2
        #
        # part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # # plt_show_g(part_card)
        # part_card = cv2.resize(part_card, (32, 32), interpolation=cv2.INTER_AREA)
        #
        # # plt_show_g(part_card)
        # part_card = preprocess_hog([part_card])
        # plt_show_g(part_card)
        # test_predict = model_chs.predict(test_data)
        # test_predict = np.argmax(test_predict, axis=1)
        part_card_1 = cv2.resize(part_card, (28, 28))
        part_card_1 = np.expand_dims(part_card_1, axis=0)
        part_card_1 = np.expand_dims(part_card_1, axis=-1)
        part_card_1 = tf.cast(part_card_1, tf.float32)
        part_card = cv2.resize(part_card, (32, 32))
        part_card = cv2.cvtColor(part_card, cv2.COLOR_GRAY2RGB)
        part_card = np.expand_dims(part_card, axis=0)
        # print(part_card.shape)

        if pi == 0:
            # resp = resmodelchs.call(inputs=part_card, training=False, mask=None)
            resp = resmodelchs.predict(part_card)
            label = np.argmax(resp)
            charactor = chi[label]
            res = new_modelchs.predict(part_card_1)
            label2 = np.argmax(res)
            charactor1 = chi[label2]
            # resp = modelchinese.predict(part_card)
            # charactor = provinces[int(resp[1][0]) - PROVINCE_START]
        elif pi == 1:
            resp = resmodelenu.predict(part_card)
            labels = np.argsort(-resp)
            for i in range(10):
                if labels[0][i] > 9:
                    label = labels[0][i]
                    break
            charactor = enu[label]
            resp1 = new_modelenu.predict(part_card_1)
            labels1 = np.argsort(-resp1)
            for i in range(10):
                if labels1[0][i] > 9:
                    label1 = labels1[0][i]
                    break
            charactor1 = enu[label1]
        else:
            # resp = model.predict(part_card)
            # charactor = chr(int(resp[1][0]))
            resp = resmodelenu.predict(part_card)
            label = np.argmax(resp)
            charactor = enu[label]
            res = new_modelenu.predict(part_card_1)
            label2 = np.argmax(res)
            charactor1 = enu[label2]
        # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
        if charactor == "1" or pi == (L - 1):
            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                continue

        predict_result.append(charactor)
        predict_str = "".join(predict_result)

        predict_result1.append(charactor1)
        predict_str1 = "".join(predict_result1)

    for i in range(len(pic)):
        plt_show_g(pic[i])
    # plt_show_g(np.hstack(pic))

    return predict_str, predict_str1


# str, str1 = separate_and_predict("D:\\TEMP_Work\\license_work\\alpr-unconstrained\\samples\\input\\car1_lp.png")
# print(str)
# print(str1)

# 导入 OpenCV 模块
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm

PROVINCE_START = 1000
SZ = 20  # 训练图片长宽
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

model = cv2.ml.SVM_load("lib/svm.dat")
modelchinese = cv2.ml.SVM_load("lib/svmchinese.dat")


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
    plt_show(source_image)

    # 高斯去噪、灰度化
    G_image = cv2.GaussianBlur(source_image, (3, 3), 0)
    gray = cv2.cvtColor(G_image, cv2.COLOR_BGR2GRAY)
    # plt_show_g(gray)

    # 二值化
    ret, gray_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # plt_show_g(gray_img)

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
    plt_show_g(gray_img)
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

    predict_result = []
    predict_str = ""
    pic = []

    for pi, part_card in enumerate(part_cards):

        # 可能是固定车牌的铆钉
        if np.mean(part_card) < 255 / 5:
            # print("a point")
            continue

        pic.append(part_card)
        part_card_old = part_card

        w = abs(part_card.shape[1] - SZ) // 2

        part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # plt_show_g(part_card)
        part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
        # plt_show_g(part_card)
        part_card = preprocess_hog([part_card])
        if pi == 0:
            resp = modelchinese.predict(part_card)
            charactor = provinces[int(resp[1][0]) - PROVINCE_START]
        else:
            resp = model.predict(part_card)
            charactor = chr(int(resp[1][0]))
        # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
        if charactor == "1":
            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                continue

        predict_result.append(charactor)
        predict_str = "".join(predict_result)

    return predict_str


str = separate_and_predict("D:\\TEMP_Work\\license_work\\alpr-unconstrained\\samples\\input\\1_lp.png")
print(str)

import os
import random
import cv2
import shutil

from sklearn import naive_bayes


def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        print(filename)

        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        img = cv2.imread(path + filename)
        if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
            os.remove(os.path.join(path, filename))
            continue
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = txt_path + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


def txt_translate_1(path, txt_path):
    for filename in os.listdir(path):
        list1 = filename.split("-")  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        need_name = list1[3]
        xy = need_name.split("_", 4)  # 第二次分割，以下划线'_'做分割
        x_pos = []
        y_pos = []
        for i in range(4):
            x, y = xy[i].split("&",1)
            x_pos.append(int(x)/720)
            y_pos.append(int(y)/1160)

        txtname = filename.split(".", 1)
        txtfile = txt_path + txtname[0] + ".txt"
        with open(txtfile, "w") as f:
            f.write(str(4) + "," 
            + str(x_pos[2]) + "," + str(y_pos[2])+","
            + str(x_pos[3]) + "," + str(y_pos[3])+","
            + str(x_pos[0]) + "," + str(y_pos[0])+","
            + str(x_pos[1]) + "," + str(y_pos[1])
            + ",,")


txt_translate_1("D:\\TEMP_Work\\car_work\\ccpd2019\\CCPD2019\\CCPD2019\\ccpd_base_1","D:\\TEMP_Work\\car_work\\ccpd2019\\CCPD2019\\CCPD2019\\ccpd_base_1\\")



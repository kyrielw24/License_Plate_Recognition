import threading
import time
import tkinter as tk
from charset_normalizer import detect
import cv2
import lib.img_function as predict
import lib.img_math as img_math
import os
import lib.screencut as screencut
from threading import Thread
from tkinter import ttk
from tkinter.filedialog import *
from PIL import Image, ImageTk, ImageGrab
import tkinter.messagebox as msgbox
import requests
from time import sleep
from license_plate_detection import *
from str_separate import *

from matplotlib import pyplot


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return1 = None
        self._return2 = None
        self._return3 = None

    def run(self):
        if self._target is not None:
            try:
                self._return1, self._return2, self._return3 = self._target(*self._args, **self._kwargs)
            except:
                pass

    def join(self):
        Thread.join(self)
        return self._return1, self._return2, self._return3


class Surface(ttk.Frame):
    pic_path = ""
    thread = None
    thread_run = False
    camera = None
    pic_source = ""

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        top = ttk.Frame(self)
        win.title("车牌识别")
        win.minsize(1160, 850)
        self.center_window()
        self.pic_path3 = ""
        self.pic_num = None
        self.now_pic_num = None
        self.pred_path = None
        self.wpod_net_path = "D:\\TEMP_Work\\License_Plate_Recognition\\lp-detector\\wpod-net_update1"
        self.ft = ('Times', 18, 'bold')
        # 网址字符串
        self.var_url = tk.StringVar()
        # top部分-------------------------------------------------------------------------------------------------
        top.pack(side='top', expand=1)
        L1 = ttk.Label(top, text='欢迎使用车牌识别', font=('楷体', 28, 'bold'))
        L1.pack(side='top', pady=15)
        L2 = ttk.Label(top, text='—————第十五组：三尺童子队', font=('宋体', 18, 'bold italic'))
        L2.pack(side='right', pady=30)
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="10", pady="10")
        frame_left.pack(side='left', expand=1)
        frame_right1.pack(side='top', expand=0, pady=20)
        frame_right2.pack(side='bottom', expand=0)

        s = ttk.Style()
        s.configure('my.TButton', font=('宋体', 16), foreground="Orange")
        s1 = ttk.Style()
        s1.configure("TButton", font=("华文行楷", 13), foreground="BlueViolet")

        # frame_right部分------------------------------------------------------------------------------------------

        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        ttk.Label(frame_right1, text='定位车牌位置：', width=20, font=('黑体', 14)).grid(column=0, row=0)
        ttk.Label(frame_right1, text='定位识别结果：', width=20, font=('黑体', 14)).grid(column=0, row=3)

        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=30, style='my.TButton', command=self.from_pic)
        from_pic_ctl.pack(anchor="center", pady="15")

        reset_ctl = ttk.Button(frame_right2, text="重置窗口", width=30, style='my.TButton', command=self.reset)
        reset_ctl.pack(anchor="se", pady="15")

        clean_ctrl = ttk.Button(frame_right2, text="清除识别数据", width=30, style='my.TButton', command=self.clean)
        clean_ctrl.pack(anchor="se", pady="15")

        pic_from_url = ttk.Button(frame_right2, text="输入网络图片地址", width=30, style='my.TButton', command=self.popup)
        pic_from_url.pack(anchor="se", pady="15")

        pic_from_url = ttk.Button(frame_right2, text="识别网络图片", width=30, style='my.TButton', command=self.url_pic)
        pic_from_url.pack(anchor="se", pady="15")

        # 图片显示-----------------------------------------------------------------------------------------------------
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, pady=30)
        self.r_ctl = ttk.Label(frame_right1, text="", font=('Times', '20'), foreground="SandyBrown")
        self.r_ctl.grid(column=0, row=4, pady=30)

        ttk.Label(frame_right1, text='---------------------------------------').grid(column=0, row=5)

        next_pic = ttk.Button(frame_right1, text="NEXT", width=20, style="TButton", command=self.next)
        next_pic.grid(column=0, row=6, pady="15")

        self.clean()
        self.apistr = None

    def popup(self):
        win_pop = tk.Toplevel()
        win_pop.title('请输入网址')
        win_pop.geometry('500x100')

        def url_ensure():
            print(self.var_url.get())
            win_pop.destroy()

        tk.Label(win_pop, text='请输入网址：', height=1).pack(side='left', expand=1)

        self.url = tk.Entry(win_pop, textvariable=self.var_url, width=45)
        self.url.pack(side='left', expand=1)

        ensure = tk.Button(win_pop, text='确定', command=url_ensure)
        ensure.pack(side='right', expand=1)

    def url_pic(self):
        IMAGE_URL = self.var_url.get()
        URL_len = len(IMAGE_URL)
        if (IMAGE_URL == ""):
            msgbox.showinfo('提示', '请输入网址！')
            return
        if (URL_len > 1050):
            msgbox.showinfo('提示', '网址过长！')
            return
        r = requests.get(IMAGE_URL).content
        # 生成图片名称
        imgname = IMAGE_URL.split('/')[-1]
        # 图片存储路径
        imgpath = './tmp/url.jpeg'
        with open(imgpath, 'wb') as f:
            f.write(r)
        # print(IMAGE_URL)
        self.thread_run = False
        self.thread_run2 = False
        self.cameraflag = 0
        path1 = "D:\\TEMP_Work\\License_Plate_Recognition\\tmp\\"+"url.jpeg"
        self.clean()
        self.pic_source = "网络地址：" + IMAGE_URL
        self.pic(path1)

    def reset(self):
        self.reset2()
        self.reset2()

    def reset2(self):
        win.geometry("1160x850")
        self.clean()
        self.thread_run7 = False
        self.count = 0
        self.center_window()

    def center_window(self):
        screenwidth = win.winfo_screenwidth()
        screenheight = win.winfo_screenheight()
        win.update()
        width = win.winfo_width()
        height = win.winfo_height()
        size = '+%d+%d' % ((screenwidth - width) / 2, (screenheight - height) / 2)
        # print(size)
        win.geometry(size)

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        w, h = im.size
        pil_image_resized = self.resize2(w, h, im)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    def resize(self, w, h, pil_image):
        w_box = 200
        h_box = 50
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return pil_image.resize((width, height))

    def resize2(self, w, h, pil_image):
        width = win.winfo_width()
        height = win.winfo_height()
        w_box = width - 250
        h_box = height - 100
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return pil_image.resize((width, height))

    def next(self):
        if self.now_pic_num == self.pic_num:
            msgbox.showinfo('温馨提示', '当前已经是车辆识别的最后一张图片，没有下一张了，亲！')
        else:
            self.pred_pic(self.now_pic_num)

    def pic(self, pic_dir):

        # self.now_pic_num = index + 1
        # self.check_flag()
        # pic_path = self.pred_path + str(index + 1) + ".jpg"
        pic_path = pic_dir
        img_bgr = img_math.img_read(pic_path)
        self.imgtk = self.get_imgtk(img_bgr)
        self.image_ctl.configure(image=self.imgtk)
        out_dir = "D:/TEMP_Work/License_Plate_Recognition/pic"
        out_path = detection(pic_path, out_dir, self.wpod_net_path)
        pred_str = separate_and_predict(out_path)
        print(pred_str)
        self.img = Image.open(out_path)
        w, h = self.img.size
        img_resized = self.resize(w, h, self.img)
        self.tkImage1 = ImageTk.PhotoImage(image=img_resized)
        self.roi_ctl.configure(image=self.tkImage1, state='enable')
        self.r_ctl.configure(text=pred_str)

        # 多张图片使用预先经过yolo处理
        # num, path = darknet_function(pic_dir)   #  num是识别出来的有几张车辆图片   path 是这几张图片保留的位置
        # num = 1
        # self.pic_num = num
        # self.pred_path = "D:\\TEMP_Work\\License_Plate_Recognition\\pic\\"
        #
        # self.pred_pic(index=0)
        # for index in range(num):
        #     self.now_pic_num = index + 1
        #     # self.check_flag()
        #     pic_path = self.pred_path + str(index + 1) + ".jpg"
        #     img_bgr = img_math.img_read(pic_path)
        #     self.imgtk = self.get_imgtk(img_bgr)
        #     self.image_ctl.configure(image=self.imgtk)
        #     out_dir = "D:/TEMP_Work/License_Plate_Recognition/pic"
        #     out_path = detection(pic_path, out_dir, self.wpod_net_path)
        #     pred_str = separate_and_predict(out_path)
        #     print(pred_str)
        #     self.img = Image.open(out_path)
        #     w, h = self.img.size
        #     img_resized = self.resize(w, h, self.img)
        #     self.tkImage1 = ImageTk.PhotoImage(image=img_resized)
        #     self.roi_ctl.configure(image=self.tkImage1, state='enable')
        #     self.r_ctl.configure(text=pred_str)

    def pred_pic(self, index):
        self.now_pic_num = index + 1
        # self.check_flag()
        pic_path = self.pred_path + str(index + 1) + ".jpg"
        img_bgr = img_math.img_read(pic_path)
        self.imgtk = self.get_imgtk(img_bgr)
        self.image_ctl.configure(image=self.imgtk)
        out_dir = "D:/TEMP_Work/License_Plate_Recognition/pic"
        out_path = detection(pic_path, out_dir, self.wpod_net_path)
        pred_str = separate_and_predict(out_path)
        print(pred_str)
        self.img = Image.open(out_path)
        w, h = self.img.size
        img_resized = self.resize(w, h, self.img)
        self.tkImage1 = ImageTk.PhotoImage(image=img_resized)
        self.roi_ctl.configure(image=self.tkImage1, state='enable')
        self.r_ctl.configure(text=pred_str)

    def from_pic(self):
        self.thread_run = False
        self.thread_run2 = False
        self.pic_path = askopenfilename(title="选择识别图片",
                                        filetypes=[("jpg图片", "*.jpg"), ("jpeg图片", "*.jpeg"), ("png图片", "*.png")])
        self.clean()
        self.pic_source = "本地文件：" + self.pic_path
        self.pic(self.pic_path)

    def clean(self):
        if self.thread_run:
            self.cameraflag = 0
            return
        self.thread_run = False
        # img_bgr3 = img_math.img_read("pic/hy.png")
        img_bgr3 = img_math.img_read("pic/src2.jpg")
        self.imgtk2 = self.get_imgtk(img_bgr3)
        self.image_ctl.configure(image=self.imgtk2)

        self.r_ctl.configure(text="")

        self.pilImage3 = Image.open("pic/locate.png")
        w, h = self.pilImage3.size
        pil_image_resized = self.resize(w, h, self.pilImage3)
        self.tkImage3 = ImageTk.PhotoImage(image=pil_image_resized)
        self.roi_ctl.configure(image=self.tkImage3, state='enable')


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()

    surface = Surface(win)
    # close,退出输出destroy
    win.protocol('WM_DELETE_WINDOW', close_window)
    # 进入消息循环
    win.mainloop()

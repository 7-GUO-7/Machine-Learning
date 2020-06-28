import random
import os
import shutil

dic = dict()
with open('train.csv') as f:
    for line in f:
        line = line.rstrip('\n')
        row = line.split(',')
        t = row[1]
        l1 = t.split(';')
        l2 = [int(i) for i in l1]
        dic[row[0]] = l2


# li = []
# for it in dic.keys():
#     li.append(it)
# random.shuffle(li)
# train = li[:500]
# val = li[500:]
#
path = "D:\\ml_dataset\\train"
file_list = [name for name in os.listdir(path)]
# print(file_list)
l1 = [0 for i in range(10)]
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
for it in file_list:
    t = dic[it]
    for i in t:
        l1[i] += 1
        if i == 5:
            l5.append(it)
        if i == 6:
            l6.append(it)
        if i == 7:
            l7.append(it)
        if i == 8:
            l8.append(it)
        if i == 9:
            l9.append(it)
print(len(l9))
print(l1)

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    包含数据增强的八种方式
    """
    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像

        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(300, 500)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                os.makedirs(path)
            return 0
        else:
            return 1
    except:
        print("error")
        return -2


opsList = ["randomRotation", "randomCrop", "randomGaussian"]
funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomGaussian": DataAugmentation.randomGaussian
               }

from random import choice


if __name__ == '__main__':
    for i in range(60 - len(l5)):
        it = choice(l5)
        path = "D:\\2020年春节机器学习大作业训练集数据\\train\\" + it
        file_list = [name for name in os.listdir(path)]
        save = "D:\\generated_images\\5-" + str(i) + "-" + it
        if not os.path.exists(save):
            if not os.path.isfile(save):
                os.makedirs(save)
        for img in file_list:
            img_name = os.path.join(path, img)
            im = Image.open(img_name)
            method = choice(opsList)
            print(method)
            if method == "randomGaussian":
                im = np.array(im)
            new_image = funcMap[method](im)
            DataAugmentation.saveImage(new_image, os.path.join(save, img))
# for it in val:
#     t = dic[it]
#     for i in t:
#         l2[i] += 1
# print(l2)

# folder_name = "D:\\ml_dataset\\val"
# f = open('test_labels.txt','w')
# file_list = [name for name in os.listdir(folder_name)]
# for it in file_list:
#     f.write(str(it))
#     f.write(' ')
#     l = dic[it]
#     for i in l:
#         f.write(str(i))
#         f.write(' ')
#     f.write('\n')
    #     shutil.copytree(folder_name + '/' + str(it), "D:\\train" + '/' + str(it))
    # else:
    #     shutil.copytree(folder_name + '/' + str(it), "D:\\val" + '/' + str(it))

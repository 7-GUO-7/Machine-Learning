import os
import shutil

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from torchvision import models
from torchvision import transforms
class TorchDataset():
    def __init__(self, filename1="train_labels.txt", filename2 = "train_labels.txt",data_dir="dataset/train/", resize_height=None, resize_width=None, repeat=1):

        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_dir = data_dir
        self.image_label_list, self.image_path_list = self.read_file(filename1,filename2)

        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        # self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        data_name, label = self.image_label_list[index]
        # print(data_name, label)
        img_path = self.image_path_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        # print(img)
        # data_path = os.path.join(self.image_dir, data_name)
        # data = self.load_data(data_path, self.resize_height, self.resize_width, normalization=False)
        # data = self.data_preproccess(img)
        # label = np.array(label)
        return img, label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename,filename2):
        image_label_list = []
        image_path_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]

                pic_path = self.image_dir + name # name is pic folder
                pic_list = [x for x in os.listdir(pic_path)]
                for it in pic_list:
                    image_path_list.append(pic_path+'/'+it)
                number = len(pic_list)

                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                label = [0. for i in range(10)]
                for i in labels:
                    label[i] = 1.
                for i1 in range(number):
                    image_label_list.append((name, torch.FloatTensor(label)))
        with open(filename2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                 # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]

                pic_path = self.image_dir + name # name is pic folder
                pic_list = [x for x in os.listdir(pic_path)]
                for it in pic_list:
                    image_path_list.append(pic_path+'/'+it)
                number = len(pic_list)

                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                label = [0. for i in range(10)]
                for i in labels:
                    label[i] = 1.
                for i1 in range(number):
                    image_label_list.append((name, torch.FloatTensor(label)))
        # print(image_label_list)
        # print(image_path_list)
        return image_label_list, image_path_list

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        file_list = [name for name in os.listdir(path)]
        l = []
        for it in file_list:
            y = np.loadtxt(path + str(it), delimiter=',').reshape(1, 2048)
            l.append(y)
        return np.array(l, dtype='float32')

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        # data = self.toTensor(data)
        return data

# train_loader = TorchDataset(repeat=1)
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

# epoch_num = 2
# for epoch in range(epoch_num):
#     for batch_data, batch_label in train_loader:
#         # data = batch_data[0,:]
#         print(batch_data.shape, batch_label)

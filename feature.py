# -*- coding:utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import models
import cv2
from torchvision import transforms
from torch.nn import functional as F

sys.path.append('..')

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
NET_WEIGHT = 1

ckp_dir = OUTPUT_DIR + 'nn_output/'
filename = 'final-vgg16.pth'
filepath = os.path.join(ckp_dir, filename)


def get_nn_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    ckp_path = OUTPUT_DIR + 'nn_output/model_best-test.pth.tar'

    # checkpoint = torch.load(ckp_path, map_location='cpu')
    checkpoint = torch.load(ckp_path)

    d = checkpoint['state_dict']
    d = {k.replace('module.', ''): v for k, v in d.items()}
    model.load_state_dict(d)
    model.eval()
    return model


# def change_img(path, name):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # print(img)
#     print(len(img))
#     print(len(img[0]))
#     img = img[160:325,83:570]
#     print(len(img))
#     print(len(img[0]))
#     print()
#     cv2.imwrite(savePath+name, img)

def load_img(path):

    img = Image.open(path)
    # print(img)
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    return img


# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.net = models.resnet50(pretrained=True)
#
#     def forward(self, input):
#         output = self.net.conv1(input)
#         output = self.net.bn1(output)
#         output = self.net.relu(output)
#         output = self.net.maxpool(output)
#         output = self.net.layer1(output)
#         output = self.net.layer2(output)
#         output = self.net.layer3(output)
#         output = self.net.layer4(output)
#         output = self.net.avgpool(output)
#         return output


# class MyModel(nn.Module):
#     def __init__(self):   # input the dim of output fea-map of Resnet:
#         super(MyModel, self).__init__()
        
#         BackBone = torch.load('models/resnet26.pth')
 
#         self.BackBone = BackBone
 
 
#     def forward(self, input):   # input is 2048!
 
#         ##### 关键步骤 #####
#         for name, midlayer in self.BackBone._modules.items():
#             x = midlayer(x)
#             print(name)
#             if name == 'layer2':    # 取出resnet中的layer2层输出
#                 break
#         ##### 关键步骤 #####
 
#         x = self.BackBone(input)
 
#         return x

# transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ])


def predict(img_path, name):
    global historical, future
    count = 0
    his_flag = 1
    # nn_model = torch.load('models/resnet26.pth')
    nn_model = torch.load('models/resnet26.pth').cpu()
    nn_model.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye_(nn_model.fc.weight)
    
    # print(nn_model)
    # nn_model = models.resnet50(pretrained=True)
    # nn_model.fc = nn.Linear(2048, 2048)
    # torch.nn.init.eye(nn_model.fc.weight)

    # print(nn_model)

    nn_model.eval()
    # if count < historical and his_flag:
    #     img = load_img(img_path)
    #     feature = nn_model(img)
    #     print(type(feature))
    #     print(feature)
    #     count += 1
    img = load_img(img_path)
    # img = Image.open(img_path)
                
    # img = img.to(device)
    # img = transform(img)
    feature = nn_model(img)
    # print(type(feature))
    # print(feature.size())

    feature = feature.data.numpy()
    # print(feature)
    np.savetxt(featurePath2048_final+name[:-4]+"txt", feature, delimiter=',')
    # y = np.loadtxt(featurePath2048+name[:-3]+"txt", delimiter=',').reshape(1, 2048)


    # print(y)
    # probs = F.softmax(logit, dim=1).data.squeeze()
    # label_predict = np.argmax(probs.cpu().numpy())
    # print(label_predict)
    # print(probs)
    # nn_score = probs[1].cpu().numpy()
    # print(nn_score)
    # score = NET_WEIGHT * nn_score
    # # score = NET_WEIGHT * nn_score + (1 - NET_WEIGHT) * gbm_score
    # return round(score * 100, 1)




# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.rnn = nn.LSTM(
#             input_size=historical*1000,
#             hidden_size=64,
#             num_layers=3,
#
#         )
#         self.out = nn.Linear(64,future*1000)
#
#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x ,None)



if __name__ == '__main__':
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path = 'dataset/test/'
    for folder in os.listdir(Path):
        picPath = Path + folder + '/'
        # savePath = 'pics/3/'
        # featurePath2048_test = 'feature2048-test/'+folder+'/'
        featurePath2048_final = 'feature2048_final/'+folder+'/'
        # ckp_dir = OUTPUT_DIR + 'nn_output/'
        # historical = 15
        # future = 20
        # if not os.path.exists(savePath):
        #     os.makedirs(savePath)
        if not os.path.exists(featurePath2048_final):
            os.makedirs(featurePath2048_final)
        # if not os.path.exists(featurePath10):
        #     os.makedirs(featurePath10)



        # state = {'lr': 0.001}

        # model = models.resnet50(pretrained=True)
        # for parma in model.parameters():
        #     parma.requires_grad = False
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 2)

        # for name in os.listdir(picPath):
        #     change_img(picPath+name, name)
        #     print(name)


        for name in os.listdir(picPath):
            count+=1
            print(picPath+name)
            predict(picPath+name, name)
            print(name)
            print(count)


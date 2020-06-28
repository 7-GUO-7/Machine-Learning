import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os

import csv

# from dataloader_att import TorchDataset, attentionDataset
from model_att import Attention, GatedAttention
from sklearn import metrics
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

class attentionDataset():
    def __init__(self, filename="train_labels.txt", data_dir="D:\\feature2048_final", resize_height=None, resize_width=None, repeat=1):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = [name for name in os.listdir(data_dir)]
        self.image_dir = data_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

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
        data_name = self.image_label_list[index]
        # print(data_name, label)
        data_path = os.path.join(self.image_dir, data_name)
        data = self.load_data(data_path, self.resize_height, self.resize_width, normalization=False)
        # data = self.data_preproccess(data)
        # label = np.array(label)
        return data, data_name

    def __len__(self):
        if self.repeat == None:
            data_len = 1000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                label = [0. for i in range(10)]
                for i in labels:
                    label[i] = 1.
                image_label_list.append((name, torch.FloatTensor(label)))
        # print(image_label_list)
        return image_label_list

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
            y = np.loadtxt(path + '/' + str(it), delimiter=',').reshape(1, 2048)
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

class attentionDataset2():
    def __init__(self, filename="test_labels.txt", data_dir="D:\\feature2048-test", resize_height=None, resize_width=None, repeat=1):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = self.read_file(filename)
        self.image_dir = data_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

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
        data_path = os.path.join(self.image_dir, data_name)
        data = self.load_data(data_path, self.resize_height, self.resize_width, normalization=False)
        # data = self.data_preproccess(data)
        # label = np.array(label)
        return data, label

    def __len__(self):
        if self.repeat == None:
            data_len = 1000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                label = [0. for i in range(10)]
                for i in labels:
                    label[i] = 1.
                image_label_list.append((name, torch.FloatTensor(label)))
        # print(image_label_list)
        return image_label_list

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
            y = np.loadtxt(path + '/' + str(it), delimiter=',').reshape(1, 2048)
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


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(attentionDataset(),
    # target_number=args.target_number,
    #                                            mean_bag_length=args.mean_bag_length,
    #                                            var_bag_length=args.var_bag_length,
    #                                            num_bag=args.num_bags_train,
    #                                            seed=args.seed,
    #                                            train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(attentionDataset(), batch_size=1, shuffle=False, **loader_kwargs)
#                                               mean_bag_length=args.mean_bag_length,
#                                               var_bag_length=args.var_bag_length,
#                                               num_bag=args.num_bags_test,
#                                               seed=args.seed,
#                                               train=False),
#                                     batch_size=1,
#                                     shuffle=False,
#                                     **loader_kwargs)

print('Init Model')
if args.model=='attention':
    # model = Attention()
    model = torch.load('attention8.pth', map_location='cpu')
elif args.model=='gated_attention':
    model = GatedAttention()
    # model = torch.load('models/ga1.pth')
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def test():
    model.eval()
    # test_loss = 0.
    y = []
    y1 = []
    y2 = []
    for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
        bag_name = label
        # y.append([int(i) for i in bag_label])
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = Variable(data)
        y_prob = model.forward(data)[0].detach().cpu().numpy()
        label = []
        s1 = ""
        for i in range(10):
            s1 += str(round(y_prob[i], 4))
            s1 += ';'
        for i in range(10):
            if y_prob[i] > 0.5:
                label.append(i)
        if len(label) == 0:
            s = str(np.argmax(y_prob))
            print(y_prob, bag_name[0])
        else:
            s = str(label[0])
            for i in range(1, len(label)):
                s += ';'
                s += str(label[i])

        path = "test.csv"
        with open(path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [bag_name[0], s]
            csv_write.writerow(data_row)

        path2 = "test-pred.csv"
        with open(path2, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [bag_name[0], s1]
            csv_write.writerow(data_row)
        # loss = model.calculate_objective(data, bag_label)
        # test_loss += loss.data.item()
        # _, predicted_label, out = model.calculate_classification_error(data, bag_label)
        # # print(out)
        # y1.append(predicted_label.detach().cpu().numpy())
        # y2.append([int(i) for i in out])
        

    # test_loss /= len(test_loader)


    # print('\nTest Set, Loss: {:.4f}'.format(test_loss))
    #
    #
    # np.savetxt("y_gate8_train.txt",np.array(y),delimiter=',')
    # np.savetxt("y_prob_gate8_train.txt",np.array(y1),delimiter=',')
    # print("Classification report: \n", (classification_report(np.array(y), np.array(y2))))
    # print(metrics.roc_auc_score(np.array(y), y1, average='macro'))
    # print(metrics.f1_score(np.array(y), np.array(y2), average='macro'))
    # print(metrics.f1_score(np.array(y), np.array(y2), average='micro'))

def test2():
    model.eval()
    test_loss = 0.
    y = []
    y1 = []
    y2 = []
    for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
        bag_label = label[0]
        y.append([int(i) for i in bag_label])
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = Variable(data)
        # y_prob = model.forward(data)[0].detach().cpu().numpy()
        # label = []
        # for i in range(10):
        #     if y_prob[i] > 0.5:
        #         label.append(i)
        # if len(label) == 0:
        #     s = str(np.argmax(y_prob))
        #     print(y_prob, bag_name[0])
        # else:
        #     s = str(label[0])
        #     for i in range(1, len(label)):
        #         s += ';'
        #         s += str(label[i])

        # path = "test.csv"
        # with open(path, 'a+', newline='') as f:
        #     csv_write = csv.writer(f)
        #     data_row = [bag_name[0], s]
        #     csv_write.writerow(data_row)

        # path2 = "test2.csv"
        # with open(path2, 'a+', newline='') as f:
        #     csv_write = csv.writer(f)
        #     data_row = [bag_name[0], y_prob.tolist()]
        #     csv_write.writerow(data_row)
        loss = model.calculate_objective(data, bag_label)
        test_loss += loss.data.item()
        _, predicted_label, out = model.calculate_classification_error(data, bag_label)
        # print(out)
        y1.append(predicted_label.detach().cpu().numpy())
        y2.append([int(i) for i in out])

    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}'.format(test_loss))
    
    # np.savetxt("y_gate8_train.txt",np.array(y),delimiter=',')
    # np.savetxt("y_prob_gate8_train.txt",np.array(y1),delimiter=',')
    print("Classification report: \n", (classification_report(np.array(y), np.array(y2))))
    print(metrics.roc_auc_score(np.array(y), y1, average='macro'))
    print(metrics.f1_score(np.array(y), np.array(y2), average='macro'))
    print(metrics.f1_score(np.array(y), np.array(y2), average='micro'))
if __name__ == "__main__":
    # print(torch.__version__)
    print('Start Testing')
    test()

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataloader import TorchDataset
from model import Attention, GatedAttention
from sklearn import metrics
from sklearn.metrics import classification_report

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
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

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_loader = data_utils.DataLoader(TorchDataset(filename="test_labels.txt", data_dir="dataset/val"), batch_size=1, shuffle=True, **loader_kwargs)
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()


def test():
    model = torch.load('model40.pth')
    model.eval()
    test_loss = 0.
    y = []
    y1 = []
    y2 = []
    for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
        bag_label = label
        y.append([int(i) for i in bag_label[0]])
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss = model.calculate_objective(data, bag_label)
        test_loss += loss.data.item()
        _, predicted_label, out = model.calculate_classification_error(data, bag_label)
        y1.append(predicted_label.detach().cpu().numpy())
        y2.append([int(i) for i in out])

    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}'.format(test_loss))

    print ("Classification report: \n")
     
    print(classification_report(np.array(y), np.array(y2)))
    print(np.array(y))
    print(np.array(y2))

    print(metrics.roc_auc_score(y, y1, average='macro'))
    print(metrics.f1_score(y, y2, average='macro'))
    print(metrics.f1_score(y, y2, average='micro'))

test()
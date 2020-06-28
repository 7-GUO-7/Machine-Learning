
import numpy as np

import argparse
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from torch.autograd import Variable
from tqdm import tqdm

from dataloader import TorchDataset
from model import Attention, GatedAttention
from torchvision import models

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
# args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = True


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(TorchDataset(),
    # target_number=args.target_number,
    #                                            mean_bag_length=args.mean_bag_length,
    #                                            var_bag_length=args.var_bag_length,
    #                                            num_bag=args.num_bags_train,
    #                                            seed=args.seed,
    #                                            train=True),
                                     batch_size=100,
                                     shuffle=True,
                                     **loader_kwargs)

# test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
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
    model = models.resnet50(pretrained=True)
    # for parma in model.parameters():
    #     parma.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)


    # model = Attention()
    # model = torch.load('model40.pth')
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def calculate_objective(model, X, Y):
    # Y = Y.float()
    Y = Y.float()
    loss = nn.BCELoss(reduction ='sum')
    Y_prob = model(X)

    ###softmax

    Y_prob = torch.sigmoid(Y_prob)
    Y_prob = torch.clamp(Y_prob, min=1e-6, max=1. - 1e-6)
    # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    print(Y_prob)
    # print(Y)
    output = loss(Y_prob, Y)
    # print(output.item())
    return output

# AUXILIARY METHODS
def calculate_classification_error(model, X, Y):
    # Y = Y.float()
    Y_prob = model(X)
    Y_prob = torch.sigmoid(Y_prob)[0]
    Y = Y[0]
    Y_hat = []
    # p = np.sum(np.array(Y))
    # print(p)
    for i in range(len(Y_prob)):
        Y_hat.append(torch.ge(Y_prob[i], 0.5).float())
    error = 0
    for i in range(len(Y_hat)):
        if int(Y_hat[i]) != int(Y[i]):
            error += 1
            break
    #     error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().item()

    return error,Y_prob,Y_hat

def train(epoch):
    model.train()
    train_loss = 0.
    loss = torch.nn.BCELoss()
    train_error = 0.
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        # print(data)
        # print(label)
        bag_label = label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        loss = calculate_objective(model, data, bag_label)
        print(loss)
        # print(data)
        # print()
        # print(bag_label)
        train_loss += loss.data.item()
        error, _, _ = calculate_classification_error(model, data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()


        # calculate loss and metrics


    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    torch.save(model, 'models/model%i.pth'%(epoch))
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


# def test():
#     model.eval()
#     test_loss = 0.
#     test_error = 0.
#     for batch_idx, (data, label) in enumerate(test_loader):
#         bag_label = label[0]
#         instance_labels = label[1]
#         if args.cuda:
#             data, bag_label = data.cuda(), bag_label.cuda()
#         data, bag_label = Variable(data), Variable(bag_label)
#         loss, attention_weights = model.calculate_objective(data, bag_label)
#         test_loss += loss.data[0]
#         error, predicted_label = model.calculate_classification_error(data, bag_label)
#         test_error += error

#         if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
#             bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
#             instance_level = list(zip(instance_labels.numpy()[0].tolist(),
#                                  np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

#             print('\nTrue Bag Label, Predicted Bag Label: {}\n'
#                   'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

#     test_error /= len(test_loader)
#     test_loss /= len(test_loader)

#     print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    # test()

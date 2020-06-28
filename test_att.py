import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataloader_att import TorchDataset, attentionDataset
# from model import Attention, GatedAttention
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

test_loader = data_utils.DataLoader(attentionDataset(filename="test_labels.txt", data_dir="feature2048-test"), batch_size=1, shuffle=True, **loader_kwargs)
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
    model = torch.load('models/gateattention19.pth')
elif args.model=='gated_attention':
    model = GatedAttention()
    # model = torch.load('models/ga1.pth')
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def test():
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
        data, bag_label = Variable(data), Variable(bag_label)
        loss = model.calculate_objective(data, bag_label)
        test_loss += loss.data.item()
        _, predicted_label, out = model.calculate_classification_error(data, bag_label)
        # print(out)
        y1.append(predicted_label.detach().cpu().numpy())
        y2.append([int(i) for i in out])
        

    test_loss /= len(test_loader)


    print('\nTest Set, Loss: {:.4f}'.format(test_loss))


    np.savetxt("y_gate8_train.txt",np.array(y),delimiter=',')
    np.savetxt("y_prob_gate8_train.txt",np.array(y1),delimiter=',')
    print("Classification report: \n", (classification_report(np.array(y), np.array(y2))))
    print(metrics.roc_auc_score(np.array(y), y1, average='macro'))
    print(metrics.f1_score(np.array(y), np.array(y2), average='macro'))
    print(metrics.f1_score(np.array(y), np.array(y2), average='micro'))
if __name__ == "__main__":
    # print(torch.__version__)
    print('Start Testing')
    test()
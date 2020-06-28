import os
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import classification_report

# dic = dict()
# with open('train.csv') as f:
#     for line in f:
#         line = line.rstrip('\n')
#         row = line.split(',')
#         t = row[1]
#         l1 = t.split(';')
#         l2 = [int(i) for i in l1]
#         dic[row[0]] = l2
#
# y = []
# y1 = []
# y2 = []
# folder_name = "feature10"
# file_list = [name for name in os.listdir(folder_name)]
# for it in file_list:
#     p = [0 for i in range(10)]
#     for i in dic[it]:
#         p[i] = 1
#     y.append(np.array(p))
#     cell_name = folder_name + '/' + str(it)
#     img_list = [name for name in os.listdir(cell_name)]
#     img_list = [folder_name + '/' + str(it) + '/' + str(name) for name in img_list]
#     y_p = np.zeros(10)
#     for t in img_list:
#         out = np.loadtxt(t, delimiter=',')
#         yq = torch.sigmoid(torch.from_numpy(out)).numpy()
#         y_p += yq
#     y_p /= len(img_list)
#     y1.append(y_p.tolist())
#     threshold = [0.3, 0.16, 0.26, 0.38, 0.22, 0.24, 0.24, 0.3, 0.32, 0.3]
#     y_o = []
#     for i in range(10):
#         if y_p[i] > threshold[i]:
#             y_o.append(1)
#         else:
#             y_o.append(0)
#     y2.append(y_o)

threshold = [0.1, 0.7, 0.95, 0.1, 0.7, 0.3, 0.5, 0.1, 0.45, 0.2]
y = np.loadtxt("y_gate8.txt", delimiter=',')
y1 = np.loadtxt("y_prob_gate8.txt", delimiter=',')
y2 = []
for it in y1:
    yr = []
    for i in range(10):
        if it[i] > 0.5:
            yr.append(1)
        else:
            yr.append(0)
    y2.append(yr)

#
# np.savetxt("y.txt", np.array(y), delimiter=',')
# np.savetxt("y_prob.txt", np.array(y1), delimiter=',')


print("Classification report: \n", (classification_report(y, np.array(y2))))
print(metrics.roc_auc_score(y, np.array(y1), average='macro'))
print(metrics.f1_score(y, np.array(y2), average='macro'))
print(metrics.f1_score(y, np.array(y2), average='micro'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.fc = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.proj = nn.Linear(self.L*self.K, 10)

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        
        H = self.fc(x)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        out = self.proj(M)

        return torch.sigmoid(out)

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        Y_prob = self.forward(X)[0]
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

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        # print(output.item())
        return output

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )
        self.fc = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier1 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier8 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier9 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        self.classifier10 = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL
        H = self.fc(x)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        prob = []
        prob.append(self.classifier1(M))
        prob.append(self.classifier2(M))
        prob.append(self.classifier3(M))
        prob.append(self.classifier4(M))
        prob.append(self.classifier5(M))
        prob.append(self.classifier6(M))
        prob.append(self.classifier7(M))
        prob.append(self.classifier8(M))
        prob.append(self.classifier9(M))
        prob.append(self.classifier10(M))
        Y_prob = np.array(prob)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        Y_hat = np.zeros(10)
        for i in range(len(Y_hat)):
            Y_hat[i] = torch.ge(Y_prob[i], 0.5).float()
        return Y_prob, Y_hat, A


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 0
        for i in range(len(Y_hat)):
            # print(type(Y[i]))
            error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y_prob, _, A = self.forward(X)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = 0
        for i in range(10):
            Y_prob[i] = torch.clamp(Y_prob[i], min=1e-5, max=1. - 1e-5)
            neg_log_likelihood += -1. * (Y[i].float * torch.log(Y_prob[i]) + (1. - Y[i].float) * torch.log(1. - Y_prob[i]))  # negative log bernoulli

        return neg_log_likelihood, A

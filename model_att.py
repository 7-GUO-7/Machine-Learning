import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                    .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.sum(dim=1).mean()

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

        self.proj = nn.Sequential(
            nn.Linear(self.L*self.K, 100),
            nn.ReLU(inplace=False),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        
        H = self.fc(x)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        out = self.proj(M)

        return out

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        Y_prob = self.forward(X)[0]
        #Y = Y[0]
        #print(Y)
        Y_hat = []
        # p = np.sum(np.array(Y))
        # print(p)
        for i in range(len(Y_prob)):
            Y_hat.append(torch.ge(Y_prob[i], 0.5).float())
        error = 0
        for i in range(len(Y_hat)):
            if int(Y_hat[i].item()) != int(Y[i].item()):
                error += 1
                break
        #     error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().item()

        return error, Y_prob, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        #print(output.item())
        return output

    def calculate_objective2(self, X, Y):
        Y = Y.float()
        loss = FocalLoss()
        # loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        Y = Y.view(1,10)
        output = loss(Y_prob, Y)
        #print(output)
        return output

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

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

        self.proj = nn.Sequential(
            nn.Linear(self.L*self.K, 100),
            nn.ReLU(inplace=False),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        
        H = self.fc(x)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        out = self.proj(M)
        return out


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        Y_prob = self.forward(X)[0]
        #Y = Y[0]
        #print(Y)
        Y_hat = []
        # p = np.sum(np.array(Y))
        # print(p)
        for i in range(len(Y_prob)):
            Y_hat.append(torch.ge(Y_prob[i], 0.5).float())
        error = 0
        for i in range(len(Y_hat)):
            if int(Y_hat[i].item()) != int(Y[i].item()):
                error += 1
                break
        #     error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().item()

        return error, Y_prob, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        #print(output.item())
        return output

    def calculate_objective2(self, X, Y):
        Y = Y.float()
        loss = FocalLoss()
        # loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        Y = Y.view(1,10)
        output = loss(Y_prob, Y)
        #print(output)
        return output

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, 10, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        # if self.activation is not None:
        #     y = self.activation(y)
        return torch.sigmoid(y)

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

        return error, Y_prob, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        #print(output.item())
        return output

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size = 1
        seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size = 1
        seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size = 1
        seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
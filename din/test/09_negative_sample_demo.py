import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


def t1(n=10, c=10000):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding(n, 2)
            self.linear = nn.Linear(in_features=2, out_features=c)

        def forward(self, x):
            x1 = self.embedding(x)
            return self.linear(x1)

    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    y0 = torch.tensor([0, 2232, 113, 26, 745, 82])
    net = Net()
    loss_fn = nn.CrossEntropyLoss()

    _t1 = time.time()
    # 得到的是六个样本分别属于c个类别的置信度
    y1 = net(x0)  # [B,C]
    # 普通的损失函数
    loss = loss_fn(y1, y0)
    print(f"耗时:{time.time() - _t1}")
    print(loss)

    # 反向的传递
    net.zero_grad()
    loss.backward()
    print(net.linear.weight.grad.size())
    print(net.linear.weight.grad)
    print(net.linear.weight.grad[100:120])


def t2(n=10, c=10000):
    class NSLayer(nn.Module):
        def __init__(self, in_features, out_features, neg_num):
            super(NSLayer, self).__init__()
            self.out_features = out_features
            self.neg_num = neg_num
            self.weights = Parameter(torch.Tensor(in_features, out_features))

        def forward(self, x, y=None):
            if y is None:
                # 这个就是普通的线性转换一样，主要应用在推理的时候
                return torch.matmul(x, self.weights)
            else:
                # 负采样
                # 正例对应的权重
                pos_w = self.weights[:, y]  # [input_features, B]
                # 负例对应的权重(TODO: 这里优化一下，保证负采样类别和正例类别不重复即可)
                neg_idx = np.random.randint(0, self.out_features, self.neg_num * y.size()[0])
                neg_w = self.weights[:, neg_idx]
                # 合并权重
                _w = torch.cat([pos_w, neg_w], dim=1)
                return torch.matmul(x, _w)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding(n, 2)
            self.linear = NSLayer(2, c, 100)

        def forward(self, x, y=None):
            x1 = self.embedding(x)
            return self.linear(x1, y)

    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    y0 = torch.tensor([0, 2232, 213, 26, 745, 82])
    net = Net()

    # 普通的损失函数
    loss_fn = nn.CrossEntropyLoss()

    _t1 = time.time()
    # 得到的是六个样本分别属于c个类别的置信度
    y1 = net(x0, y=y0)  # [B,C]
    # 更改样本所属类别(和负采样里面参数组合有关)
    y0_ = torch.tensor(np.arange(0, y0.size()[0])).long()
    loss = loss_fn(y1, y0_)
    print(loss)
    print(f"耗时:{time.time() - _t1}")

    # # 反向的传递
    # net.zero_grad()
    # loss.backward()
    # print(net.linear.weights.grad.size())
    # print(net.linear.weights.grad)
    # print(net.linear.weights.grad[:, 100:120].t())
    #
    # # 得到的是六个样本分别属于c个类别的置信度
    # y1 = net(x0, y=y0)  # [B,C]
    # # 更改样本所属类别(和负采样里面参数组合有关)
    # y0_ = torch.tensor(numpy.arange(0, y0.size()[0])).long()
    # loss = loss_fn(y1, y0_)
    # print(loss)
    #
    # # 反向的传递
    # net.zero_grad()
    # loss.backward()
    # print(net.linear.weights.grad.size())
    # print(net.linear.weights.grad)
    # print(net.linear.weights.grad[:, 100:120].t())
    for i in range(10000):
        # 得到的是六个样本分别属于c个类别的置信度
        y1 = net(x0, y=y0)  # [B,C]
        # y1 = net(x0)  # [B,C]
        # 更改样本所属类别(和负采样里面参数组合有关)
        y0_ = torch.tensor(np.arange(0, y0.size()[0])).long()
        # y0_ = y0
        loss = loss_fn(y1, y0_)

        # 反向的传递
        net.zero_grad()
        loss.backward()
        # print(net.linear.weights.grad.size())
        # print(net.linear.weights.grad)
        d = net.linear.weights.grad[:, 100:120].numpy()
        d = np.mean(d, 0)
        d = d[d != 0]
        if len(d) > 1:
            print(net.linear.weights.grad[:, 100:120])
            if i > 10:
                break


if __name__ == '__main__':
    _c = 10000
    t1(c=_c)
    t2(c=_c)

import torch
from torch.nn import modules as nn


def t1(n=10000):
    """
    第一种基于OneHot的深度学习的代码应用
    :return:
    """
    # [6]
    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    # [6, n]
    x1 = torch.nn.functional.one_hot(x0, n).float()
    # FC(FC里面的w维度是[n,2]，而且n非常非常的大，计算量就是: 针对每条数据2N次乘法)
    x2 = nn.Linear(n, 2, bias=False)(x1)
    print(x0)
    print(x1)
    print(x2)


def t2(n=10):
    """
    理解t1中的计算过程，如果我们认为x2是x0的某种映射（embedding）
    实际上就发现虽然内部是矩阵乘法，但是由于输入的是OneHot之后的向量(只有1个位置为1，其它所有位置为0)
    所以最终就相当于直接在FC里面的参数w中直接提取
    :return:
    """
    # [6]
    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    # [6, n]
    x1 = torch.nn.functional.one_hot(x0, n).float()
    # FC
    linear = nn.Linear(n, 2, bias=False)
    x2 = linear(x1)
    print(x0)
    print(x1)
    print(x2)
    weight = torch.transpose(linear.weight, 1, 0)
    print(weight)
    x3 = weight[x0]  # 直接按照索引提取数据
    print(x3)


def t3(n=10):
    """
    理解t1中的计算过程，如果我们认为x2是x0的某种映射（embedding）
    实际上就发现虽然内部是矩阵乘法，但是由于输入的是OneHot之后的向量(只有1个位置为1，其它所有位置为0)
    所以最终就相当于直接在FC里面的参数w中直接提取
    :return:
    """
    # [6]
    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    # embedding
    embedding = nn.Embedding(num_embeddings=n, embedding_dim=2)
    weight = embedding.weight
    x2 = embedding(x0)
    print(x0)
    print(x2)
    print(weight[x0])


def t4(n=10):
    """
    理解t1中的计算过程，如果我们认为x2是x0的某种映射（embedding）
    实际上就发现虽然内部是矩阵乘法，但是由于输入的是OneHot之后的向量(只有1个位置为1，其它所有位置为0)
    所以最终就相当于直接在FC里面的参数w中直接提取
    :return:
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding(n, 2)
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            x1 = self.embedding(x)
            print(x1)
            return self.linear(x1)

    # [6]
    x0 = torch.tensor([1, 2, 3, 6, 7, 8])  # 同一个类别的离散特征的输入
    y0 = torch.rand_like(x0, dtype=torch.float32)
    net = Net()

    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    print(list(net.parameters()))

    for i in range(2):
        print("=" * 100)
        y1 = net(x0)
        loss = loss_fn(y1.view(-1), y0.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == '__main__':
    t1()
    t2()
    t3()

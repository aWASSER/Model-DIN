import numpy as np
import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.eps = 1e-10
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        进行激活操作
        :param x: [N,E]N表示样本数目，E表示每个样本的特征维度大小
        :return:
        """
        # 1. 计算每个样本的均值和标准差
        avg = x.mean(dim=0)  # [E]
        std = x.std(dim=0)  # [E]
        # 2. 归一化处理
        norm_x = (x - avg) / (std + self.eps)  # [N,E]
        # 3. 得到概率值
        p = torch.sigmoid(norm_x)  # [N,E]
        # 结果返回
        z = x.mul(p) + self.alpha * x.mul(1 - p)
        return z


# noinspection PyAbstractClass,PyShadowingNames
class SparseFeaturesEmbedding(nn.Module):
    """
    针对稀疏特征进行转换
    """

    def __init__(self, field_dims, embed_dim):
        """
        进行稀疏特征的embedding操作
        :param field_dims: [2, 3, 4, 5]给定各个属性的取值类别数目
        :param embed_dim: 映射的范围大小
        """
        super(SparseFeaturesEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=sum(field_dims), embedding_dim=embed_dim)
        # e.g. field_dims = [2, 3, 4, 5], offsets = [0, 2, 5, 9]
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        # 位置偏移(本来每个类别的取值是从0开始，所以需要进行偏移)
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)


# noinspection PyAbstractClass
class DenseFeaturesEmbedding(nn.Module):
    def __init__(self, num_fields, out_features):
        """
        稠密特征的映射
        :param num_fields: 总的稠密特征属性的数目
        :param out_features: 每个特征属性映射的输出特征维度大小
        """
        super(DenseFeaturesEmbedding, self).__init__()
        self.num_fields = num_fields
        self.linear = nn.Linear(1, out_features, bias=False)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, out_features)
        """
        x = x.view(-1, self.num_fields, 1)
        return self.linear(x)


def get_activation_function(act):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'dice':
        return Dice()
    else:
        raise ValueError(f"当前不支持激活函数:{act}")


# noinspection PyAbstractClass
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_features, units=None, act='prelu'):
        super(MultilayerPerceptron, self).__init__()
        if units is None:
            units = [128, 64, 32, 1]
        _layers = []
        in_features = input_features
        n = len(units) - 1
        for idx, unit in enumerate(units):
            _layers.append(nn.Linear(in_features=in_features, out_features=unit))
            if idx != n:
                # 除了最后一层外，其它所有层加激活函数
                _layers.append(get_activation_function(act))
            in_features = unit
        self.mlp = nn.Sequential(*_layers)

    def forward(self, v):
        return self.mlp(v)


class DINActivationUnit(nn.Module):
    def __init__(self, embedding_dim):
        super(DINActivationUnit, self).__init__()
        self.mlp = MultilayerPerceptron(
            input_features=3 * embedding_dim,
            units=[36, 1],
            act='dice'
        )

    def forward(self, user_behavior_features, ad_features):
        """
        进行DIN的特征激活单元
        :param user_behavior_features: [N,T,E] N个样本/用户，每个样本存在T个行为，每个行为用长度为E的向量进行表示
        :param ad_features: [N,E] N个样本/商品，每个样本用长度为E的向量进行表示
        :return: [N,T,E]
        """
        T = user_behavior_features.size()[1]
        ad_features = ad_features[:, None, :]  # [N,E] --> [N,1,E]
        x = user_behavior_features * ad_features  # [N,T,E]
        x = torch.cat([user_behavior_features, x, torch.tile(ad_features, dims=[1, T, 1])], dim=-1)  # [N,T,3E]
        w = self.mlp(x)  # [N,T,1]
        v = user_behavior_features * w  # [N,T,E] * [N,T,1] =[N,T,E]
        return v

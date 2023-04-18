import os

import torch
import torch.nn as nn
from sklearn import metrics
from common import SparseFeaturesEmbedding, MultilayerPerceptron, DINActivationUnit
from surprise import accuracy

class BasicDIN(nn.Module):
    def __init__(self):
        super(BasicDIN, self).__init__()
        embedding_size = 8
        self.user_profile_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[2, 10],  # 给定用户基础特征每个特征属性的类别数目列表，总特征数目是C1=len(field_dims)
            embed_dim=embedding_size
        )
        self.ad_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[100000, 1000, 100],  # 给定的商品的特征的每个类别数目列表，长度一定为3:商品id、店铺id、品类id
            embed_dim=embedding_size
        )
        self.context_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[10, 10],
            embed_dim=embedding_size
        )
        self.mlp = MultilayerPerceptron(input_features=80, units=[200, 80, 2])

    def forward(self, user_profile_features, user_behaviors, candidate_ad, context_features):
        """
        基于输入的四个方面的特征信息，进行前向过程，最终输出对于候选商品的点击、不点击的置信度
        NOTE：
            N表示批次的样本大小
        :param user_profile_features: [N,C1]用户基础特征，假定全部都是离散特征，也就是每个用户用C1个int类型的id值进行特征描述
        :param user_behaviors: [N,T,3]用户的行为特征，每个用户都存在一个行为序列，序列长度为T，序列内每个时刻对应操作包括三个特征:商品id、店铺id、品类id
        :param candidate_ad: [N,1,3]候选商品基础特征，包括：商品id、店铺id、品类id
        :param context_features:
        :return:
        """
        batch_size = user_profile_features.size()[0]
        # 1. embedding层
        user_features = self.user_profile_embedding_layer(user_profile_features)  # [N,C1] --> [N,C1,E]
        user_features = user_features.view(batch_size, -1)  # [N,C1,E] --> [N,C1*E]
        user_behavior_features = self.ad_embedding_layer(user_behaviors)  # [N,T,3] --> [N,T,3,E]
        behavior_length = user_behavior_features.size()[1]
        user_behavior_features = user_behavior_features.view(batch_size, behavior_length, -1)  # [N,T,3,E] ->  [N,T,3*E]
        user_behavior_features = user_behavior_features.sum(dim=1)  # [N,T,3*E] --> [N,3*E]
        ad_features = self.ad_embedding_layer(candidate_ad)  # [N,1,3] --> [N,1,3,E]
        ad_features = ad_features.view(batch_size, -1)  # [N,1,3,E] -> [N,1*3*E]
        context_features = self.context_embedding_layer(context_features)  # [N,C2] -> [N,C2,E]
        context_features = context_features.view(batch_size, -1)  # [N,C2,E] -> [N,C2*E]

        # 2. concat合并embedding的结果
        x = torch.cat([user_features, user_behavior_features, ad_features, context_features], dim=1)

        # 3. 全连接
        x = self.mlp(x)

        return x


class DIN(nn.Module):
    def __init__(self):
        super(DIN, self).__init__()
        embedding_size = 8
        self.user_profile_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[2, 10],  # 给定用户基础特征每个特征属性的类别数目列表，总特征数目是C1=len(field_dims)
            embed_dim=embedding_size
        )
        self.ad_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[100000, 1000, 100],  # 给定的商品的特征的每个类别数目列表，长度一定为3:商品id、店铺id、品类id
            embed_dim=embedding_size
        )
        self.context_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[10, 10],
            embed_dim=embedding_size
        )
        self.act_unit_layer = DINActivationUnit(embedding_dim=embedding_size * 3)  # 商品id、店铺id、品类id
        self.mlp = MultilayerPerceptron(input_features=80, units=[200, 80, 2], act='Dice')

    def forward(self, user_profile_features, user_behaviors, candidate_ad, context_features, return_l2_loss=False):
        """
        基于输入的四个方面的特征信息，进行前向过程，最终输出对于候选商品的点击、不点击的置信度
        NOTE：
            N表示批次的样本大小
        :param user_profile_features: [N,C1]用户基础特征，假定全部都是离散特征，也就是每个用户用C1个int类型的id值进行特征描述
        :param user_behaviors: [N,T,3]用户的行为特征，每个用户都存在一个行为序列，序列长度为T，序列内每个时刻对应操作包括三个特征:商品id、店铺id、品类id
        :param candidate_ad: [N,1,3]候选商品基础特征，包括：商品id、店铺id、品类id
        :param context_features:
        :return:
        """
        batch_size = user_profile_features.size()[0]
        l2_loss = []
        # 1. embedding层
        user_features = self.user_profile_embedding_layer(user_profile_features)  # [N,C1] --> [N,C1,E]
        user_features = user_features.view(batch_size, -1)  # [N,C1,E] --> [N,C1*E]
        l2_loss.append(user_features)
        user_behavior_features = self.ad_embedding_layer(user_behaviors)  # [N,T,3] --> [N,T,3,E]
        behavior_length = user_behavior_features.size()[1]
        user_behavior_features = user_behavior_features.view(batch_size, behavior_length, -1)  # [N,T,3,E] ->  [N,T,3*E]
        l2_loss.append(user_behavior_features)
        ad_features = self.ad_embedding_layer(candidate_ad)  # [N,1,3] --> [N,1,3,E]
        ad_features = ad_features.view(batch_size, -1)  # [N,1,3,E] -> [N,1*3*E]
        l2_loss.append(ad_features)
        # 特征融合&加权
        user_behavior_features = self.act_unit_layer(user_behavior_features, ad_features)
        user_behavior_features = user_behavior_features.sum(dim=1)  # [N,T,3*E] --> [N,3*E]
        context_features = self.context_embedding_layer(context_features)  # [N,C2] -> [N,C2,E]
        context_features = context_features.view(batch_size, -1)  # [N,C2,E] -> [N,C2*E]
        l2_loss.append(context_features)

        # 2. concat合并embedding的结果
        x = torch.cat([user_features, user_behavior_features, ad_features, context_features], dim=1)

        # 3. 全连接
        x = self.mlp(x)

        # 4. 计算损失
        if return_l2_loss:
            l2_loss = sum([torch.pow(v, 2).sum() for v in l2_loss])
            l2_loss += sum([torch.pow(v, 2).sum() for v in self.mlp.parameters()])
            return x, l2_loss
        else:
            return x


class DIEN(nn.Module):
    def __init__(self):
        super(DIEN, self).__init__()
        embedding_size = 8
        self.user_profile_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[2, 10],  # 给定用户基础特征每个特征属性的类别数目列表，总特征数目是C1=len(field_dims)
            embed_dim=embedding_size
        )
        self.ad_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[100000, 1000, 100],  # 给定的商品的特征的每个类别数目列表，长度一定为3:商品id、店铺id、品类id
            embed_dim=embedding_size
        )
        self.context_embedding_layer = SparseFeaturesEmbedding(
            field_dims=[10, 10],
            embed_dim=embedding_size
        )
        self.gru = nn.GRU(input_size=3 * embedding_size, hidden_size=3 * embedding_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=3 * embedding_size, hidden_size=3 * embedding_size, num_layers=1,
                           batch_first=True)
        self.act_unit_layer = DINActivationUnit(embedding_dim=embedding_size * 3)  # 商品id、店铺id、品类id
        self.mlp = MultilayerPerceptron(input_features=80, units=[200, 80, 2], act='Dice')

    def forward(self, user_profile_features, user_behaviors, candidate_ad, context_features, return_l2_loss=False):
        """
        基于输入的四个方面的特征信息，进行前向过程，最终输出对于候选商品的点击、不点击的置信度
        NOTE：
            N表示批次的样本大小
        :param user_profile_features: [N,C1]用户基础特征，假定全部都是离散特征，也就是每个用户用C1个int类型的id值进行特征描述
        :param user_behaviors: [N,T,3]用户的行为特征，每个用户都存在一个行为序列，序列长度为T，序列内每个时刻对应操作包括三个特征:商品id、店铺id、品类id
        :param candidate_ad: [N,1,3]候选商品基础特征，包括：商品id、店铺id、品类id
        :param context_features:
        :return:
        """
        batch_size = user_profile_features.size()[0]
        l2_loss = []
        # 1. embedding层
        user_features = self.user_profile_embedding_layer(user_profile_features)  # [N,C1] --> [N,C1,E]
        user_features = user_features.view(batch_size, -1)  # [N,C1,E] --> [N,C1*E]
        l2_loss.append(user_features)
        user_behavior_features = self.ad_embedding_layer(user_behaviors)  # [N,T,3] --> [N,T,3,E]
        behavior_length = user_behavior_features.size()[1]
        user_behavior_features = user_behavior_features.view(batch_size, behavior_length, -1)  # [N,T,3,E] ->  [N,T,3*E]
        l2_loss.append(user_behavior_features)
        ad_features = self.ad_embedding_layer(candidate_ad)  # [N,1,3] --> [N,1,3,E]
        ad_features = ad_features.view(batch_size, -1)  # [N,1,3,E] -> [N,1*3*E]
        l2_loss.append(ad_features)
        context_features = self.context_embedding_layer(context_features)  # [N,C2] -> [N,C2,E]
        context_features = context_features.view(batch_size, -1)  # [N,C2,E] -> [N,C2*E]
        l2_loss.append(context_features)

        # 特征融合&加权
        user_behavior_features, _ = self.gru(user_behavior_features)
        user_behavior_features = self.act_unit_layer(user_behavior_features, ad_features)
        # user_behavior_features = user_behavior_features.sum(dim=1)  # [N,T,3*E] --> [N,3*E]
        _, user_behavior_features = self.gru2(user_behavior_features)
        user_behavior_features = user_behavior_features[0]

        # 2. concat合并embedding的结果
        x = torch.cat([user_features, user_behavior_features, ad_features, context_features], dim=1)

        # 3. 全连接
        x = self.mlp(x)

        # 4. 计算损失
        if return_l2_loss:
            l2_loss = sum([torch.pow(v, 2).sum() for v in l2_loss])
            l2_loss += sum([torch.pow(v, 2).sum() for v in self.mlp.parameters()])
            l2_loss += sum([torch.pow(v, 2).sum() for v in self.gru.parameters()])
            return x, l2_loss
        else:
            return x


def test():
    user_profile_features = torch.tensor([
        [0, 7],
        [1, 8],
        [0, 5]
    ]).int()
    user_behaviors = torch.tensor([
        [
            [12, 15, 34],
            [16, 23, 45],
            [18, 23, 45],
            [23, 34, 12]
        ],
        [
            [879, 23, 12],
            [213, 23, 12],
            [2334, 12, 34],
            [12542, 23, 12]
        ],
        [
            [879, 23, 12],
            [213, 23, 12],
            [2334, 12, 34],
            [12542, 23, 12]
        ]
    ]).int()
    candidate_ad = torch.tensor([
        [
            [12, 15, 34]
        ],
        [
            [879, 23, 12]
        ],
        [
            [879, 23, 12]
        ]
    ]).int()
    context_features = torch.tensor([
        [2, 4],
        [5, 3],
        [5, 3]
    ]).int()

    m = DIEN()
    r, new_l2_loss = m(
        user_profile_features=user_profile_features,
        user_behaviors=user_behaviors,
        candidate_ad=candidate_ad,
        context_features=context_features,
        return_l2_loss=True
    )
    print(r.shape)
    print(r)


    reg_lambda = 0.01
    # 普通L2 loss的计算方式
    l2_loss = reg_lambda * sum([torch.pow(v, 2).sum() for v in m.parameters()])
    # 优化后的L2 loss的计算方式 --> 必须写到模型内部 --> ne_l2_loss
    new_l2_loss = reg_lambda * new_l2_loss

    # 可视化输出
    mode = torch.jit.trace(
        DIN().cpu().eval(),
        example_inputs=(user_profile_features.cpu(), user_behaviors.cpu(), candidate_ad.cpu(), context_features.cpu())
    )
    torch.jit.save(mode, os.path.join(".", "jit_din.pt"))
    mode = torch.jit.trace(
        BasicDIN().cpu().eval(),
        example_inputs=(user_profile_features.cpu(), user_behaviors.cpu(), candidate_ad.cpu(), context_features.cpu())
    )
    torch.jit.save(mode, os.path.join(".", "jit_din_basic.pt"))


if __name__ == '__main__':
    test()

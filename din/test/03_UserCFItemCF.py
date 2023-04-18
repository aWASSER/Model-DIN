
from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# import sys
#
# sys.path.append("../src")


# 加载数据集
data = Dataset.load_builtin('ml-100k')

# 数据划分
trainset, testset = train_test_split(data, test_size=.25)

# 模型对象构建
_sim_options = {
    'name': 'msd',
    # 'user_based': True,  # UserCF算法
    'user_based': False,  # ItemCF算法
    'min_support': 8  # 控制相似度计算过程中，至少需要多少个共同评分
}
algo = KNNBasic(k=40, min_k=1, sim_options=_sim_options)

# 训练
algo.fit(trainset)

# 预测
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_.est}")
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_.est}")
y_ = algo.predict("gerry", "242")
print(f"未知用户预测评分:{y_.est}")

# 评估
predictions = algo.test(testset)
print(f"RMSE:{accuracy.rmse(predictions)}")

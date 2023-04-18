from surprise import BaselineOnly
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 加载数据集
data = Dataset.load_builtin('ml-100k')

# 数据划分
trainset, testset = train_test_split(data, test_size=.25, random_state=14)

# 模型对象构建
_bsl_options = {
    'method': 'sgd',  # 给定求解方式
    'learning_rate': .00005  # 梯度下降的学习率
}
algo = BaselineOnly(bsl_options=_bsl_options)

# 训练
algo.fit(trainset)

# 预测
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_}")
print(f"预测评分:{y_.est}")
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_}")
print(f"预测评分:{y_.est}")
y_ = algo.predict("gerry", "242")
print(f"预测评分:{y_}")
print(f"预测评分:{y_.est}")

# 评估
predictions = algo.test(testset)
print(f"RMSE:{accuracy.rmse(predictions)}")

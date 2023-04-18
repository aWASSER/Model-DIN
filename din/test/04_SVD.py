
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 加载数据集
data = Dataset.load_builtin('ml-100k')

# 数据划分
trainset, testset = train_test_split(data, test_size=.25)

# 模型对象构建
algo = SVD(n_factors=8, n_epochs=10)

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

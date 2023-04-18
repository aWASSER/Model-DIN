from surprise import NormalPredictor
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 加载数据集
data = Dataset.load_builtin('ml-100k')

# 数据划分(train_test_split方法里面会进行数据的分割构建build_full_trainset)
trainset, testset = train_test_split(data, test_size=.25)

# 模型对象构建
algo = NormalPredictor()

# 训练
algo.fit(trainset)

# 预测
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_.est}")
y_ = algo.predict("196", "242", 3.0)
print(f"预测评分:{y_.est}")

# 评估
predictions = algo.test(testset)
print(f"RMSE:{accuracy.rmse(predictions)}")
print(f"MSE:{accuracy.mse(predictions)}")
print(f"FCP:{accuracy.fcp(predictions)}")

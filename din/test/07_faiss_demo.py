import time

import faiss
import numpy as np

if __name__ == '__main__':
    np.random.seed(10)
    # 模拟一个10w个128维的向量
    xb = np.random.randn(50000, 128).astype('float32')  # 向量库
    x1 = np.random.randn(100, 128).astype('float32')  # 新增商品向量
    x0 = np.random.randn(1, 128).astype('float32')  # 待检索的向量

    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2
    param = 'Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 索引模型是否已经训练好，如果为False，需要单独的调用train方法
    print(type(index))
    index.add(xb)  # 添加
    print(index.is_trained)
    t1 = time.time()
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标
    print(time.time() - t1)
    # 将新增商品添加进去
    index.add(x0)
    index.add(x1)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标

    print("=" * 100)
    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2
    param = 'PCA32,Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    if not index.is_trained:
        # 训练相关参数
        index.train(xb)
    index.add(xb)  # 将向量添加到索引中，进行索引的构建
    print(index.is_trained)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标

    print("=" * 100)
    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2
    param = 'IVF32,Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    t1 = time.time()
    if not index.is_trained:
        # 训练相关参数（单独这里来讲其实就是聚类中心点）
        index.train(xb)
    index.add(xb)  # 将向量添加到索引中，进行索引的构建
    print(time.time() - t1)
    print(index.is_trained)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标
    # 将新增商品添加进去（NOTE:如果添加的新商品向量可能会导致聚类发现变化，需要注意）
    # NOTE: 所以只要这个index需要训练的情况，那么add就不允许动态添加
    index.add(x0)
    index.add(x1)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标

    print("=" * 100)
    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2
    param = 'IVF32,PQ4'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    t1 = time.time()
    if not index.is_trained:
        # 训练相关参数
        index.train(xb)
    index.add(xb)  # 将向量添加到索引中，进行索引的构建
    print(time.time() - t1)
    print(index.is_trained)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标
    # 将新增商品添加进去（NOTE:如果添加的新商品向量可能会导致聚类发现变化，需要注意）
    # NOTE: 所以只要这个index需要训练的情况，那么add就不允许动态添加
    index.add(x0)
    index.add(x1)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标

    print("=" * 100)
    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2
    param = 'HNSW12'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    t1 = time.time()
    index.add(xb)  # 将向量添加到索引中，进行索引的构建
    print(time.time() - t1)
    print(index.is_trained)
    t1 = time.time()
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标
    print(time.time() - t1)
    # NOTE: 所以只要这个index需要训练的情况，那么add就不允许动态添加; 如果index不需要训练，那么add就可以支持
    index.add(x0)
    index.add(x1)
    print(index.search(x0, 5))  # 第一部分是距离， 第二部分就是最相似度的index下标

import numpy as np

from six.moves import range
from six import iteritems


def cosine(n_x, yr, min_support):
    min_sprt = min_support
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    # UserCF x就是user，y就是item
    # ItemCF x就是item，y就是user
    # yr其实是一个字典，key就是y，value就是所有和y相关的x以及其评分
    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj  # 存的是分子
                sqi[xi, xj] += ri ** 2  # 存的是分母
                sqj[xi, xj] += rj ** 2  # 存的是分母

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


def msd(n_x, yr, min_support):
    print("执行修改后的相似度矩阵计算方式!")
    min_sprt = min_support
    sq_diff = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                sq_diff[xi, xj] += (ri - rj) ** 2
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            w = 1.0
            if freq[xi, xj] < 2:
                s = 0
            else:
                s = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)
                if freq[xi, xj] < min_sprt:
                    w = freq[xi, xj] / min_sprt
            sim[xi, xj] = s * w
            sim[xj, xi] = sim[xi, xj]

    return sim

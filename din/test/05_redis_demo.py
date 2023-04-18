import time

import numpy as np
import redis


def t0():
    pool = redis.ConnectionPool(host='121.40.96.93', port=6379, db=0, password='123456')
    r = redis.Redis(connection_pool=pool)
    # r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    # r = redis.Redis(host='121.40.96.93', port=6379, db=0, password='123456')
    # 往redis中插入一条key-value， 而且value类型为String的数据，如果成功返回True字符串，否则返回None，还有可能出现异常
    o = r.set(
        name="name",
        value="我是来及湖南的小明同学",  # value可以是字符串，也可以是任意一个byte字节数组
        ex=60,  # 设置当前加入的key ex秒后过期(删除)
        # px=10000,  # 设置当前加入的key px毫秒后过期；不能和ex同时给定
        nx=False,  # 当设置为True的时候，当却仅当name(key)在redis中不存在的时候，才会插入
        xx=False  # 当设置为True的时候，当却仅当name(key)在redis中存在的时候，才会插入；不能和nx同时给定为True
    )
    print(type(o))
    print(o)
    # get就是获取String类型的key对应的value值
    print("_" * 100)
    o = r.get(name="name")
    print(type(o))
    print(o is None)
    if o is not None:
        print(str(o, encoding='utf-8'))
    # 设置key对应数据过期时间
    r.expire(name='user:1001', time=10000)  # 过期秒数
    r.pexpire(name='name', time=10000)  # 过期毫秒数
    print(f"剩余多少过期时间:{r.ttl(name='name')}s")

    print("_" * 100)
    # 让hash结构的value中添加数据
    o = r.hset(name="user:1001", key="id", value="1001")
    print(o)
    o = r.hset(name="user:1001", key="name", value="小明")
    print(o)
    o = r.hset(name="user:1001", mapping={
        "age": 17,
        "sex": "男",
        "address": "上海"
    })
    print(o)
    # 建议添加数据的写法（从redis理论上来讲）
    o = r.hmset(name="user:1002", mapping={
        "id": "1002",
        "name": "小明1111",
        "age": 17,
        "sex": "男",
        "address": "上海",
        "address2": "北京"
    })
    print(o)
    # 一次性获取所有field的value数据 --> hgetall返回结果是一个字典，如果key不存在，那么返回字典为空{}
    o = r.hgetall(name='user:1001')
    print(type(o))
    print(o is None)
    print({str(k, encoding='utf-8'): str(o[k], encoding='utf-8') for k in o})
    # 针对我们需要的field可以采用(单一field)，只要key(name)或者field不存在，返回就是None
    o = r.hget(name='user:1001', key='address')
    print(type(o))
    print(o is None)
    if o is not None:
        print(str(o, encoding='utf-8'))
    # 针对我们需要的field可以采用(多个field)，返回一个list列表，如果field存在对应位置为bytes，如果field不存在，那么对应位置为None
    o = r.hmget(name='user:1002', keys=['name', 'age', 'address', 'address2'])
    print(type(o))
    print(o is None)
    print([v if v is None else str(v, encoding='utf-8') for v in o])

    r.close()  # 当不使用的时候，记得close关闭


def t1():
    pool = redis.ConnectionPool(host='121.40.96.93', port=6379, db=0, password='123456', max_connections=100)
    with redis.Redis(connection_pool=pool) as r:
        _t1 = time.time()
        n = 10000
        for i in range(n):
            r.set(name=f'k:{i}', value=i, ex=220)
        print(time.time() - _t1)


def t2():
    pool = redis.ConnectionPool(host='121.40.96.93', port=6379, db=0, password='123456', max_connections=100)
    with redis.Redis(connection_pool=pool) as r:
        _t1 = time.time()
        n = 10000
        _pipeline = r.pipeline()
        for i in range(n):
            _pipeline.set(name=f'l:{i}', value=i, ex=220)
        _pipeline.execute()
        print(time.time() - _t1)


def t3():
    pool = redis.ConnectionPool(host='121.40.96.93', port=6379, db=0, password='123456', max_connections=100)
    with redis.Redis(connection_pool=pool) as r:
        _t1 = time.time()
        _pipeline = r.pipeline()
        _pipeline.get('l:99')
        _pipeline.get('k:78')
        _pipeline.get('n:78')
        _pipeline.hmget('user:1001', ['name', 'address'])
        _pipeline.hmget('user:1002', ['name', 'address'])
        _pipeline.hmget('user:1003', ['name', 'address'])
        _r = _pipeline.execute()
        print(len(_r))
        print(_r)
        print(time.time() - _t1)

        # 召回(当前用户各个需要的召回策略对应的推荐商品id列表, 当前用户对应的各个向量召回模型对应的向量, 热门商品列表, 新品商品列表)
        _pipeline = r.pipeline()
        _pipeline.hmget('recall:10001', ['usercf', 'itemcf', 'mf'])
        _pipeline.get('fm:10001')
        _pipeline.get('dssm:10001')
        _pipeline.get('hot_product')
        _pipeline.get('new_product')
        _r = _pipeline.execute()


if __name__ == '__main__':
    # t0()
    # t1()
    # t2()
    # t1()
    # t2()
    t3()

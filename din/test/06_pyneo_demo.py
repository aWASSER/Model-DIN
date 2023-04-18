import py2neo

if __name__ == '__main__':
    graph = py2neo.Graph("bolt://121.40.96.93:7687", auth = ("neo4j", "123456789"))
    rs = graph.run("MATCH (n:Person) RETURN n")
    for r in rs:
        r = r.get('n')  # n就是return里面的值
        name = r.get('name')
        birthday = r.get('birthday')
        birthplace = r.get('birthplace')
        print(name, birthday, birthplace)

    rs = graph.run("MERGE (p:Person {name:'小明'}) ON CREATE SET p.name='小明'")
    print(rs)
    rs = graph.run("MERGE (p:Person {name:'小红'}) ON CREATE SET p.name='小红'")
    print(rs)

    # 匹配一个节点
    node = graph.nodes.match('Person', name='周星驰').first()
    print(node)
    print(node.get("name"))
    node2 = graph.nodes.match('Movie', name='戏剧之王').first()
    rs = graph.match(nodes=[node,node2])
    print("=" * 100)
    for r in rs:
        print(r)

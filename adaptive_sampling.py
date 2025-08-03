import networkx as nx

def analyze_graph_properties(graph):
    """分析图的基本属性：关系类型数量和密度"""
    # 计算节点和边的数量
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    
    # 计算图密度
    density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    # 计算关系类型数量
    relation_types = set()
    for _, _, data in graph.edges(data=True):
        if 'type' in data:
            relation_types.add(data['type'])
    n_relations = len(relation_types)
    
    return n_relations, density

def select_best_sampling_method(graph):
    """根据图的特征选择最佳子图采样方法"""
    # 分析图特征
    n_relations, density = analyze_graph_properties(graph)
    
    # 设置阈值
    RELATION_THRESHOLD = 20  # 关系类型数量阈值
    DENSITY_THRESHOLD = 0.001  # 密度阈值
    
    if density < DENSITY_THRESHOLD:
        if n_relations < RELATION_THRESHOLD:
            # 稀疏 + 关系类型少 => 使用k-hop子图
            return "kHop", n_relations, density
        else:
            # 稀疏 + 关系类型多 => 用概率采样选重要子图
            return "probability", n_relations, density
    else:
        if n_relations < RELATION_THRESHOLD:
            # 密集 + 关系类型少 => 用随机游走获取丰富局部路径
            return "randomWalk", n_relations, density
        else:
            # 密集 + 关系多：建议使用概率采样（防爆炸）
            return "probability", n_relations, density
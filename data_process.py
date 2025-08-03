import pandas as pd
import numpy as np
import os
import csv
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
import torch
from rdkit.Chem import MolFromSmiles
from multiprocessing import Pool
import networkx as nx
from randomWalk import Node2vec
from torch_geometric.utils import subgraph, degree, get_laplacian
from utils import *
from torch import Tensor
import numpy as np
from torch_geometric.data import Data
from gensim.models import Word2Vec
from adaptive_sampling import select_best_sampling_method
import tqdm


e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

import csv

def load_id_mapping(mapping_path):
    numid_to_dbid = {}
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            numid_to_dbid[str(row['Drug idxs'])] = row['Drug IDs']
    return numid_to_dbid

def smile_to_graph(datapath, ligands):
    smile_graph = {}
    paths = datapath + "/mol_sp.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s]["graph_info"][6]) if max(smile_graph[s]["graph_info"][6]) > max_rel else max_rel
            max_degree = smile_graph[s]["graph_info"][7] if smile_graph[s]["graph_info"][7] > max_degree else max_degree

        return smile_graph, max_rel, max_degree

    smiles_max_node_degree = []
    num_rel_mol_update = 0

    # 遍历每个药物的SMILES序列
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))  ##还是smiles序列
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(lg)

        if c_size == 0: ##证明这个药物只由一个atom组成，这种的不考虑
            continue

        # 更新分子图的最大关系数量
        if max(s_value) > num_rel_mol_update:
            num_rel_mol_update = max(s_value)

        # 将图数据转换为 PyTorch Geometric 的 Data 对象
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        data.rel_index = torch.tensor(rel_index, dtype=torch.long)  # 添加其他属性
        data.s_edge_index = torch.tensor(s_edge_index, dtype=torch.long)
        data.s_value = torch.tensor(s_value, dtype=torch.float)
        data.s_rel = torch.tensor(s_rel, dtype=torch.long)
        data.deg = deg

        # 保存分子图信息以及原始 SMILES 序列，转换 Data 为字典
        smile_graph[d] = {
            "graph_info": data_to_dict(data),
            "smiles": ligands[d]
        }
        smiles_max_node_degree.append(deg)

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, num_rel_mol_update, max(smiles_max_node_degree)


def data_to_dict(data):
    """
    将 PyTorch Geometric Data 对象转换为元组
    返回: (c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg)
    """
    c_size = data.x.size(0)
    features = data.x.tolist()
    edge_index = data.edge_index.tolist()
    rel_index = data.rel_index.tolist()
    s_edge_index = data.s_edge_index.tolist()
    s_value = data.s_value.tolist()
    s_rel = data.s_rel.tolist()
    deg = data.deg.tolist() if isinstance(data.deg, (np.ndarray, torch.Tensor)) else data.deg
    
    # 返回元组，与参考代码格式一致
    return (c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg)


# mol smile to mol graph edge index
def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2]
    mol_rel_index = mol_index[:,2]

    ##在这个位置应该计算的是最短路径
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value
    s_rel[np.where(s_value == 1)] = mol_rel_index  ##将直接相连的关
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    ##c_size:原子的个数
    ##features:每个原子的特征 c_size * 67
    ##edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist(), s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(), max(degrees)

def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)


def read_interactions(path, drug_dict):
    interactions = []
    all_drug_in_ddi = []
    positive_drug_inter_dict = {}
    positive_num = 0
    negative_num = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            drug1_id, drug2_id, rel, label = line.strip().split(" ")[:4]
            if drug1_id in drug_dict and drug2_id in drug_dict:
                all_drug_in_ddi.append(drug1_id)
                all_drug_in_ddi.append(drug2_id)
                if float(label) > 0:
                    positive_num += 1
                else:
                    negative_num += 1
                if drug1_id in positive_drug_inter_dict:
                    if drug2_id not in positive_drug_inter_dict[drug1_id]:
                        positive_drug_inter_dict[drug1_id].append(drug2_id)
                        interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
                else:
                    positive_drug_inter_dict[drug1_id] = [drug2_id]
                    interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
        f.close()

    print(positive_num)
    print(negative_num)

    assert negative_num == positive_num

    return np.array(interactions, dtype=int), set(all_drug_in_ddi)

def read_network(path):

    edge_index = []
    rel_index = []

    flag = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            else:
                flag += 1
                head, tail, rel = line.strip().split(" ")[:3]
                edge_index.append([int(head), int(tail)])
                rel_index.append(int(rel))

        f.close()
    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(len(list(set(rel_index))))

    return num_node, edge_index, rel_index, num_rel

def load_protein_embeddings(path="protein_embeddings.pt"):
    """
    加载 ESM-2 生成的蛋白质嵌入
    参数:
        path: ESM-2 嵌入文件的路径
    返回:
        embedding_dict: {protein_id: embedding_tensor} 的字典
    """
    print(f"Loading ESM-2 embeddings from {path}")
    
    data = torch.load(path)
    embeddings = data['embeddings']  # shape: [num_proteins, 1280]
    descriptions = data['descriptions']  # 蛋白质序列描述列表
    
    # 创建 description 到 embedding 的映射
    embedding_dict = {}
    for desc, emb in zip(descriptions, embeddings):
        # 从描述中提取蛋白质 ID
        # 例如: ">sp|P35228|NOS2_HUMAN" -> "P35228"
        try:
            protein_id = desc.split('|')[1]
            embedding_dict[protein_id] = emb
        except IndexError:
            print(f"Warning: Could not parse protein ID from description: {desc}")
            continue
    
    print(f"Loaded {len(embedding_dict)} protein embeddings")
    return embedding_dict

def read_targets(target_path, esm2_embeddings=None):
    """
    读取靶点数据并使用 ESM-2 嵌入
    """
    print(f"Reading target data from {target_path}")
    
    # 读取 CSV 文件
    df = pd.read_csv(target_path)
    
    # 初始化结果字典和计数器
    drug_target_map = {}
    embedding_dim = 1280
    drugs_with_embeddings = 0
    total_drugs = 0
    
    # 按药物ID分组处理靶点
    grouped = df.groupby('Drug IDs')
    
    for drug_id, group in grouped:
        total_drugs += 1
        # 收集该药物所有靶点的嵌入
        target_embeddings = []
        
        # 获取该药物的所有UniProt IDs
        uniprot_ids = group['UniProt ID'].dropna().unique()
        
        for uniprot_id in uniprot_ids:
            if uniprot_id in esm2_embeddings:
                target_embeddings.append(esm2_embeddings[uniprot_id])
        
        if target_embeddings:
            # 如果找到靶点嵌入，取平均值
            drug_target_map[str(drug_id)] = torch.stack(target_embeddings).mean(dim=0)
            drugs_with_embeddings += 1
        else:
            # 如果没有找到靶点嵌入，使用零向量
            drug_target_map[str(drug_id)] = torch.zeros(embedding_dim)
    
    print(f"\nTarget Processing Statistics:")
    print(f"Total proteins in ESM-2 embeddings: {len(esm2_embeddings)}")
    print(f"Total drugs processed: {total_drugs}")
    print(f"Drugs with at least one ESM-2 embedding: {drugs_with_embeddings}")
    print(f"Drugs without any ESM-2 embeddings: {total_drugs - drugs_with_embeddings}")
    print(f"Coverage rate: {(drugs_with_embeddings/total_drugs)*100:.2f}%")
    
    return drug_target_map

def read_smiles(path):
    print("Read " + path + "!")
    out = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            id, sequence = line.strip().split("\t")
            if id not in out:
                out[id] = sequence  ## 这里面的id是str类型

    return out


def generate_node_subgraphs(dataset, drug_id, network_edge_index, network_rel_index, num_rel, args):
    """生成节点子图的主函数，根据数据集特征自动选择最佳采样方法"""

    method = args.extractor
    edge_index = torch.from_numpy(np.array(network_edge_index).T) ##[2, num_edges]
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row),0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index),1)

    paths = "data/" + str(dataset) + "/" + str(method) + "/"

    if not os.path.exists(paths):
        os.mkdir(paths)

    # 如果是自适应方法，使用adaptive_sampling来决定具体的采样方法
    if method == "adaptive":
        # 直接从rel_index获取关系类型数量，无需构建完整图
        unique_relations = set(rel_index.numpy().tolist())
        n_relations = len(unique_relations)
        
        # 构建NetworkX图用于分析密度
        g = nx.DiGraph()
        g.name = dataset  # 设置图名称
        
        print(f"正在分析数据集特征...")
        print(f"关系类型总数: {n_relations}")
        
        # 使用所有边构建图
        edge_list = undirected_edge_index.transpose(1, 0).numpy().tolist()
        print(f"构建图用于密度分析 (添加 {len(edge_list)} 条边)...")
        
        # 使用tqdm添加进度条
        for i, (src, dst) in tqdm.tqdm(enumerate(edge_list), total=len(edge_list), desc="构建图"):
            g.add_edge(src, dst)
        
        # 计算图密度
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        # 根据关系类型数量和密度选择最佳方法
        RELATION_THRESHOLD = 20  # 关系类型数量阈值
        DENSITY_THRESHOLD = 0.001  # 密度阈值
        
        if n_relations < RELATION_THRESHOLD and density > DENSITY_THRESHOLD:
            actual_method = "randomWalk"
        else:
            actual_method = "probability"
        
        print(f"原始边数量: {len(network_edge_index)}")
        print(f"无向图边数量: {undirected_edge_index.shape[1]}")
        print(f"图分析结果: 节点数={n_nodes}, 边数={n_edges}, 关系类型数量={n_relations}, 图密度={density:.6f}")
        print(f"根据图特征选择最佳子图提取方法: {actual_method}")
    else:
        # 非自适应方法，直接使用指定的方法
        actual_method = method
    
    # 根据选择的方法调用相应的提取器
    if actual_method == "khop-subtree":
        subgraphs, max_degree, max_rel_num = subtreeExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                             fixed_num=args.fixed_num, khop=args.khop)
    elif actual_method == "probability":
        pagerank_paths = "data/" + str(dataset) + "/" + str(actual_method) + "/pageRank.json"
        subgraphs, max_degree, max_rel_num = probExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                          fixed_num=args.fixed_num, pagerank_path=pagerank_paths)
    elif actual_method == "randomWalk":
        subgraphs, max_degree, max_rel_num = rwExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                        sub_num=args.graph_fixed_num, length=args.fixed_num)
    
    # 统计生成子图的药物数量
    drugs_with_subgraph = list(subgraphs.keys())
    print(f"Successfully generated subgraphs for {len(drugs_with_subgraph)}/{len(drug_id)} drugs")
    
    return subgraphs, max_degree, max_rel_num

def subtreeExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, khop):

    all_degree = []
    num_rel_update = []
    subgraphs = {}

    json_path = shortest_paths + "subtree_fixed_" + str(fixed_num) + "_hop_" + str(khop) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

        return subgraphs, max_degree, max_rel;

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    for d in drug_id:
        subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(int(d), khop, edge_index, undirected_rel_index, fixed_num, relabel_nodes=True)  ##subset是所有集合的节点，mapping指示的是center node是哪个
        row, col = sub_edge_index
        all_degree.append(torch.max(degree(col)).item())

        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1,0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()
        node_idx = subset.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(np.max(s_rel))

        subgraphs[d] = node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    ## subset: LongTensor
    ## edge_index: LongTensor
    ## subgraph_rel: Tensor
    return subgraphs, max(all_degree), max(num_rel_update)


def probExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, pagerank_path):

    json_path = shortest_paths + "prob_fix_" + str(fixed_num) + "_sp.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                subgraphs = json.load(f)
                max_rel = 0
                max_degree = 0
                for s in subgraphs.keys():
                    # 先检查子图是否为空，再调用max()
                    if subgraphs[s][6]:
                        max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                    if subgraphs[s][7]:
                        max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

            return subgraphs, max_degree, max_rel
        except Exception as e:
            print(f"Error loading JSON from {json_path}: {e}")
            return {}, 0, 0  # 确保返回空字典和默认值

    g = nx.DiGraph()
    g.add_edges_from(edge_index.transpose(1, 0).tolist())

    if not os.path.exists(pagerank_path):
        pagerank = np.array(google_matrix(g), dtype='float64')
        page_dict = {}
        for d in drug_id:
            page_dict[d] = list(pagerank[list(g.nodes()).index(int(d))])
        with open(pagerank_path, 'w') as f:
            json.dump(page_dict, f)
    else:
        with open(pagerank_path, 'r') as f:
            page_dict = json.load(f)

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []  # 确保num_rel_update始终初始化为空列表
    max_degree = []
    subgraphs = {}

    for d in drug_id:
        subsets = [int(d)]

        # 检查 PageRank 的概率分布
        if np.sum(page_dict[d]) == 0:
            print(f"Warning: PageRank for drug {d} is all zeros. Skipping...")
            continue

        # 归一化概率分布
        probs = np.array(page_dict[d], dtype='float64')
        probs = probs / np.sum(probs)  # 确保概率和为1

        try:
            neighbors = np.random.choice(
                a=list(g.nodes()),
                size=fixed_num,
                replace=False,
                p=probs)
        except ValueError as e:
            print(f"Warning: Error in probability sampling for drug {d}: {e}")
            continue

        subsets.extend(neighbors)
        subsets = list(set(subsets))

        # 检查邻居采样结果
        if len(subsets) == 1:  # 只有药物节点自己
            print(f"Warning: No neighbors sampled for drug {d}. Skipping...")
            continue

        mapping_list = [False for _ in subsets]
        mapping_idx = subsets.index(int(d))
        mapping_list[mapping_idx] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index

        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        if not new_s_edge_index or not new_s_value or not new_s_rel:
            print(f"Warning: Empty subgraph generated for drug {d}. Skipping...")
            continue

        num_rel_update.append(int(np.max(s_rel)) if s_rel else 0)  # 处理空s_rel的情况，默认值为0
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    # 如果num_rel_update为空，设置默认值为0
    return subgraphs, max(max_degree) if max_degree else 0, max(num_rel_update) if num_rel_update else 0


def rwExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, sub_num, length):

    json_path = shortest_paths + "rw_num_" + str(sub_num) + "_length_" + str(length) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel;

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_index.transpose(1,0).numpy().tolist())
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        subsets = Node2vec(start_nodes=[int(d)], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() ##返回一个list
        mapping_id = subsets.index(int(d))
        mapping_list = [False for _ in range(len((subsets)))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index
        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)


def k_hop_subgraph(node_idx, num_hops, edge_index, rel_index, fixed_num, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    np.random.seed(42)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        #print(col[edge_mask].shape)
        if fixed_num == None:
            subsets.append(col[edge_mask])
        elif col[edge_mask].size(0) > fixed_num:
            neighbors = np.random.choice(a=col[edge_mask].numpy(), size=fixed_num, replace=False)
            subsets.append(torch.LongTensor(neighbors))
        else:
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    #print(subset)

    rel_index = rel_index[edge_mask] if rel_index is not None else None


    mapping_mask = [False for _ in range(len(subset))]
    mapping_mask[inv] = True


    return subset, edge_index, rel_index, mapping_mask


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def min_max(data:list):
    min_value = min(data)
    max_value = max(data)

    norm_data = []
    for d in data:
        norm_data.append((d-min_value+0.00001)/(max_value-min_value))

    return [d/sum(norm_data) for d in norm_data]


def google_matrix(
    G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None
):

    if nodelist is None:
        nodelist = list(G)

    M = np.asmatrix(nx.to_numpy_array(G, nodelist=nodelist, weight=weight), dtype='float64')
    N = len(G)
    if N == 0:
        return M

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N).astype('float64')
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype="float64")
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()


    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype='float64')
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1).astype('float64')  # Normalize rows to sum to 1

    return np.multiply(alpha, M, dtype='float64') + np.multiply(1 - alpha, p, dtype='float64')
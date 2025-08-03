import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch, Dataset
from torch_geometric import data as DATA
import torch
import numpy as np


class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, target_dict=None, numid_to_drugid=None):
        super(DTADataset, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 添加设备
        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.target_dict = target_dict  # 新增靶点数据
        self.numid_to_drugid = numid_to_drugid

    def read_drug_info(self, drug_id, labels):
        """读取药物信息"""
        try:
            # 从 smile_graph 中提取分子图信息和 SMILES 序列
            smile_data = self.smile_graph[str(drug_id)]
            c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = smile_data["graph_info"]
            smiles_sequence = smile_data["smiles"]

            # 检查 edge_index 的维度
            edge_index = np.array(edge_index)
            if edge_index.ndim == 1:
                # 如果是一维数组，重塑为 2xN 形式
                edge_index = edge_index.reshape(-1, 2).T
            elif edge_index.shape[0] != 2:
                # 如果不是 2xN 形式，转置
                edge_index = edge_index.T

            # 分子图数据
            data_mol = DATA.Data(x=torch.Tensor(np.array(features)).to(self.device),
                            edge_index=torch.LongTensor(edge_index).to(self.device),
                            y=torch.LongTensor([labels]).to(self.device),
                            rel_index=torch.Tensor(np.array(rel_index, dtype=int)).to(self.device),
                            sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0).to(self.device),
                            sp_value=torch.Tensor(np.array(sp_value, dtype=int)).to(self.device)    ,
                            sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int)).to(self.device))
            data_mol.__setitem__('c_size', torch.LongTensor([c_size]).to(self.device))

            # 获取子图数据
            subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

            # 检查 subgraph_edge_index 的维度
            subgraph_edge_index = np.array(subgraph_edge_index)
            if subgraph_edge_index.ndim == 1:
                # 如果是一维数组，重塑为 2xN 形式
                subgraph_edge_index = subgraph_edge_index.reshape(-1, 2).T
            elif subgraph_edge_index.shape[0] != 2:
                # 如果不是 2xN 形式，转置
                subgraph_edge_index = subgraph_edge_index.T

            # 创建子图数据对象
            data_graph = DATA.Data(x=torch.LongTensor(subset).to(self.device),
                            edge_index=torch.LongTensor(subgraph_edge_index).to(self.device),
                            y=torch.LongTensor([labels]).to(self.device),
                            id=torch.LongTensor(np.array(mapping_id, dtype=bool)).to(self.device),
                            rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)).to(self.device),
                            sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0).to(self.device),
                            sp_value=torch.Tensor(np.array(s_value, dtype=int)).to(self.device) ,
                            sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int)).to(self.device))

            # 获取靶点特征
            # target_features = self.target_dict.get(str(drug_id), torch.zeros(1280)).to(self.device)
            dbid = self.numid_to_drugid.get(str(drug_id)) if self.numid_to_drugid else None
            if dbid and dbid in self.target_dict:
                target_features = self.target_dict[dbid].to(self.device)
            else:
                target_features = torch.zeros(1280).to(self.device)

            return data_mol, data_graph, target_features, smiles_sequence

        except Exception as e:
            print(f"Error processing drug {drug_id}: {str(e)}")
            return None

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        """
        获取数据集中的一项
        返回两个药物的所有相关信息
        """
        # 检查 idx 范围
        assert idx < len(self.drug_ID), f"Index {idx} out of range"

        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        labels = int(self.labels[idx])

        # 主要问题药物列表（合并之前的和新的）
        problem_drugs = {
            8311, 3706, 42351, 38691, 52489, 45057, 31861, 
            77571, 39537, 74321, 89629, 25228, 89482,
            39593, 37755, 32553, 32826, 54374, 41843  # 添加新发现的问题药物
        }
        
        if drug1_id in problem_drugs or drug2_id in problem_drugs:
            return self.__getitem__((idx + 1) % len(self))  # 直接跳到下一个样本

        try:
            # 读取两个药物的信息
            drug1_mol, drug1_subgraph, drug1_target, drug1_smiles = self.read_drug_info(drug1_id, labels)
            drug2_mol, drug2_subgraph, drug2_target, drug2_smiles = self.read_drug_info(drug2_id, labels)

            # 检查数据是否完整
            if None in (drug1_mol, drug1_subgraph, drug1_target, drug1_smiles,
                    drug2_mol, drug2_subgraph, drug2_target, drug2_smiles):
                return self.__getitem__((idx + 1) % len(self))

            return drug1_mol, drug1_subgraph, drug1_target, drug1_smiles, \
                drug2_mol, drug2_subgraph, drug2_target, drug2_smiles

        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

def smiles_to_tensor(smiles_list):
    tokenized = [[ord(char) for char in smiles] for smiles in smiles_list]
    max_len = max(len(seq) for seq in tokenized)
    tensor = torch.zeros((len(tokenized), max_len), dtype=torch.long)
    for i, seq in enumerate(tokenized):
        tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return tensor


def collate(data_list):
    """
    将数据列表组合成批次
    处理 None 值并确保数据完整性
    """
    try:
        # 过滤掉 None 值的数据项
        data_list = [data for data in data_list if data is not None]

        # 如果 data_list 为空，抛出错误
        if len(data_list) == 0:
            raise ValueError("[ERROR] All data entries are None! Cannot create batch.")
        
        # 转换图数据为 Batch 类型
        batchA = Batch.from_data_list([data[0] for data in data_list])  # 药物1 分子图
        batchB = Batch.from_data_list([data[1] for data in data_list])  # 药物1 知识图
        batchC = Batch.from_data_list([data[4] for data in data_list])  # 药物2 分子图
        batchD = Batch.from_data_list([data[5] for data in data_list])  # 药物2 知识图

        # 转换靶点特征为张量
        target1 = torch.stack([data[2] for data in data_list])
        target2 = torch.stack([data[6] for data in data_list])

        # 转换 SMILES 数据为张量
        smiles1 = smiles_to_tensor([data[3] for data in data_list])
        smiles2 = smiles_to_tensor([data[7] for data in data_list])

    except Exception as e:
        print(f"[ERROR] Error in collate function: {e}")
        raise

    return batchA, batchB, target1, smiles1, batchC, batchD, target2, smiles2
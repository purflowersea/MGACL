import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from .GraphTransformer import GraphTransformer
from fusion import CrossModalFusion
import os
from torch_geometric.data import Batch
from torch_geometric.data import Data
import ast
from graph_contrastive import GraphContrastiveLearning



class NodeFeatures(torch.nn.Module):
    def __init__(self, degree, feature_num, embedding_dim, layer=2, type='graph'):
        super(NodeFeatures, self).__init__()

        if type == 'graph': ##代表有feature num
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)  ##将度的值映射成embedding
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):

        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())

        return node_feature

class MHGCL(torch.nn.Module):
    def __init__(self, args, max_layer = 6, num_features_drug = 78, num_nodes = 200, num_relations_mol = 10, num_relations_graph = 10, output_dim=64, max_degree_graph=100, max_degree_node=100, sub_coeff = 0.2, mi_coeff = 0.5, dropout=0.2, device = 'cuda', num_heads = 4, fusion_temperature=1.0, contrastive_weight=0.1, mol_ratio=0.5, kg_ratio=0.5):
        super(MHGCL, self).__init__()

        print("MHGCL Loaded")
        self.device = device

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout
        self.output_dim = output_dim # 添加这一行以便后续使用

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, feature_num=num_features_drug, embedding_dim=output_dim, type='graph')
        self.drug_node_feature = NodeFeatures(degree=max_degree_node, feature_num=num_nodes, embedding_dim=output_dim, type='node')

        ##学习的模块
        self.mol_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_mol, dropout= dropout, type='graph')
        self.node_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_graph, dropout=dropout, type='node')
        
        # 靶点嵌入模块
        self.target_encoder = nn.Linear(1280, output_dim) 

        # SMILES 编码器模块
        self.smiles_encoder = SMILESSequenceEncoder(vocab_size=128, embed_dim=output_dim, hidden_dim=output_dim)

        # 对比学习配置 
        self.contrastive_weight = contrastive_weight     # 对比学习权重
        self.mol_ratio = mol_ratio
        self.kg_ratio = kg_ratio

        # 计算实际权重
        self.mol_contrastive_weight = contrastive_weight * mol_ratio
        self.kg_contrastive_weight = contrastive_weight * kg_ratio
        
        # 显示对比学习配置
        print(f"Contrastive Learning Configuration:")
        print(f"  - Molecular Graph Contrastive: Enabled (Weight: {self.mol_contrastive_weight:.4f})")
        print(f"  - Knowledge Graph Contrastive: Enabled (Weight: {self.kg_contrastive_weight:.4f})")
        print(f"  - Total Contrastive Weight: {self.contrastive_weight}")
        
        self.mol_contrastive = GraphContrastiveLearning(
                embed_dim=output_dim,
                temperature=0.1,        # 温度参数
                aug_ratio=0.2          # 分子图增强比例
            )
        
        self.kg_contrastive = GraphContrastiveLearning(
                embed_dim=output_dim,
                temperature=0.1,        # 温度参数
                aug_ratio=0.3          # 知识图谱增强比例
            )

        # 跨模态自注意力融合模块
        self.cross_modal_fusion = CrossModalFusion(
            dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            fusion_temperature=fusion_temperature,
            knowledge_weight=args.knowledge_weight,
            molecular_weight=args.molecular_weight,
            smiles_weight=args.smiles_weight,
            target_weight=args.target_weight,
            fixed_weights=args.fixed_weights
            )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()


    def to(self, device):

        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)

        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)

        # self.fc1.to(device)
        self.cross_modal_fusion.to(device) 
        self.fc2.to(device)

        self.disc.to(device)
        self.b_xent.to(device)

        self.smiles_encoder.to(device)  
        self.target_encoder.to(device)  
    
        self.mol_contrastive.to(device)
        self.kg_contrastive.to(device)

        self.device = device

        return self

    def reset_parameters(self):

        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()

        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()


    def forward(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph, drug1_smiles, drug2_smiles, drug1_target, drug2_target):

        # 处理 drug2_mol 的数据类型
        if isinstance(drug2_mol, list):
            print("[INFO] drug2_mol is a list, converting to Batch...")
            drug2_mol = Batch.from_data_list(drug2_mol)
        elif isinstance(drug2_mol, torch.Tensor):
            print("[ERROR] drug2_mol is a Tensor, expected Data or Batch. Check collate function.")
            raise TypeError(f"drug2_mol should be Data or Batch, but got {type(drug2_mol)}")

        # 确保 drug2_mol 是 Data 或 Batch
        assert isinstance(drug2_mol, (Data, Batch)), f"drug2_mol should be Data or Batch, got {type(drug2_mol)}"

        device = self.device  # 确保所有张量在同一设备上

        # 特征提取和表示学习
        # **1. 提取分子图 & 知识图 特征**
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_embedding, drug1_sub_embedding, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        drug2_node_embedding, drug2_sub_embedding, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        # 计算对比损失 (仅在训练模式下)
        mol_contrast_loss = 0.0
        kg_contrast_loss = 0.0

        if self.training:
            # 分子图对比学习
            try:
                mol1_view1, mol1_view2 = self.mol_contrastive.augment(drug1_mol)
                mol2_view1, mol2_view2 = self.mol_contrastive.augment(drug2_mol)
                    
                # 为增强视图提取特征
                mol1_view1_feat = self.mol_atom_feature(mol1_view1)
                mol1_view2_feat = self.mol_atom_feature(mol1_view2)
                mol2_view1_feat = self.mol_atom_feature(mol2_view1)
                mol2_view2_feat = self.mol_atom_feature(mol2_view2)
                    
                # 获取增强视图的表示
                mol1_view1_emb, _, _ = self.mol_representation_learning(mol1_view1_feat, mol1_view1)
                mol1_view2_emb, _, _ = self.mol_representation_learning(mol1_view2_feat, mol1_view2)
                mol2_view1_emb, _, _ = self.mol_representation_learning(mol2_view1_feat, mol2_view1)
                mol2_view2_emb, _, _ = self.mol_representation_learning(mol2_view2_feat, mol2_view2)
                    
                # 计算分子图对比损失
                mol1_cl_loss = self.mol_contrastive.contrastive_loss(mol1_view1_emb, mol1_view2_emb)
                mol2_cl_loss = self.mol_contrastive.contrastive_loss(mol2_view1_emb, mol2_view2_emb)
                    
                # 累加分子图对比损失
                mol_contrast_loss = mol1_cl_loss + mol2_cl_loss
            except Exception as e:
                print(f"Error in molecular contrastive learning: {e}")
        
            # 知识图谱对比学习
            try:
                kg1_view1, kg1_view2 = self.kg_contrastive.augment(drug1_subgraph)
                kg2_view1, kg2_view2 = self.kg_contrastive.augment(drug2_subgraph)
                    
                # 为知识图谱增强视图提取特征
                kg1_view1_feat = self.drug_node_feature(kg1_view1)
                kg1_view2_feat = self.drug_node_feature(kg1_view2)
                kg2_view1_feat = self.drug_node_feature(kg2_view1)
                kg2_view2_feat = self.drug_node_feature(kg2_view2)
                    
                # 获取知识图谱增强视图的表示
                kg1_view1_emb, _, _ = self.node_representation_learning(kg1_view1_feat, kg1_view1)
                kg1_view2_emb, _, _ = self.node_representation_learning(kg1_view2_feat, kg1_view2)
                kg2_view1_emb, _, _ = self.node_representation_learning(kg2_view1_feat, kg2_view1)
                kg2_view2_emb, _, _ = self.node_representation_learning(kg2_view2_feat, kg2_view2)
                    
                # 计算知识图谱对比损失
                kg1_cl_loss = self.kg_contrastive.contrastive_loss(kg1_view1_emb, kg1_view2_emb)
                kg2_cl_loss = self.kg_contrastive.contrastive_loss(kg2_view1_emb, kg2_view2_emb)
                    
                # 累加知识图谱对比损失
                kg_contrast_loss = kg1_cl_loss + kg2_cl_loss
            except Exception as e:
                print(f"Error in knowledge graph contrastive learning: {e}")

        # **2. 计算 SMILES & 靶点特征**
        drug1_smiles_embedding = self.smiles_encoder(drug1_smiles)
        drug2_smiles_embedding = self.smiles_encoder(drug2_smiles)

        # **手动移动 target 到 GPU**
        drug1_target = drug1_target.to(device)
        drug2_target = drug2_target.to(device)
        drug1_target_embedding = self.target_encoder(drug1_target)  # 线性映射
        drug2_target_embedding = self.target_encoder(drug2_target)
        
        # 跨模态自注意力融合
        drug1_embedding = self.cross_modal_fusion(
            drug1_node_embedding,     # 知识图谱特征
            mol1_graph_embedding,     # 分子图特征
            drug1_smiles_embedding,   # SMILES特征
            drug1_target_embedding    # 靶点特征
        )
        
        drug2_embedding = self.cross_modal_fusion(
            drug2_node_embedding,
            mol2_graph_embedding,
            drug2_smiles_embedding,
            drug2_target_embedding
        )
        
        # **4. 计算交互分数**
        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))
        
        # **5. 计算 MI 损失**
        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(self.MI(drug2_embedding, mol2_atom_embedding))
        loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(self.MI(drug2_embedding, drug2_sub_embedding))
        
        # **6. 计算最终损失**
        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        loss = loss_label + self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d
        # 应用对比学习损失 - 使用分别定义的权重
        if self.training:
            if mol_contrast_loss > 0:
                loss += self.mol_contrastive_weight * mol_contrast_loss
            if kg_contrast_loss > 0:
                loss += self.kg_contrastive_weight * kg_contrast_loss

        return torch.exp(predicts_drug)[:, 1], loss


    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub)) ##pos
            c_1_list.append(c_1.expand_as(sub)) ##neg
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):

        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1,-1]), lbl.view([1, -1]))

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path

# SMILES序列嵌入模块
class SMILESSequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SMILESSequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, smiles_tokens):
        embed = self.embedding(smiles_tokens)
        _, (hidden, _) = self.lstm(embed)
        return hidden.squeeze(0)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits

def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
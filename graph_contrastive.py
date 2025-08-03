import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Batch, Data
import copy

class GraphContrastiveLearning(nn.Module):
    """
    用于药物图数据的对比学习模块
    可作为插件无缝集成到现有GraphTransformer中
    """
    def __init__(self, embed_dim, temperature=0.1, aug_ratio=0.2):
        super(GraphContrastiveLearning, self).__init__()
        
        # 对比学习参数
        self.temperature = temperature  # 温度参数
        self.aug_ratio = aug_ratio  # 数据增强比例
        
        # 投影头 - 将表示映射到对比学习空间
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, data, embedding_func):
        """对比学习前向传播"""
        try:
            # 调试信息
            print(f"Data type: {type(data)}")
            if hasattr(data, 'x') and data.x is not None:
                print(f"Node feature shape: {data.x.shape}")
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                print(f"Edge index shape: {data.edge_index.shape}")
            
            # 区分知识图谱和分子图
            is_kg = hasattr(data, 'kg_indices') or hasattr(data, 'kg_data') or hasattr(data, 'num_rel_type')
            
            # 获取原始数据的嵌入
            original_emb = embedding_func(data)
            batch_size = original_emb.size(0)
            
            # 创建简单的特征级增强视图（避免图结构增强的问题）
            emb_dim = original_emb.size(1)
            
            # 创建两个增强视图 - 直接在嵌入空间进行增强
            # 这是一种更稳健的方法，绕过图结构增强可能导致的问题
            mask1 = torch.rand_like(original_emb) > self.aug_ratio
            mask2 = torch.rand_like(original_emb) > self.aug_ratio
            
            # 确保至少保留一半的特征
            if mask1.sum() < emb_dim * batch_size / 2:
                mask1 = torch.ones_like(original_emb, dtype=torch.bool)
                mask1[:, torch.randperm(emb_dim)[:emb_dim//2]] = 0
            
            if mask2.sum() < emb_dim * batch_size / 2:
                mask2 = torch.ones_like(original_emb, dtype=torch.bool)
                mask2[:, torch.randperm(emb_dim)[:emb_dim//2]] = 0
            
            # 添加随机噪声
            z1 = original_emb.clone() * mask1 + torch.randn_like(original_emb) * 0.1 * (~mask1)
            z2 = original_emb.clone() * mask2 + torch.randn_like(original_emb) * 0.1 * (~mask2)
            
            # 计算对比损失
            loss = self.contrastive_loss(z1, z2)
            print(f"Contrastive loss computed: {loss.item()}")
            return loss
        except Exception as e:
            print(f"Error in contrastive learning forward: {str(e)}")
            print(f"Error details:", flush=True)
            import traceback
            traceback.print_exc()
            # 如果出错返回零损失
            return torch.tensor(0.0, device=data.edge_index.device, requires_grad=True)
    
    def contrastive_loss(self, z1, z2):
        """计算InfoNCE对比损失"""
        try:
            batch_size = z1.size(0)
            device = z1.device
            
            # 对特征进行投影和归一化
            p1 = F.normalize(self.projector(z1), dim=1)
            p2 = F.normalize(self.projector(z2), dim=1)
            
            # 计算余弦相似度矩阵
            sim_matrix = torch.matmul(p1, p2.t()) / self.temperature
            
            # 正样本是对角线元素 (相同索引的两个视图)
            labels = torch.arange(batch_size, device=device)
            
            # 双向对比损失
            loss_12 = F.cross_entropy(sim_matrix, labels)
            loss_21 = F.cross_entropy(sim_matrix.t(), labels)
            
            return (loss_12 + loss_21) / 2.0
        except Exception as e:
            print(f"Error in contrastive loss calculation: {str(e)}")
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

    # 保留其他辅助方法以便向后兼容
    def augment(self, data):
        """创建两个增强视图 (用于分子图，但现在我们使用更安全的方式)"""
        # 注意：此方法仅保留用于兼容，实际上我们通过嵌入后增强来避免尺寸问题
        return data, data
    
    def _augment_single(self, data):
        """单个图的增强 (用于兼容)"""
        return data
    
    def augment_kg(self, data):
        """知识图谱增强 (用于兼容)"""
        return data, data
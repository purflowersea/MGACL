import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, fusion_temperature=1.0, 
                 knowledge_weight=0.25, molecular_weight=0.25, 
                 smiles_weight=0.25, target_weight=0.25, fixed_weights=True):
        super().__init__()
        
        # 模态特定的投影层
        self.knowledge_proj = nn.Linear(dim, dim)
        self.molecular_proj = nn.Linear(dim, dim)
        self.smiles_proj = nn.Linear(dim, dim)
        self.target_proj = nn.Linear(dim, dim)
        
        # 多头自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # 权重生成网络
        self.weight_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )
        
        # 添加 fusion_mlp 层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        # 基础权重设置 - 从参数初始化
        self.base_weights = {
            'knowledge': knowledge_weight,
            'molecular': molecular_weight,
            'smiles': smiles_weight,
            'target': target_weight
        }

        # 是否使用固定权重
        self.fixed_weights = fixed_weights

        # 将权重转换为可注册的参数（不需要梯度）
        self.register_buffer('knowledge_weight_param', 
                           torch.tensor([knowledge_weight], dtype=torch.float))
        self.register_buffer('molecular_weight_param', 
                           torch.tensor([molecular_weight], dtype=torch.float))
        self.register_buffer('smiles_weight_param', 
                           torch.tensor([smiles_weight], dtype=torch.float))
        self.register_buffer('target_weight_param', 
                           torch.tensor([target_weight], dtype=torch.float))
        
        # 初始化last_weights
        self.last_weights = self.base_weights.copy()
        
        # 模态顺序
        self.modality_order = ['knowledge', 'molecular', 'smiles', 'target']
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * fusion_temperature)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, knowledge_emb, mol_emb, smiles_emb, target_emb):
        # 设备和形状检查
        device = knowledge_emb.device
        if not all(x.device == device for x in [mol_emb, smiles_emb, target_emb]):
            raise ValueError("All input tensors must be on the same device")
            
        expected_shape = knowledge_emb.shape
        tensors = {
            'knowledge': knowledge_emb,
            'molecular': mol_emb,
            'smiles': smiles_emb,
            'target': target_emb
        }
        for name, tensor in tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} tensor shape {tensor.shape} does not match expected shape {expected_shape}")
        
        # 特征投影
        knowledge_proj = self.knowledge_proj(knowledge_emb)
        mol_proj = self.molecular_proj(mol_emb)
        smiles_proj = self.smiles_proj(smiles_emb)
        target_proj = self.target_proj(target_emb)
        
        # 堆叠特征
        stacked_features = torch.stack(
            [knowledge_proj, mol_proj, smiles_proj, target_proj], 
            dim=1
        )
        
        # 自注意力处理
        attn_output, _ = self.self_attention(
            stacked_features,
            stacked_features,
            stacked_features
        )
        
        # 残差连接和层归一化
        attn_output = self.norm1(attn_output + stacked_features)
        
        # FFN处理
        ffn_output = self.ffn(attn_output)
        ffn_output = self.norm2(ffn_output + attn_output)
        
        # 根据fixed_weights确定最终权重
        if self.fixed_weights:
            # 直接使用固定权重
            final_weights = torch.tensor(
                [[self.knowledge_weight_param[0], self.molecular_weight_param[0],
                self.smiles_weight_param[0], self.target_weight_param[0]]],
                device=device
            )
        else:
            # 动态权重计算
            modal_weights = []
            for i in range(4):
                weight = self.weight_net(ffn_output[:, i])
                modal_weights.append(weight)
            
            # 权重归一化
            weights = torch.cat(modal_weights, dim=1)
            weights = torch.clamp(weights / self.temperature, min=-10.0, max=10.0)
            weights = F.softmax(weights, dim=1)
            
            # 验证权重
            weights = self._validate_weights(weights)
            
            # 混合基础权重
            base_weights = torch.tensor(
                [self.base_weights[k] for k in self.modality_order],
                device=device
            ).view(1, -1)
            
            alpha = torch.sigmoid(self.temperature)
            final_weights = (1 - alpha) * weights + alpha * base_weights
        
        # 加权融合
        weighted_features = []
        for i in range(4):
            weighted_feat = ffn_output[:, i] * final_weights[:, i].unsqueeze(-1)
            weighted_features.append(weighted_feat)
        
        # 特征拼接和最终融合
        concat_features = torch.cat(weighted_features, dim=1)
        fused = self.fusion_mlp(concat_features)
        
        # 更新权重记录
        with torch.no_grad():
            self.last_weights = {
                k: final_weights[:, i].mean().item()
                for i, k in enumerate(self.modality_order)
            }
            
            # 只有在非固定权重模式下才更新参数中的权重值
            if not self.fixed_weights:
                self.knowledge_weight_param[0] = self.last_weights['knowledge']
                self.molecular_weight_param[0] = self.last_weights['molecular']
                self.smiles_weight_param[0] = self.last_weights['smiles']
                self.target_weight_param[0] = self.last_weights['target']
        
        return fused
    
    def _validate_weights(self, weights):
        """验证权重的有效性"""
        # 检查NaN
        if torch.isnan(weights).any():
            print("Warning: NaN weights detected, using base weights")
            return torch.tensor(
                [self.base_weights[k] for k in self.modality_order],
                device=weights.device
            ).view(1, -1)
        
        # 处理负值
        if (weights < 0).any():
            weights = F.relu(weights)
        
        # 确保权重和为1
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / (weights_sum + 1e-8)
        
        return weights

    def get_modal_weights(self):
        """获取当前模态权重"""
        return self.last_weights

    def reset_weights(self):
        """重置权重到初始值"""
        self.last_weights = self.base_weights.copy()
        
        # 同时重置参数中的权重值
        with torch.no_grad():
            self.knowledge_weight_param[0] = self.base_weights['knowledge']
            self.molecular_weight_param[0] = self.base_weights['molecular']
            self.smiles_weight_param[0] = self.base_weights['smiles']
            self.target_weight_param[0] = self.base_weights['target']
            
    def update_base_weights(self, knowledge_weight=None, molecular_weight=None, 
                           smiles_weight=None, target_weight=None):
        """更新基础权重值"""
        if knowledge_weight is not None:
            self.base_weights['knowledge'] = knowledge_weight
            self.knowledge_weight_param[0] = knowledge_weight
            
        if molecular_weight is not None:
            self.base_weights['molecular'] = molecular_weight
            self.molecular_weight_param[0] = molecular_weight
            
        if smiles_weight is not None:
            self.base_weights['smiles'] = smiles_weight
            self.smiles_weight_param[0] = smiles_weight
            
        if target_weight is not None:
            self.base_weights['target'] = target_weight
            self.target_weight_param[0] = target_weight
            
        # 同步更新last_weights
        self.last_weights = self.base_weights.copy()
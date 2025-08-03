# 存储各数据集的最优权重配置

# 每个数据集的最优权重
DATASET_OPTIMAL_WEIGHTS = {
    "drugbank": {
        # 模态权重参数
        "knowledge_weight": 0.37,
        "molecular_weight": 0.30,
        "smiles_weight": 0.15,
        "target_weight": 0.18,
        # 对比学习参数
        "contrastive_weight": 0.35,
        "mol_ratio": 3/7,
        "kg_ratio": 4/7,
        "mol_temperature": 0.1,
        "kg_temperature": 0.2,
        "mol_aug_ratio": 0.1,
        "kg_aug_ratio": 0.3
    },
    "kegg": {
        # 模态权重参数
        "knowledge_weight": 0.38,
        "molecular_weight": 0.30,
        "smiles_weight": 0.12,
        "target_weight": 0.20,
        # 对比学习参数
        "contrastive_weight": 0.07,
        "mol_ratio": 0.65,
        "kg_ratio": 0.35,                
        "mol_temperature": 0.05,
        "kg_temperature": 0.13,
        "mol_aug_ratio": 0.16,
        "kg_aug_ratio": 0.22
    },
    "ogbl-biokg": {
        # 模态权重参数
        "knowledge_weight": 0.26,
        "molecular_weight": 0.25,
        "smiles_weight": 0.24,
        "target_weight": 0.25,
        # 对比学习参数
        "contrastive_weight": 0.1,
        "mol_ratio": 0.5,
        "kg_ratio": 0.5,
        "mol_temperature": 0.1,
        "kg_temperature": 0.2,
        "mol_aug_ratio": 0.1,
        "kg_aug_ratio": 0.2
    }
    # 可以根据需要添加更多数据集
}

def get_optimal_weights(dataset_name):
    """获取指定数据集的最优权重
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        dict: 包含权重的字典，如果找不到指定数据集则返回默认权重
    """
    # 转换为小写并处理可能的前缀
    dataset_key = dataset_name.lower()
    for key in DATASET_OPTIMAL_WEIGHTS.keys():
        if dataset_key.endswith(key):
            return DATASET_OPTIMAL_WEIGHTS[key]
    
    # 如果找不到匹配的数据集，返回默认权重
    print(f"Warning: No optimal weights found for dataset '{dataset_name}'. Using default weights.")
    return {
        # 模态权重参数  
        "knowledge_weight": 0.25,
        "molecular_weight": 0.25,
        "smiles_weight": 0.25,
        "target_weight": 0.25,
        # 对比学习参数
        "contrastive_weight": 0.1,
        "mol_ratio": 0.5,
        "kg_ratio": 0.5,
        "mol_temperature": 0.1,
        "kg_temperature": 0.2,
        "mol_aug_ratio": 0.1,
        "kg_aug_ratio": 0.2
    }
import time
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from torch_geometric.data import Batch
from tqdm import *
import copy
import os

# training function at each epoch

def train(loop, model, optimizer):
    correct, total_loss = 0, 0
    main_loss = 0  # 主任务损失
    contrast_loss = 0  # 对比学习损失
    model.train()

    prob_all = []
    label_all = []

    for data in loop:
        # # Debug: 打印 data 的内容和结构
        # print(f"data structure in loop: {data}")
        # print(f"data types: {[type(d) for d in data]}")

        try:
            # 检查是否使用 GPU
            device = model.device  # **确保所有数据在同一个设备**
            
            if torch.cuda.is_available():
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            else:
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)

            # print(f"[DEBUG] data_mol2 type after .cuda() or .cpu(): {type(data_mol2)}")

            
            # # Debug: 确保数据类型在传递到 forward 之前是正确的
            # print(f"[DEBUG] data_mol2 type before forward: {type(data_mol2)}")
            # print(f"[DEBUG] data_mol2 structure before forward: {data_mol2}")

            optimizer.zero_grad(set_to_none=True)

            # 前向传播
            predicts, loss = model(
                drug1_mol=data_mol1,
                drug1_subgraph=data_drug1,
                drug2_mol=data_mol2,
                drug2_subgraph=data_drug2,
                drug1_smiles=smiles1,
                drug2_smiles=smiles2,
                drug1_target=target1,  # **新增**
                drug2_target=target2   # **新增**
            )


            # # Debug: 检查预测结果和损失
            # print(f"predicts shape: {predicts.shape}, loss value: {loss.item()}")

            # 反向传播
            loss.backward()

            # 更新梯度
            optimizer.step()

            prob_all.append(predicts)
            label_all.append(data_mol1.y)

            # 计算总损失和分量损失
            batch_size = num_graphs(data_mol1)
            total_loss += loss.item() * batch_size
            
            # 如果使用对比学习，分别记录主任务损失和对比损失
            has_contrastive = True
            
            if has_contrastive and model.training:
                # 近似计算对比损失部分 (假设对比损失占总损失的比例由contrastive_weight决定)
                cl_weight = model.contrastive_weight
                if cl_weight > 0:
                    base_loss = loss.item() / (1 + cl_weight)
                    cl_component = loss.item() - base_loss
                    contrast_loss += cl_component * batch_size
                    main_loss += base_loss * batch_size
                else:
                    main_loss += loss.item() * batch_size
            else:
                main_loss += loss.item() * batch_size

        except Exception as e:
            print(f"Error during training loop: {e}")
            raise e

    # 计算指标
    train_loss = total_loss / len(loop)
    avg_main_loss = main_loss / len(loop)
    avg_contrast_loss = contrast_loss / len(loop) if contrast_loss > 0 else 0
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    train_acc, train_f1, train_auc, train_aupr = get_score(label_all, prob_all)

    # 获取当前模态权重（修改这部分）
    modal_weights = None
    # 尝试多种可能的属性名称
    if hasattr(model, 'cross_modal_fusion') and hasattr(model.cross_modal_fusion, 'get_modal_weights'):
        modal_weights = model.cross_modal_fusion.get_modal_weights()
    elif hasattr(model, 'fusion_module') and hasattr(model.fusion_module, 'get_modal_weights'):
        modal_weights = model.fusion_module.get_modal_weights()
    elif hasattr(model, 'fusion') and hasattr(model.fusion, 'get_modal_weights'):
        modal_weights = model.fusion.get_modal_weights()
    
    # 打印训练结果和权重
    print(f"Training Results - Accuracy: {train_acc}, F1: {train_f1}, AUC: {train_auc}, AUPR: {train_aupr}, Loss: {train_loss}")

    has_contrastive = True
    
    if has_contrastive:
        print(f"Losses - Total: {train_loss:.4f}, Main: {avg_main_loss:.4f}, Contrastive: {avg_contrast_loss:.4f}")
    else:
        print(f"Loss: {train_loss:.4f}")

    if modal_weights:
        weights_str = ', '.join([f"{k}: {v:.4f}" for k, v in modal_weights.items()])
        print(f"Modal Weights: {weights_str}")

    return train_acc, train_f1, train_auc, train_aupr, train_loss, modal_weights


def eval(loader, model):
    correct, total_loss = 0, 0
    model.eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            # 检查是否使用 GPU
            device = model.device  # **确保所有数据在同一个设备**
            
            if torch.cuda.is_available():
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            else:
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)


            # 前向传播
            predicts, loss = model(
                drug1_mol=data_mol1,
                drug1_subgraph=data_drug1,
                drug2_mol=data_mol2,
                drug2_subgraph=data_drug2,
                drug1_smiles=smiles1,
                drug2_smiles=smiles2,
                drug1_target=target1,
                drug2_target=target2
            )

            ##获取指标
            prob_all.append(predicts)
            label_all.append(data_mol1.y)
            total_loss += loss.item() * num_graphs(data_mol1)

    eval_loss = total_loss / len(loader.dataset)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    eval_acc, eval_f1, eval_auc, eval_aupr = get_score(label_all, prob_all)

    return eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss

def test(loader, model):
    correct, total_loss = 0, 0
    model.eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            # 检查是否使用 GPU
            device = model.device  # **确保所有数据在同一个设备**
            
            if torch.cuda.is_available():
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            else:
                data_mol1, data_drug1, target1, smiles1 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                data_mol2, data_drug2, target2, smiles2 = data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)

            # 前向传播
            predicts, loss = model(
                drug1_mol=data_mol1,
                drug1_subgraph=data_drug1,
                drug2_mol=data_mol2,
                drug2_subgraph=data_drug2,
                drug1_smiles=smiles1,
                drug2_smiles=smiles2,
                drug1_target=target1,
                drug2_target=target2
            )

            ##获取指标
            prob_all.append(predicts)
            label_all.append(data_mol1.y)
            total_loss += loss.item() * num_graphs(data_mol1)


    test_loss = total_loss / len(loader.dataset)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    test_acc, test_f1, test_auc, test_aupr = get_score(label_all, prob_all)

    return {"acc":test_acc,
            "f1":test_f1,
            "auc":test_auc,
            "aupr":test_aupr,
            "loss":test_loss}

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.c_size

def get_score(label_all, prob_all):

    predicts_label = [1 if prob >= 0.5 else 0 for prob in prob_all]

    acc = accuracy_score(label_all, predicts_label)
    f1 = f1_score(label_all, predicts_label)
    auroc = roc_auc_score(label_all, prob_all)
    p, r, t = precision_recall_curve(label_all, prob_all)
    auprc = auc(r, p)

    return acc, f1, auroc, auprc

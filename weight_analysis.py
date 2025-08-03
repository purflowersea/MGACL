import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_weights(weights_history, save_path):
    """绘制权重演变图
    
    Args:
        weights_history: 包含权重历史记录的字典列表
        save_path: 图表保存路径
    """
    if not weights_history:
        return
    
    plt.figure(figsize=(12, 8))
    
    # 绘制权重变化
    epochs = [entry['epoch'] for entry in weights_history]
    modalities = list(weights_history[0]['weights'].keys())
    
    for modality in modalities:
        values = [entry['weights'][modality] for entry in weights_history]
        plt.plot(epochs, values, 'o-', label=modality)
    
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Modal Weight Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # 如果有性能指标，绘制权重与性能的关系图
    if 'eval_auc' in weights_history[0] and 'eval_f1' in weights_history[0]:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        auc_values = [entry['eval_auc'] for entry in weights_history]
        f1_values = [entry['eval_f1'] for entry in weights_history]
        
        plt.plot(epochs, auc_values, 'o-', label='AUC')
        plt.plot(epochs, f1_values, 's-', label='F1')
        plt.xlabel('Epoch')
        plt.ylabel('Performance')
        plt.title('Validation Performance')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        base_path = os.path.splitext(save_path)[0]
        plt.savefig(f"{base_path}_with_performance.png")
        plt.close()

def analyze_optimal_weights(weights_history, save_dir, eval_metric='eval_auc', top_n=5):
    """分析并找出最优权重
    
    Args:
        weights_history: 包含权重历史记录的字典列表
        save_dir: 报告保存目录
        eval_metric: 用于排序的评估指标，默认为'eval_auc'
        top_n: 选择前N个最佳结果，默认为5
    
    Returns:
        dict: 包含最优权重信息的字典
    """
    if not weights_history or not weights_history[0].get('weights'):
        print("No valid weight history data found.")
        return None
    
    # 确保评估指标存在
    if eval_metric not in weights_history[0]:
        print(f"Warning: {eval_metric} not found in weight history. Using first entry.")
        sorted_entries = weights_history
    else:
        # 按照评估指标排序
        sorted_entries = sorted(weights_history, key=lambda x: x[eval_metric], reverse=True)
    
    # 获取前N个最佳的entries
    best_entries = sorted_entries[:min(top_n, len(sorted_entries))]
    
    # 提取这些entries的权重
    best_weights = [entry['weights'] for entry in best_entries]
    best_epochs = [entry['epoch'] for entry in best_entries]
    best_metrics = [entry.get(eval_metric, 'N/A') for entry in best_entries]
    
    # 计算平均最优权重
    modalities = best_weights[0].keys()
    optimal_weights = {}
    weight_stds = {}
    
    for modality in modalities:
        values = [w[modality] for w in best_weights]
        optimal_weights[modality] = sum(values) / len(values)
        weight_stds[modality] = np.std(values)
    
    # 创建权重分析报告
    report = {
        'optimal_weights': optimal_weights,
        'weight_stds': weight_stds,
        'best_epochs': best_epochs,
        'best_metrics': best_metrics
    }
    
    # 保存报告
    with open(os.path.join(save_dir, 'optimal_weights_analysis.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印分析结果
    print("\n" + "="*50)
    print("最优权重分析 (基于前{}个最佳{}):".format(top_n, eval_metric))
    print("-"*50)
    print("最佳epochs: {}".format(best_epochs))
    print("对应{}: {}".format(eval_metric, [f"{m:.4f}" if isinstance(m, (int, float)) else m for m in best_metrics]))
    print("\n平均最优权重:")
    for modality, weight in optimal_weights.items():
        print(f"{modality}: {weight:.4f} ± {weight_stds[modality]:.4f}")
    print("="*50 + "\n")
    
    # 创建最优权重可视化
    plt.figure(figsize=(10, 6))
    x = np.arange(len(modalities))
    plt.bar(x, [optimal_weights[m] for m in modalities], 
           yerr=[weight_stds[m] for m in modalities],
           alpha=0.7, capsize=10)
    plt.xlabel('Modality')
    plt.ylabel('Average Weight')
    plt.title('Optimal Modality Weights')
    plt.xticks(x, modalities, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimal_weights_chart.png'))
    plt.close()
    
    return report

def analyze_all_folds(base_dir):
    """分析所有fold的权重数据，生成综合报告
    
    Args:
        base_dir: 包含所有fold权重数据的基础目录
    """
    # 查找所有fold目录中的权重分析文件
    weight_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'optimal_weights_analysis.json':
                weight_files.append(os.path.join(root, file))
    
    if not weight_files:
        print(f"No weight analysis files found in {base_dir}")
        return
    
    print(f"Found {len(weight_files)} weight analysis files")
    
    # 读取所有fold的权重数据
    all_optimal_weights = []
    all_weight_stds = []
    
    for file in weight_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_optimal_weights.append(data['optimal_weights'])
            all_weight_stds.append(data['weight_stds'])
    
    # 计算所有fold的平均最优权重
    modalities = all_optimal_weights[0].keys()
    final_weights = {}
    final_stds = {}
    
    for modality in modalities:
        values = [w[modality] for w in all_optimal_weights]
        final_weights[modality] = sum(values) / len(values)
        final_stds[modality] = np.std(values)
    
    # 创建综合报告
    report = {
        'final_optimal_weights': final_weights,
        'final_weight_stds': final_stds,
        'all_fold_weights': all_optimal_weights
    }
    
    # 保存报告
    output_dir = os.path.dirname(base_dir)
    with open(os.path.join(output_dir, 'final_weight_analysis.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # 创建条形图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(modalities))
    plt.bar(x, [final_weights[m] for m in modalities], 
           yerr=[final_stds[m] for m in modalities],
           alpha=0.7, capsize=10)
    plt.xlabel('Modality')
    plt.ylabel('Average Weight')
    plt.title('Final Optimal Modality Weights Across All Folds')
    plt.xticks(x, modalities, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_optimal_weights.png'))
    
    # 打印分析结果
    print("\n" + "="*50)
    print("最终优化权重 (跨所有fold):")
    print("-"*50)
    for modality, weight in final_weights.items():
        print(f"{modality}: {weight:.4f} ± {final_stds[modality]:.4f}")
    print("="*50)
    print(f"完整报告已保存至: {os.path.join(output_dir, 'final_weight_analysis.json')}")
    print(f"权重图表已保存至: {os.path.join(output_dir, 'final_optimal_weights.png')}")

def record_weights_in_training(train_log, modal_weights, i_episode, eval_metrics):
    """在训练过程中记录模态权重
    
    Args:
        train_log: 训练日志字典
        modal_weights: 当前模态权重
        i_episode: 当前epoch
        eval_metrics: 包含评估指标的字典或元组
    
    Returns:
        dict: 更新后的训练日志
    """
    if modal_weights:
        if 'modal_weights' not in train_log:
            train_log['modal_weights'] = []
        
        # 创建权重记录条目
        weight_entry = {
            'epoch': i_episode + 1,
            'weights': modal_weights
        }
        
        # 添加评估指标
        if isinstance(eval_metrics, dict):
            for k, v in eval_metrics.items():
                weight_entry[k] = v
        elif isinstance(eval_metrics, tuple) and len(eval_metrics) >= 2:
            weight_entry['eval_acc'] = eval_metrics[0]
            weight_entry['eval_f1'] = eval_metrics[1]
            if len(eval_metrics) > 2:
                weight_entry['eval_auc'] = eval_metrics[2]
            if len(eval_metrics) > 3:
                weight_entry['eval_aupr'] = eval_metrics[3]
        
        train_log['modal_weights'].append(weight_entry)
    
    return train_log

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze weights from training')
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Base directory containing weight records')
    parser.add_argument('--mode', type=str, choices=['single', 'all'], default='all',
                       help='Analyze a single fold or all folds (default: all)')
    parser.add_argument('--weights_file', type=str, default=None,
                       help='Path to weights history file (only for single mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'single' and args.weights_file:
        with open(args.weights_file, 'r') as f:
            weights_history = json.load(f)['modal_weights']
        save_dir = os.path.dirname(args.weights_file)
        visualize_weights(weights_history, os.path.join(save_dir, 'weight_evolution.png'))
        analyze_optimal_weights(weights_history, save_dir)
    else:
        analyze_all_folds(args.base_dir)
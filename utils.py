"""
通用工具函数
包含设备设置、随机种子、打印等辅助功能
"""

import torch
import numpy as np
import random


def setup_device(use_cuda=True):
    """
    设置训练设备（GPU/CPU）
    
    Args:
        use_cuda: 是否使用 CUDA（如果可用）
        
    Returns:
        device: torch.device 对象
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用设备: {device}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"✓ 使用设备: {device}")
    
    return device


def set_seed(seed=42):
    """
    设置所有随机种子以保证可复现性
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 确保 CUDA 卷积的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ 随机种子设置为: {seed}")


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        total, trainable: 总参数数，可训练参数数
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: PyTorch 模型
    """
    total, trainable = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"模型信息")
    print(f"{'='*70}")
    print(f"  架构: {model.__class__.__name__}")
    print(f"  总参数: {total:,}")
    print(f"  可训练参数: {trainable:,}")
    
    # 显示注意力机制状态（如果是TCN模型）
    if hasattr(model, 'use_attention'):
        attention_status = "✅ 启用" if model.use_attention else "❌ 禁用"
        print(f"  注意力机制: {attention_status}")
    
    print(f"{'='*70}\n")


def save_checkpoint(model, optimizer, epoch, best_metric, filepath):
    """
    保存模型检查点
    
    Args:
        model: PyTorch 模型
        optimizer: 优化器
        epoch: 当前 epoch
        best_metric: 最佳指标值
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ 模型保存到: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    加载模型检查点
    
    Args:
        model: PyTorch 模型
        optimizer: 优化器
        filepath: 检查点路径
        device: 设备
        
    Returns:
        epoch, best_metric: 训练 epoch 和最佳指标
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    
    print(f"✓ 从 {filepath} 加载模型 (Epoch {epoch}, 最佳指标: {best_metric:.4f})")
    return epoch, best_metric

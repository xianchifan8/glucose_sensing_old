"""
Glucose Sensing Training Framework with W&B Integration
主训练文件 - 整合模型、数据加载和训练流程
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 从子模块导入
from Util.data_utils import load_and_preprocess_data


def setup_device():
    """设置训练设备（GPU/CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"CUDA版本: {torch.version.cuda}")
    return device


def get_default_config():
    """获取默认配置"""
    return {
        # 训练参数
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "weight_decay": 1e-4,
        
        # 模型参数（自动从数据推断 input_size）
        "architecture": "MLP",
        "hidden_size": 256,
        "dropout": 0.5,
        "task_type": "regression",  # 回归任务
        
        # 数据参数
        "user_name": "Tao",
        "dataset_root": "./Dataset",
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        
        # 优化器参数
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "patience": 5,
        
        # 其他
        "seed": 42,
        "num_workers": 0,  # Linux 上设为0避免多进程问题
    }


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_model(config, device):
    """创建回归模型"""
    # 回归模型：输出单个值（血糖浓度）
    class RegressionModel(nn.Module):
        def __init__(self, input_size, hidden_size, dropout):
            super(RegressionModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.bn2 = nn.BatchNorm1d(hidden_size // 2)
            
            self.fc3 = nn.Linear(hidden_size // 2, 1)  # 输出1个值
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.fc3(x)
            return x.squeeze()  # 输出 shape: (batch_size,)
    
    model = RegressionModel(
        config.input_size,
        config.hidden_size,
        config.dropout
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model


def load_data(config):
    """加载真实数据"""
    train_loader, val_loader, test_loader, metadata = load_and_preprocess_data(
        config,
        dataset_root=config.dataset_root
    )
    
    # 从第一个batch获取实际的输入维度
    sample_batch = next(iter(train_loader))
    actual_input_size = sample_batch[0].shape[1]
    config.input_size = actual_input_size
    
    print(f"✓ 实际输入维度: {actual_input_size}")
    
    return train_loader, val_loader, test_loader, metadata


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch（回归任务）"""
    model.train()
    total_loss = 0
    total_mae = 0  # Mean Absolute Error
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(output - target).sum().item()
        total_samples += batch_size
        
        # 每10个batch打印一次
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_loss, avg_mae


def validate(model, val_loader, criterion, device):
    """验证模型（回归任务）"""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(output - target).sum().item()
            total_rmse += ((output - target) ** 2).sum().item()
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = (total_rmse / total_samples) ** 0.5
    
    return avg_loss, avg_mae, avg_rmse


def train(config, device):
    """主训练函数（回归任务）"""
    # 设置随机种子
    set_seed(config.seed)
    
    # 加载数据（先加载以获取实际输入维度）
    print("\n加载数据...")
    train_loader, val_loader, test_loader, metadata = load_data(config)
    
    # 创建模型（使用实际的输入维度）
    print("\n创建模型...")
    model = create_model(config, device)
    
    # 定义损失函数和优化器（回归任务使用 MSE Loss）
    criterion = nn.MSELoss()
    
    if config.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")
    
    # 学习率调度器
    if config.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.patience,
            factor=0.5
        )
    
    # 训练循环
    print("\n开始训练...")
    best_val_mae = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f"\n训练 - Loss: {train_loss:.4f}, MAE: {train_mae:.4f} mmol/L")
        print(f"验证 - Loss: {val_loss:.4f}, MAE: {val_mae:.4f} mmol/L, RMSE: {val_rmse:.4f} mmol/L")
        print(f"学习率: {current_lr:.6f}")
        
        # 记录到 wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "learning_rate": current_lr,
        })
        
        # 保存最佳模型（基于 MAE）
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path = os.path.join(wandb.run.dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ 保存最佳模型 (验证MAE: {val_mae:.4f} mmol/L)")
            
            # 保存到 wandb
            wandb.save(checkpoint_path)
    
    print(f"\n训练完成！最佳验证MAE: {best_val_mae:.4f} mmol/L")
    
    # 记录最终结果
    wandb.log({"best_val_mae": best_val_mae})
    
    return model, best_val_mae


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Glucose Sensing Training')
    parser.add_argument('--run_name', type=str, default=None, help='W&B运行名称')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--notes', type=str, default=None, help='实验备注')
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 获取配置
    config_dict = get_default_config()
    config_dict['device'] = str(device)
    
    # 命令行参数覆盖
    if args.epochs:
        config_dict['epochs'] = args.epochs
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.lr:
        config_dict['learning_rate'] = args.lr
    
    # 初始化 wandb
    wandb.init(
        entity="glucose_msra",
        project="sensing",
        name=args.run_name,
        notes=args.notes,
        config=config_dict,
        tags=["training", config_dict['architecture']],
    )
    
    # 获取wandb配置对象
    config = wandb.config
    
    print("\n" + "="*50)
    print("配置信息")
    print("="*50)
    for key, value in dict(config).items():
        print(f"{key}: {value}")
    print("="*50 + "\n")
    
    # 训练
    model, best_mae = train(config, device)
    
    # 完成
    wandb.finish()
    print(f"\n实验完成！最佳验证MAE: {best_mae:.4f} mmol/L")
    print("结果已保存到 W&B")


if __name__ == "__main__":
    main()

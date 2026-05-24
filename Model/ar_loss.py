"""
Auto-Regressive Loss Functions
包含自回归模式的损失函数，包括平滑性正则化和历史一致性正则化

设计原则：
1. 主损失：预测值与真实值的误差（MSE/MAE/Huber）
2. 平滑性正则化：限制预测值与上一个历史血糖的跳变
3. 历史一致性正则化：预测值应符合历史趋势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoRegressiveLoss(nn.Module):
    """
    自回归模式的复合损失函数
    
    Loss = main_loss + smoothness_weight * smoothness_loss + history_weight * history_loss
    
    其中：
    - main_loss: 预测误差 (MSE/MAE/Huber)
    - smoothness_loss: |pred - last_glucose|^2，限制预测跳变
    - history_loss: |pred - trend_prediction|^2，限制偏离历史趋势
    """
    def __init__(self, base_criterion='MSE', smoothness_weight=0.1, 
                 history_weight=0.05, huber_delta=1.0):
        """
        Args:
            base_criterion: 基础损失函数类型 'MSE', 'MAE', 'Huber'
            smoothness_weight: 平滑性正则化权重
            history_weight: 历史一致性正则化权重
            huber_delta: Huber损失的delta参数
        """
        super(AutoRegressiveLoss, self).__init__()
        
        self.smoothness_weight = smoothness_weight
        self.history_weight = history_weight
        
        # 创建基础损失函数
        if base_criterion.upper() == 'MSE':
            self.base_criterion = nn.MSELoss()
        elif base_criterion.upper() == 'MAE':
            self.base_criterion = nn.L1Loss()
        elif base_criterion.upper() == 'HUBER':
            self.base_criterion = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"不支持的损失函数: {base_criterion}")
        
        self.base_criterion_name = base_criterion.upper()
    
    def compute_smoothness_loss(self, predictions, last_glucose):
        """
        计算平滑性损失：限制预测值与最近历史血糖的跳变
        
        Args:
            predictions: (batch,) 预测的血糖值
            last_glucose: (batch,) 历史序列中最后一个血糖值
        
        Returns:
            smoothness_loss: 标量
        """
        # L2损失：(pred - last)^2
        diff = predictions - last_glucose
        return torch.mean(diff ** 2)
    
    def compute_history_trend_loss(self, predictions, glucose_history):
        """
        计算历史趋势一致性损失
        基于历史血糖的线性趋势，预测当前值不应偏离太多
        
        Args:
            predictions: (batch,) 预测的血糖值
            glucose_history: (batch, history_len) 历史血糖序列
        
        Returns:
            trend_loss: 标量
        """
        batch_size, history_len = glucose_history.shape
        
        if history_len < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # 计算历史趋势（简单线性回归的斜率）
        # y = ax + b, 其中x是时间步 [0, 1, 2, ..., history_len-1]
        x = torch.arange(history_len, dtype=torch.float32, device=glucose_history.device)
        x = x.unsqueeze(0).expand(batch_size, -1)  # (batch, history_len)
        
        # 计算斜率 a = sum((x-x_mean)(y-y_mean)) / sum((x-x_mean)^2)
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = glucose_history.mean(dim=1, keepdim=True)
        
        numerator = ((x - x_mean) * (glucose_history - y_mean)).sum(dim=1)
        denominator = ((x - x_mean) ** 2).sum(dim=1) + 1e-8
        slope = numerator / denominator  # (batch,)
        
        # 预测下一个时间步的趋势值
        # trend_pred = y_mean + slope * (history_len - x_mean)
        # 简化：使用最后一个值 + 斜率
        last_glucose = glucose_history[:, -1]
        trend_prediction = last_glucose + slope  # (batch,)
        
        # 计算预测值与趋势预测的偏差
        diff = predictions - trend_prediction
        return torch.mean(diff ** 2)
    
    def forward(self, predictions, targets, glucose_history=None, 
                return_components=False):
        """
        计算总损失
        
        Args:
            predictions: (batch,) 模型预测值
            targets: (batch,) 真实血糖值
            glucose_history: (batch, history_len) 历史血糖序列（可选）
            return_components: 是否返回各分量
        
        Returns:
            total_loss: 总损失
            如果return_components=True，还返回:
                main_loss, smoothness_loss, history_loss
        """
        # 1. 主损失
        main_loss = self.base_criterion(predictions, targets)
        
        # 如果没有历史血糖数据，只返回主损失
        if glucose_history is None:
            if return_components:
                return main_loss, main_loss, torch.tensor(0.0), torch.tensor(0.0)
            return main_loss
        
        # 2. 平滑性损失
        last_glucose = glucose_history[:, -1]  # 最后一个历史值
        smoothness_loss = self.compute_smoothness_loss(predictions, last_glucose)
        
        # 3. 历史趋势一致性损失
        history_loss = self.compute_history_trend_loss(predictions, glucose_history)
        
        # 4. 总损失
        total_loss = (main_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.history_weight * history_loss)
        
        if return_components:
            return total_loss, main_loss, smoothness_loss, history_loss
        
        return total_loss
    
    def extra_repr(self):
        return (f"base={self.base_criterion_name}, "
                f"smoothness_weight={self.smoothness_weight}, "
                f"history_weight={self.history_weight}")


class AdaptiveARLoss(AutoRegressiveLoss):
    """
    自适应权重的自回归损失函数
    在训练过程中动态调整正则化权重
    
    策略：
    - 训练初期：较低的正则化权重，让模型学习基本映射
    - 训练中期：逐渐增加正则化权重，学习平滑预测
    - 训练后期：保持稳定的正则化权重
    """
    def __init__(self, base_criterion='MSE', smoothness_weight=0.1,
                 history_weight=0.05, huber_delta=1.0,
                 warmup_epochs=10, max_epochs=100):
        """
        Args:
            warmup_epochs: 预热期（正则化权重从0增长到目标值）
            max_epochs: 总训练轮数
        """
        super().__init__(base_criterion, smoothness_weight, history_weight, huber_delta)
        
        self.target_smoothness_weight = smoothness_weight
        self.target_history_weight = history_weight
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        """更新当前epoch，调整权重"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # 线性预热
            scale = epoch / self.warmup_epochs
            self.smoothness_weight = self.target_smoothness_weight * scale
            self.history_weight = self.target_history_weight * scale
        else:
            self.smoothness_weight = self.target_smoothness_weight
            self.history_weight = self.target_history_weight


def create_ar_criterion(config):
    """
    根据配置创建AR损失函数
    
    Args:
        config: 配置对象
    
    Returns:
        AutoRegressiveLoss 或 AdaptiveARLoss
    """
    return AutoRegressiveLoss(
        base_criterion=config.training.loss_function,
        smoothness_weight=config.data.ar_smoothness_weight,
        history_weight=config.data.ar_history_weight,
        huber_delta=config.training.huber_delta
    )


# 导出
__all__ = [
    'AutoRegressiveLoss',
    'AdaptiveARLoss',
    'create_ar_criterion',
]

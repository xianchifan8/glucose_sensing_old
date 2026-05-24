"""
模型定义文件
包含所有用于血糖预测的深度学习模型
支持分类和回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlucoseMLP(nn.Module):
    """
    多层感知机（MLP）模型
    适用于回归和分类任务
    """
    def __init__(self, input_size, hidden_size=256, num_layers=2, 
                 dropout=0.5, task_type='regression'):
        super(GlucoseMLP, self).__init__()
        
        self.task_type = task_type
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size // 2))
            layers.append(nn.BatchNorm1d(hidden_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            hidden_size = hidden_size // 2
        
        self.features = nn.Sequential(*layers)
        
        # 输出层
        if task_type == 'regression':
            self.output = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"任务类型 {task_type} 暂不支持，请使用 'regression'")
    
    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)  # (batch_size,)
        return x


class GlucoseCNN(nn.Module):
    """
    1D CNN模型用于glucose sensing
    将特征看作1D序列进行卷积
    """
    def __init__(self, input_size=1001, hidden_size=256, dropout=0.5, 
                 task_type='regression'):
        super(GlucoseCNN, self).__init__()
        
        self.task_type = task_type
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # 计算全连接层输入维度
        fc_input_size = 128 * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        
        if task_type == 'regression':
            self.fc2 = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"任务类型 {task_type} 暂不支持")
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)
        return x


class GlucoseTransformer(nn.Module):
    """
    Transformer模型用于glucose sensing
    
    改进：
    1. 降低dropout率，避免过度正则化
    2. 添加LayerNorm稳定训练
    3. 使用更合理的位置编码初始化
    4. 增加残差连接
    """
    def __init__(self, input_size=1001, hidden_size=256, nhead=8, 
                 num_layers=3, dropout=0.5, task_type='regression'):
        super(GlucoseTransformer, self).__init__()
        
        self.task_type = task_type
        self.hidden_size = hidden_size
        
        # Embedding层 + LayerNorm
        self.embedding = nn.Linear(1, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        
        # 位置编码：使用更小的初始化，降低影响
        self.pos_encoder = nn.Parameter(torch.randn(1, input_size, hidden_size) * 0.01)
        
        # Transformer编码器：降低dropout率
        # 对于频谱数据，dropout=0.5太高，改为0.1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,  # 降低dropout率
            batch_first=True,
            norm_first=True  # Pre-LN架构，更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层：降低dropout率
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)
        
        # 初始化
        self._init_weights()
        
        if task_type != 'regression':
            raise ValueError(f"任务类型 {task_type} 暂不支持")
    
    def _init_weights(self):
        """改进的权重初始化"""
        # Embedding层使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        
        # 输出层使用较小的初始化
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Reshape: (batch_size, input_size) -> (batch_size, input_size, 1)
        x = x.unsqueeze(2)
        
        # Embedding + Normalization
        x = self.embedding(x)
        x = self.embedding_norm(x)
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # 输出层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x.squeeze(-1)  # (batch_size,)


def create_model(config):
    """
    根据配置创建模型
    
    Args:
        config: ModelConfig 对象
    
    Returns:
        model: PyTorch模型
    """
    arch = config.architecture.upper()
    
    # 通用参数
    kwargs = {
        'input_size': config.input_size,
        'hidden_size': config.hidden_size,
        'dropout': config.dropout,
        'task_type': config.task_type,
    }
    
    if arch == 'MLP':
        kwargs['num_layers'] = config.num_layers
        return GlucoseMLP(**kwargs)
    
    elif arch == 'CNN':
        return GlucoseCNN(**kwargs)
    
    elif arch == 'TRANSFORMER':
        kwargs['num_layers'] = config.num_layers
        kwargs['nhead'] = 8  # 可以添加到config中
        return GlucoseTransformer(**kwargs)
    
    else:
        raise ValueError(f"不支持的模型架构: {arch}")


# 保留旧的工厂函数以保持兼容性
def get_model(model_name, **kwargs):
    """
    工厂函数：根据名称创建模型（兼容旧版本）
    
    Args:
        model_name: 模型名称 ('mlp', 'cnn', 'rnn', 'transformer')
        **kwargs: 模型参数
    
    Returns:
        model: PyTorch模型
    """
    model_dict = {
        'mlp': GlucoseMLP,
        'cnn': GlucoseCNN,
        'transformer': GlucoseTransformer,
    }
    
    model_name = model_name.lower()
    if model_name not in model_dict:
        raise ValueError(f"不支持的模型: {model_name}. 可选: {list(model_dict.keys())}")
    
    return model_dict[model_name](**kwargs)


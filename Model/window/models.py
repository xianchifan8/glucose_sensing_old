"""
Window模式的模型定义
使用历史窗口数据预测当前血糖浓度

Input: (batch_size, window_size, n_features)
Output: (batch_size,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowMLP(nn.Module):
    """
    Window模式的MLP模型
    将窗口数据展平后通过MLP预测
    """
    def __init__(self, input_size, window_size, hidden_size=256, num_layers=2, 
                 dropout=0.5, task_type='regression'):
        super(WindowMLP, self).__init__()
        
        self.task_type = task_type
        self.window_size = window_size
        self.input_size = input_size
        
        # 展平后的输入维度
        flattened_size = window_size * input_size
        
        layers = []
        
        # 输入层 - 使用LayerNorm替代BatchNorm (对MLP更合适)
        layers.append(nn.Linear(flattened_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        current_size = hidden_size
        for _ in range(num_layers - 1):
            next_size = current_size // 2
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.LayerNorm(next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = next_size
        
        self.features = nn.Sequential(*layers)
        
        # 输出层
        if task_type == 'regression':
            self.output = nn.Linear(current_size, 1)
        else:
            raise ValueError(f"任务类型 {task_type} 暂不支持")
    
    def extract_features(self, x):
        """提取特征（用于Late Fusion）"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.features(x)
    
    def get_feature_dim(self):
        """返回extract_features输出的特征维度"""
        # 计算features最后一层的输出维度
        current_size = self.features[0].out_features  # hidden_size
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                current_size = layer.out_features
        return current_size
    
    def forward(self, x):
        # x shape: (batch_size, window_size, n_features)
        features = self.extract_features(x)
        x = self.output(features)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)  # (batch_size,)
        return x


class WindowCNN(nn.Module):
    """
    Window模式的CNN模型
    在时间维度和特征维度上进行2D卷积
    """
    def __init__(self, input_size, window_size, hidden_size=256, dropout=0.5, 
                 task_type='regression'):
        super(WindowCNN, self).__init__()
        
        self.task_type = task_type
        
        # 2D卷积层 - 将窗口数据看作2D图像 - 使用LayerNorm替代BatchNorm2d
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.ln1 = nn.LayerNorm(32)  # 对通道维度归一化
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.ln2 = nn.LayerNorm(64)  # 对通道维度归一化
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.ln3 = nn.LayerNorm(128)  # 对通道维度归一化
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层 - 使用LayerNorm
        self.fc1 = nn.Linear(128, hidden_size)
        self.ln_fc = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self._feature_dim = hidden_size
        if task_type == 'regression':
            self.output = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"任务类型 {task_type} 暂不支持")
    
    def _encode(self, x):
        """内部编码方法"""
        x = x.unsqueeze(1)  # (batch_size, 1, window_size, n_features)
        
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.ln1(x))
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.ln2(x))
        x = x.permute(0, 3, 1, 2)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.ln3(x))
        x = x.permute(0, 3, 1, 2)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # (batch_size, 128)
        x = F.relu(self.ln_fc(self.fc1(x)))
        x = self.dropout(x)
        return x
    
    def extract_features(self, x):
        """提取特征（用于Late Fusion）"""
        return self._encode(x)
    
    def get_feature_dim(self):
        """返回extract_features输出的特征维度"""
        return self._feature_dim
    
    def forward(self, x):
        # x shape: (batch_size, window_size, n_features)
        features = self._encode(x)
        x = self.output(features)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)  # (batch_size,)
        return x


class WindowTransformer(nn.Module):
    """
    Window模式的Transformer模型
    使用自注意力机制处理时间序列窗口
    """
    def __init__(self, input_size, window_size, hidden_size=256, nhead=8,
                 num_layers=3, dropout=0.5, task_type='regression'):
        super(WindowTransformer, self).__init__()
        
        self.task_type = task_type
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        # Embedding层
        self.embedding = nn.Linear(input_size, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, hidden_size) * 0.01)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.dropout = nn.Dropout(0.1)
        if task_type == 'regression':
            self.output = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"任务类型 {task_type} 暂不支持")
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)
    
    def _encode(self, x):
        """内部编码方法"""
        x = self.embedding(x)
        x = self.embedding_norm(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.dropout(x)
    
    def extract_features(self, x):
        """提取特征（用于Late Fusion）"""
        return self._encode(x)
    
    def get_feature_dim(self):
        """返回extract_features输出的特征维度"""
        return self.hidden_size
    
    def forward(self, x):
        # x shape: (batch_size, window_size, n_features)
        features = self._encode(x)
        x = self.output(features)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)  # (batch_size,)
        return x


def create_window_model(config, window_size):
    """
    根据配置创建window模式的模型
    
    Args:
        config: ModelConfig 对象
        window_size: 窗口大小
    
    Returns:
        model: PyTorch模型
    """
    arch = config.architecture.upper()
    
    # 通用参数
    kwargs = {
        'input_size': config.input_size,
        'window_size': window_size,
        'hidden_size': config.hidden_size,
        'dropout': config.dropout,
        'task_type': config.task_type,
    }
    
    if arch == 'MLP':
        kwargs['num_layers'] = config.num_layers
        return WindowMLP(**kwargs)
    
    elif arch == 'CNN':
        return WindowCNN(**kwargs)
    
    elif arch == 'TRANSFORMER':
        kwargs['num_layers'] = config.num_layers
        kwargs['nhead'] = 8
        return WindowTransformer(**kwargs)
    
    elif arch == 'TCN':
        # 使用TCN模型
        from .tcn_models import create_tcn_model
        return create_tcn_model(config, window_size)
    
    else:
        raise ValueError(f"不支持的模型架构: {arch}")

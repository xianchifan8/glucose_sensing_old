"""
TCN (Temporal Convolutional Network) 模型
专为频谱时序数据设计

架构：Spectral CNN (特征提取) -> Temporal TCN (时序建模) -> FC (预测)

参考：
- TCN论文: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"
- 适用场景: 长序列时序预测，频谱数据分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class DropBlock1D(nn.Module):
    """
    DropBlock for 1D data (时序数据)
    相比Dropout，DropBlock丢弃连续的区域块，防止过拟合更强
    
    论文: "DropBlock: A regularization method for convolutional networks"
    
    Args:
        drop_prob: 丢弃概率
        block_size: 丢弃块的大小（时间步数）
    """
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock1D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    
    def forward(self, x):
        # x: (batch, channels, length)
        if not self.training or self.drop_prob == 0.0:
            return x
        
        # 计算gamma（需要丢弃的中心点概率）
        gamma = self.drop_prob / (self.block_size ** 1)
        
        # 为每个位置采样伯努利分布
        batch_size, channels, length = x.shape
        
        # 生成mask
        mask = torch.bernoulli(
            torch.ones(batch_size, channels, length, device=x.device) * gamma
        )
        
        # 将mask扩展为block
        if self.block_size > 1:
            mask = F.max_pool1d(
                mask,
                kernel_size=self.block_size,
                stride=1,
                padding=self.block_size // 2
            )
        
        # 反转mask（0变1，1变0）
        mask = 1 - mask
        
        # 归一化（保持期望不变）
        normalize_factor = mask.numel() / (mask.sum() + 1e-7)
        
        return x * mask * normalize_factor


class ChannelAttention(nn.Module):
    """
    通道注意力机制（频谱注意力）
    学习哪些频点对血糖预测更重要
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, length)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * out


class SpatialAttention(nn.Module):
    """
    空间注意力机制（时间注意力）
    学习哪些时间步对当前预测更重要
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, length)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out


class TemporalAggregation(nn.Module):
    """
    智能时间步聚合模块
    将窗口中所有时间步的信息智能地组合起来
    
    支持多种聚合策略：
    1. attention: 注意力加权求和（学习每个时间步的重要性）
    2. weighted: 可学习的加权平均（最近的时间步权重更大）
    3. concat: 拼接所有时间步 + FC降维
    4. lstm: 使用LSTM聚合所有时间步
    5. mean: 全局平均池化
    6. last: 仅使用最后一个时间步（原始方法）
    """
    def __init__(self, hidden_size, window_size, strategy='attention', dropout=0.1):
        super(TemporalAggregation, self).__init__()
        self.strategy = strategy
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        if strategy == 'attention':
            # 注意力机制：学习每个时间步的权重
            self.attention_fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1)
            )
            
        elif strategy == 'weighted':
            # 可学习的时间权重（初始化为线性递增，最近的时间步权重更大）
            weights = torch.linspace(0.5, 1.0, window_size)
            self.time_weights = nn.Parameter(weights.unsqueeze(0).unsqueeze(0))  # (1, 1, window_size)
            
        elif strategy == 'concat':
            # 拼接所有时间步 + FC降维
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * window_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            
        elif strategy == 'lstm':
            # 使用LSTM聚合
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, hidden_size, window_size) 或 (batch_size, window_size, hidden_size)
        Returns:
            out: (batch_size, hidden_size)
        """
        # 确保输入是 (batch_size, window_size, hidden_size)
        if x.size(1) == self.hidden_size:
            x = x.transpose(1, 2)  # (batch_size, window_size, hidden_size)
        
        batch_size = x.size(0)
        
        if self.strategy == 'attention':
            # 注意力加权求和
            # 计算每个时间步的注意力分数
            attn_scores = self.attention_fc(x)  # (batch_size, window_size, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, window_size, 1)
            
            # 加权求和
            out = torch.sum(x * attn_weights, dim=1)  # (batch_size, hidden_size)
            
        elif self.strategy == 'weighted':
            # 可学习的加权平均
            weights = F.softmax(self.time_weights, dim=-1)  # (1, 1, window_size)
            x_permuted = x.transpose(1, 2)  # (batch_size, hidden_size, window_size)
            out = torch.sum(x_permuted * weights, dim=-1)  # (batch_size, hidden_size)
            
        elif self.strategy == 'concat':
            # 拼接 + FC
            x_flat = x.reshape(batch_size, -1)  # (batch_size, hidden_size * window_size)
            out = self.fc(x_flat)  # (batch_size, hidden_size)
            
        elif self.strategy == 'lstm':
            # LSTM聚合
            lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, window_size, hidden_size)
            out = h_n[-1]  # 取最后一层的hidden state: (batch_size, hidden_size)
            out = self.dropout(out)
            
        elif self.strategy == 'last':
            # 仅取最后一个时间步（原始方法）
            out = x[:, -1, :]  # (batch_size, hidden_size)
            
        else:
            # 默认：全局平均池化
            out = x.mean(dim=1)  # (batch_size, hidden_size)
        
        return out


class Chomp1d(nn.Module):
    """
    裁剪层：用于因果卷积
    去除padding引入的未来信息
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN的基本块：dilated causal conv + residual
    
    优化改进:
    - 添加PreNorm（LayerNorm）提升训练稳定性
    - 添加SE Block（Squeeze-and-Excitation）增强通道注意力
    - 使用可学习的残差缩放因子（alpha scaling）
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # PreNorm: LayerNorm before first convolution
        self.norm1 = nn.LayerNorm(n_inputs)
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # PreNorm: LayerNorm before second convolution
        self.norm2 = nn.LayerNorm(n_outputs)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # SE Block: Squeeze-and-Excitation for channel attention
        # 通过全局池化和全连接层学习通道重要性
        se_reduction = max(n_outputs // 16, 4)  # 确保reduction不会太小
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_outputs, se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_reduction, n_outputs, 1),
            nn.Sigmoid()
        )
        
        # 残差连接：如果输入输出通道数不同，需要1x1卷积
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Alpha scaling: 可学习的残差缩放因子
        # 初始化为0.5，让模型学习最优的残差权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, channels, length)
        
        # PreNorm + Conv1
        # LayerNorm需要在(batch, length, channels)格式上操作
        out = x.transpose(1, 2)  # (batch, length, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (batch, channels, length)
        
        out = self.conv1(out)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # PreNorm + Conv2
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # SE Block: 学习通道注意力权重
        se_weight = self.se_block(out)  # (batch, channels, 1)
        out = out * se_weight  # 通道级别的特征重标定
        
        # Residual connection with learnable alpha scaling
        # out * alpha + res * (1 - alpha) 让模型学习最优残差比例
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out * self.alpha + res * (1 - self.alpha))


class TemporalConvNet(nn.Module):
    """
    TCN网络：多层TemporalBlock堆叠
    """
    def __init__(self, num_inputs, num_channels: List[int], kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长的dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                    stride=1, dilation=dilation_size,
                                    padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SpectralFeatureExtractor(nn.Module):
    """
    频谱特征提取器（增强版）
    从1001个频点中提取有意义的特征
    添加注意力机制以关注重要频段
    
    注意：使用LayerNorm替代BatchNorm，避免训练/测试不一致问题
    """
    def __init__(self, input_size=1001, output_size=128, dropout=0.2, use_attention=True):
        super(SpectralFeatureExtractor, self).__init__()
        
        self.use_attention = use_attention
        
        # 多尺度卷积：同时捕捉局部和全局频谱特征
        # 小kernel：捕捉相邻频点的细节
        self.conv_small = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        # 中kernel：捕捉中等范围的频谱模式
        self.conv_medium = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        # 大kernel：捕捉全局频谱趋势
        self.conv_large = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        
        # 使用LayerNorm替代BatchNorm (不依赖batch统计，训练/测试一致)
        self.ln1 = nn.LayerNorm(96)  # 32*3=96
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)
        
        # 添加频谱注意力
        if use_attention:
            self.channel_attn1 = ChannelAttention(96, reduction=8)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(96, 128, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(128)  # 128通道
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(dropout)
        
        # 添加频谱注意力
        if use_attention:
            self.channel_attn2 = ChannelAttention(128, reduction=8)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(128, output_size, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm(output_size)  # output_size通道
        
        # 自适应池化到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # 多尺度特征提取
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        
        # 拼接多尺度特征
        x = torch.cat([x_small, x_medium, x_large], dim=1)  # (batch_size, 96, input_size)
        # LayerNorm需要转置: (N,C,L) -> (N,L,C) -> normalize -> (N,C,L)
        x = x.transpose(1, 2)  # (batch_size, input_size, 96)
        x = F.relu(self.ln1(x))
        x = x.transpose(1, 2)  # (batch_size, 96, input_size)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第一层注意力
        if self.use_attention:
            x = self.channel_attn1(x)
        
        # 第二层
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (N,C,L) -> (N,L,C)
        x = F.relu(self.ln2(x))
        x = x.transpose(1, 2)  # (N,L,C) -> (N,C,L)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第二层注意力
        if self.use_attention:
            x = self.channel_attn2(x)
        
        # 第三层
        x = self.conv3(x)
        x = x.transpose(1, 2)  # (N,C,L) -> (N,L,C)
        x = F.relu(self.ln3(x))
        x = x.transpose(1, 2)  # (N,L,C) -> (N,C,L)
        x = self.adaptive_pool(x)  # (batch_size, output_size, 1)
        x = self.dropout3(x)
        
        x = x.squeeze(-1)  # (batch_size, output_size)
        
        return x


class SpectralTCN(nn.Module):
    """
    Spectral CNN + Temporal TCN 混合架构（增强版）
    
    步骤：
    1. 对每个时间步的频谱用CNN提取特征 (Spectral Feature Extractor with Attention)
    2. 将长T的embedding序列输入TCN (Temporal Convolutional Network)
    3. 添加时间注意力，关注重要时间步
    4. 智能聚合窗口中所有时间步的信息
    5. FC -> 输出
    
    适用场景：频谱时序数据预测
    """
    def __init__(self, input_size=1001, window_size=10, hidden_size=256,
                 num_layers=4, kernel_size=3, dropout=0.2, task_type='regression',
                 use_attention=True, aggregation='attention', use_dropblock=True, use_skip=True):
        super(SpectralTCN, self).__init__()
        
        self.task_type = task_type
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.aggregation = aggregation
        self.use_dropblock = use_dropblock
        self.use_skip = use_skip
        
        # 1. 频谱特征提取器（带注意力）
        self.spectral_extractor = SpectralFeatureExtractor(
            input_size=input_size,
            output_size=128,  # embedding维度
            dropout=dropout,
            use_attention=use_attention
        )
        
        # 2. TCN层配置：渐进式通道扩张策略
        # 从128平滑增长到hidden_size，避免突然跳跃造成信息瓶颈
        if num_layers == 1:
            tcn_channels = [hidden_size]
        elif num_layers == 2:
            tcn_channels = [128, hidden_size]
        elif num_layers == 3:
            # 例: hidden_size=256 -> [128, 192, 256]
            mid = int(128 + (hidden_size - 128) * 0.5)
            tcn_channels = [128, mid, hidden_size]
        elif num_layers == 4:
            # 例: hidden_size=256 -> [128, 160, 208, 256]
            # 例: hidden_size=320 -> [128, 176, 224, 320]
            step = (hidden_size - 128) / 3
            tcn_channels = [int(128 + i * step) for i in range(3)] + [hidden_size]
        elif num_layers == 5:
            # 例: hidden_size=320 -> [128, 176, 224, 272, 320]
            # 例: hidden_size=384 -> [128, 192, 256, 320, 384]
            step = (hidden_size - 128) / 4
            tcn_channels = [int(128 + i * step) for i in range(4)] + [hidden_size]
        else:  # num_layers >= 6
            # 线性插值生成中间通道数
            step = (hidden_size - 128) / (num_layers - 1)
            tcn_channels = [int(128 + i * step) for i in range(num_layers - 1)] + [hidden_size]
        
        self.tcn = TemporalConvNet(
            num_inputs=128,  # spectral extractor的输出维度
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # TCN的实际输出维度
        self.tcn_output_dim = tcn_channels[-1]
        
        # 2.5 DropBlock正则化（可选，比Dropout更强）
        if use_dropblock:
            self.dropblock = DropBlock1D(drop_prob=dropout, block_size=5)
        
        # 3. 时间注意力（关注重要时间步）
        if use_attention:
            self.temporal_attn = SpatialAttention(kernel_size=7)
        
        # 4. 智能时间步聚合
        self.temporal_aggregation = TemporalAggregation(
            hidden_size=self.tcn_output_dim,
            window_size=window_size,
            strategy=aggregation,
            dropout=dropout
        )
        
        # 5. 增强输出层（支持跳跃连接）
        # 如果使用跳跃连接，需要拼接频谱特征(128)和TCN输出(hidden_size)
        if use_skip:
            self.fc_input_dim = self.tcn_output_dim + 128  # TCN输出 + 频谱特征
        else:
            self.fc_input_dim = self.tcn_output_dim
        
        # 增强输出头：4层深度网络 with LayerNorm
        # 渐进降维：input_dim -> hidden_size -> hidden_size//2 -> hidden_size//4 -> 1
        self.output_head = nn.Sequential(
            # 第一层：扩展或保持维度
            nn.Linear(self.fc_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二层：开始降维
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第三层：继续降维
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # 最后一层降低dropout
            
            # 第四层：输出
            nn.Linear(hidden_size // 4, 1)
        )
        
        self._init_weights()
        
        if task_type != 'regression':
            raise ValueError(f"任务类型 {task_type} 暂不支持")
    
    def _init_weights(self):
        """使用Xavier初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _encode_spectrum(self, x):
        """
        内部方法：编码频谱特征
        
        Args:
            x: (batch_size, window_size, input_size) - 输入频谱数据
        
        Returns:
            aggregated_features: (batch_size, feature_dim) - 聚合后的特征
            spectral_features: (batch_size, window_size, 128) - 中间频谱特征（用于skip connection）
        """
        batch_size = x.size(0)
        window_size = x.size(1)
        
        # 1. 对每个时间步提取频谱特征
        spectral_features = []
        for t in range(window_size):
            feat = self.spectral_extractor(x[:, t, :])  # (batch_size, 128)
            spectral_features.append(feat)
        
        # (batch_size, window_size, 128)
        spectral_features = torch.stack(spectral_features, dim=1)
        
        # 2. TCN期望输入: (batch_size, channels, seq_len)
        tcn_input = spectral_features.transpose(1, 2)  # (batch_size, 128, window_size)
        
        # 3. TCN编码
        tcn_out = self.tcn(tcn_input)  # (batch_size, hidden_size, window_size)
        
        # 3.5 应用DropBlock正则化（可选）
        if self.use_dropblock and hasattr(self, 'dropblock'):
            tcn_out = self.dropblock(tcn_out)
        
        # 4. 时间注意力
        if self.use_attention:
            tcn_out = self.temporal_attn(tcn_out)
        
        # 5. 智能聚合所有时间步
        aggregated_tcn = self.temporal_aggregation(tcn_out)  # (batch_size, hidden_size)
        
        return aggregated_tcn, spectral_features
    
    def extract_features(self, x):
        """
        提取特征（用于Late Fusion）
        返回用于融合的特征向量，不经过最终输出层
        
        Args:
            x: (batch_size, window_size, input_size) - 输入频谱数据
        
        Returns:
            features: (batch_size, feature_dim) - 用于融合的特征
        """
        aggregated_tcn, spectral_features = self._encode_spectrum(x)
        
        # 应用skip connection逻辑（与forward一致）
        if self.use_skip:
            if self.aggregation == 'attention' and self.use_attention:
                spectral_tcn_format = spectral_features.transpose(1, 2)
                spectral_importance = spectral_tcn_format.mean(dim=1, keepdim=True)
                spectral_weights = torch.softmax(spectral_importance, dim=-1)
                spectral_pooled = torch.sum(
                    spectral_features * spectral_weights.transpose(1, 2),
                    dim=1
                )
            elif self.aggregation == 'last':
                spectral_pooled = spectral_features[:, -1, :]
            else:
                spectral_pooled = spectral_features.mean(dim=1)
            
            # 初始化skip_gate（如果还没有）
            if not hasattr(self, 'skip_gate'):
                self.skip_gate = nn.Sequential(
                    nn.Linear(128 + self.tcn_output_dim, 128 + self.tcn_output_dim),
                    nn.Sigmoid()
                ).to(spectral_pooled.device)
                nn.init.constant_(self.skip_gate[0].bias, 2.0)
            
            concat_features = torch.cat([spectral_pooled, aggregated_tcn], dim=1)
            gate = self.skip_gate(concat_features)
            features = concat_features * gate
        else:
            features = aggregated_tcn
        
        return features
    
    def get_feature_dim(self):
        """返回extract_features输出的特征维度"""
        if self.use_skip:
            return 128 + self.tcn_output_dim
        else:
            return self.tcn_output_dim
    
    def forward(self, x):
        # x: (batch_size, window_size, input_size)
        aggregated_tcn, spectral_features = self._encode_spectrum(x)
        
        # 5.5 优化跳跃连接：使用智能聚合策略
        if self.use_skip:
            # 优化1：使用与TCN相同的聚合策略处理频谱特征
            if self.aggregation == 'attention' and self.use_attention:
                spectral_tcn_format = spectral_features.transpose(1, 2)
                spectral_importance = spectral_tcn_format.mean(dim=1, keepdim=True)
                spectral_weights = torch.softmax(spectral_importance, dim=-1)
                spectral_pooled = torch.sum(
                    spectral_features * spectral_weights.transpose(1, 2),
                    dim=1
                )
            elif self.aggregation == 'last':
                spectral_pooled = spectral_features[:, -1, :]
            else:
                spectral_pooled = spectral_features.mean(dim=1)
            
            if not hasattr(self, 'skip_gate'):
                self.skip_gate = nn.Sequential(
                    nn.Linear(128 + self.tcn_output_dim, 128 + self.tcn_output_dim),
                    nn.Sigmoid()
                ).to(spectral_pooled.device)
                nn.init.constant_(self.skip_gate[0].bias, 2.0)
            
            concat_features = torch.cat([spectral_pooled, aggregated_tcn], dim=1)
            gate = self.skip_gate(concat_features)
            out = concat_features * gate
        else:
            out = aggregated_tcn
        
        # 6. 增强输出层（4层深度网络）
        out = self.output_head(out)
        
        if self.task_type == 'regression':
            return out.squeeze(-1)
        return out


def create_tcn_model(config, window_size):
    """
    创建TCN模型
    
    Args:
        config: ModelConfig对象
        window_size: 窗口大小
    
    Returns:
        model: PyTorch模型
    """
    arch = config.architecture.upper()
    
    kwargs = {
        'input_size': config.input_size,
        'window_size': window_size,
        'hidden_size': config.hidden_size,
        'dropout': config.dropout,
        'task_type': config.task_type,
    }
    
    if arch == 'TCN':
        # Spectral CNN + TCN混合架构（推荐，可选注意力机制）
        kwargs['num_layers'] = config.num_layers if hasattr(config, 'num_layers') else 4
        kwargs['kernel_size'] = 3  # TCN kernel size
        kwargs['use_attention'] = getattr(config, 'use_attention', True)  # 注意力开关
        kwargs['aggregation'] = getattr(config, 'aggregation', 'attention')  # 时间步聚合策略
        kwargs['use_dropblock'] = getattr(config, 'use_dropblock', True)
        kwargs['use_skip'] = getattr(config, 'use_skip', True)
        return SpectralTCN(**kwargs)
    else:
        raise ValueError(f"不支持的TCN架构: {arch}")


# 导出
__all__ = [
    'SpectralTCN',
    'TemporalConvNet',
    'SpectralFeatureExtractor',
    'create_tcn_model',
]

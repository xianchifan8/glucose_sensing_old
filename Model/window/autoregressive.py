"""
Auto-Regressive (自回归) 模块
为血糖预测模型添加自回归能力，使用历史血糖值辅助预测

设计原则：
1. 低耦合：作为包装器/混入模块，不修改基础TCN架构
2. 可扩展：后续TCN架构修改会自动作用到AR模式
3. 灵活性：支持多种历史血糖融合方式

重要说明 - 训练与推理的区别：
========================================
【训练阶段】- Teacher Forcing
  - 使用真实的历史血糖标签作为输入
  - 这是序列预测的标准做法，加速收敛
  - glucose_history[i] = [label[i-N], ..., label[i-1]]（都是真实值）

【推理阶段】- Auto-Regressive Inference
  - 第一次预测：使用真实历史 → 得到预测值pred[0]
  - 第二次预测：历史更新为[真实[1:], pred[0]] → 得到pred[1]
  - 第三次预测：历史更新为[真实[2:], pred[0], pred[1]] → ...
  - 逐步用预测值替换历史（滚动预测）

【Scheduled Sampling】- 缓解训练-推理不一致
  - 训练时以一定概率使用模型预测值替代真实值
  - 概率随训练进度增加，逐渐从Teacher Forcing过渡到自回归
  - 可减少Exposure Bias问题
========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlucoseHistoryEncoder(nn.Module):
    """
    历史血糖编码器
    将历史血糖序列编码为特征向量
    
    支持多种编码策略：
    1. mlp: 简单MLP编码
    2. lstm: 使用LSTM捕捉时序依赖
    3. attention: 使用自注意力机制学习重要性
    """
    def __init__(self, history_len, hidden_dim=32, output_dim=32, 
                 strategy='lstm', dropout=0.1):
        """
        Args:
            history_len: 历史血糖序列长度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            strategy: 编码策略 'mlp', 'lstm', 'attention'
            dropout: Dropout率
        """
        super(GlucoseHistoryEncoder, self).__init__()
        
        self.history_len = history_len
        self.output_dim = output_dim
        self.strategy = strategy
        
        if strategy == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(history_len, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True)
            )
        
        elif strategy == 'lstm':
            self.lstm = nn.LSTM(
                input_size=1,  # 每个时间步一个血糖值
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if 2 > 1 else 0,
                bidirectional=False  # 因果模型，只看过去
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        
        elif strategy == 'attention':
            # 位置编码
            self.pos_embedding = nn.Parameter(torch.randn(1, history_len, hidden_dim) * 0.01)
            # 值嵌入
            self.value_embedding = nn.Linear(1, hidden_dim)
            # 自注意力
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True)
            )
        
        else:
            raise ValueError(f"不支持的编码策略: {strategy}")
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, glucose_history):
        """
        Args:
            glucose_history: (batch_size, history_len) - 历史血糖序列
        
        Returns:
            encoded: (batch_size, output_dim) - 编码后的特征
        """
        batch_size = glucose_history.size(0)
        
        if self.strategy == 'mlp':
            return self.encoder(glucose_history)
        
        elif self.strategy == 'lstm':
            # (batch, history_len) -> (batch, history_len, 1)
            x = glucose_history.unsqueeze(-1)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # 使用最后一个隐藏状态
            encoded = self.fc(h_n[-1])
            return self.norm(encoded)
        
        elif self.strategy == 'attention':
            # (batch, history_len) -> (batch, history_len, 1)
            x = glucose_history.unsqueeze(-1)
            # 值嵌入 + 位置编码
            x = self.value_embedding(x) + self.pos_embedding
            # 自注意力
            attn_out, _ = self.attention(x, x, x)
            # 全局平均池化
            pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)
            return self.fc(pooled)


class AutoRegressiveWrapper(nn.Module):
    """
    自回归包装器
    将任意基础模型包装为支持自回归的模型
    
    融合策略：
    1. concat: 将历史血糖特征与频谱特征拼接后送入输出层
    2. film: 使用FiLM调制频谱特征
    3. gating: 使用门控机制动态融合
    """
    def __init__(self, base_model, ar_history_len, fusion_strategy='concat',
                 glucose_encoder_dim=32, dropout=0.1, has_aux=False):
        """
        Args:
            base_model: 基础TCN模型（需有extract_features和get_feature_dim方法）
            ar_history_len: 历史血糖序列长度
            fusion_strategy: 融合策略 'concat', 'film', 'gating'
            glucose_encoder_dim: 血糖编码器输出维度
            dropout: Dropout率
            has_aux: 是否有aux传感器数据
        """
        super(AutoRegressiveWrapper, self).__init__()
        
        self.base_model = base_model
        self.ar_history_len = ar_history_len
        self.fusion_strategy = fusion_strategy
        self.has_aux = has_aux
        
        # 获取基础模型的特征维度
        if hasattr(base_model, 'get_feature_dim'):
            self.base_feature_dim = base_model.get_feature_dim()
        elif hasattr(base_model, 'fc_input_dim'):
            self.base_feature_dim = base_model.fc_input_dim
        else:
            # 回退到hidden_size
            self.base_feature_dim = base_model.hidden_size
        
        # 历史血糖编码器
        self.glucose_encoder = GlucoseHistoryEncoder(
            history_len=ar_history_len,
            hidden_dim=glucose_encoder_dim,
            output_dim=glucose_encoder_dim,
            strategy='lstm',  # 使用LSTM捕捉趋势
            dropout=dropout
        )
        self.glucose_feature_dim = glucose_encoder_dim
        
        # 根据融合策略创建输出层
        if fusion_strategy == 'concat':
            self.fused_dim = self.base_feature_dim + glucose_encoder_dim
            self._build_output_head(self.fused_dim, dropout)
        
        elif fusion_strategy == 'film':
            # FiLM: γ * feature + β
            self.film_gamma = nn.Linear(glucose_encoder_dim, self.base_feature_dim)
            self.film_beta = nn.Linear(glucose_encoder_dim, self.base_feature_dim)
            self.fused_dim = self.base_feature_dim
            self._build_output_head(self.fused_dim, dropout)
        
        elif fusion_strategy == 'gating':
            # 门控融合：学习历史血糖对当前预测的影响程度
            self.gate = nn.Sequential(
                nn.Linear(self.base_feature_dim + glucose_encoder_dim, self.base_feature_dim),
                nn.Sigmoid()
            )
            self.fused_dim = self.base_feature_dim
            self._build_output_head(self.fused_dim, dropout)
        
        else:
            raise ValueError(f"不支持的融合策略: {fusion_strategy}")
        
        # 记录任务类型
        self.task_type = getattr(base_model, 'task_type', 'regression')
        
        # Scheduled Sampling配置（用于缓解训练-推理不一致）
        # sampling_prob: 使用预测值替代真实值的概率
        # 训练时可以通过set_sampling_prob动态调整
        self.sampling_prob = 0.0  # 默认不使用（纯Teacher Forcing）
    
    def _build_output_head(self, input_dim, dropout):
        """构建输出头"""
        hidden = max(input_dim // 2, 32)
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )
    
    def get_feature_dim(self):
        """返回融合后的特征维度"""
        return self.fused_dim
    
    def set_sampling_prob(self, prob):
        """
        设置Scheduled Sampling概率
        
        训练时可以根据epoch逐渐增加此概率：
        - epoch 0-10: prob = 0.0 (纯Teacher Forcing)
        - epoch 10-30: prob = 0.0 → 0.3 (逐渐引入)
        - epoch 30+: prob = 0.3 (稳定)
        
        Args:
            prob: float, 0.0~1.0, 使用预测值的概率
        """
        self.sampling_prob = max(0.0, min(1.0, prob))
    
    def get_scheduled_sampling_prob(self, current_epoch, warmup_epochs=10, 
                                     max_prob=0.3, increase_epochs=20):
        """
        根据当前epoch计算推荐的sampling概率
        
        Args:
            current_epoch: 当前训练epoch
            warmup_epochs: 前N个epoch不使用sampling（纯Teacher Forcing）
            max_prob: 最大sampling概率
            increase_epochs: 从0增加到max_prob需要的epoch数
        
        Returns:
            recommended_prob: 推荐的sampling概率
        """
        if current_epoch < warmup_epochs:
            return 0.0
        
        progress = (current_epoch - warmup_epochs) / increase_epochs
        return min(max_prob, max_prob * progress)
    
    def extract_features(self, spectrum, aux=None, glucose_history=None):
        """
        提取融合后的特征（用于Late Fusion等场景）
        
        Args:
            spectrum: 频谱数据
            aux: 辅助传感器数据（可选）
            glucose_history: 历史血糖序列
        
        Returns:
            fused_features: 融合后的特征
        """
        # 从基础模型提取特征
        if hasattr(self.base_model, 'extract_features'):
            if self.has_aux and aux is not None:
                base_features = self.base_model.extract_features(spectrum, aux)
            else:
                base_features = self.base_model.extract_features(spectrum)
        else:
            # 如果基础模型没有extract_features，直接使用forward
            raise RuntimeError("基础模型必须实现extract_features方法")
        
        # 编码历史血糖
        glucose_features = self.glucose_encoder(glucose_history)
        
        # 融合
        if self.fusion_strategy == 'concat':
            fused = torch.cat([base_features, glucose_features], dim=-1)
        
        elif self.fusion_strategy == 'film':
            gamma = self.film_gamma(glucose_features)  # (batch, base_dim)
            beta = self.film_beta(glucose_features)
            fused = gamma * base_features + beta
        
        elif self.fusion_strategy == 'gating':
            concat = torch.cat([base_features, glucose_features], dim=-1)
            gate = self.gate(concat)
            fused = base_features * gate
        
        return fused
    
    def forward(self, spectrum, *args, glucose_history=None, aux=None):
        """
        前向传播（训练模式：使用真实历史标签，即Teacher Forcing）
        
        注意：这是训练时使用的方法。推理时如果需要自回归预测，
        请使用 predict_autoregressive() 或 predict_batch_autoregressive()
        
        支持多种调用方式：
        1. forward(spectrum, glucose_history) - 无aux
        2. forward(spectrum, aux, glucose_history) - 有aux
        3. forward(spectrum, glucose_history=xxx, aux=xxx) - 关键字参数
        
        Args:
            spectrum: 频谱数据 (batch, window, features) 或 (batch, features)
            args: 位置参数（可能是aux或glucose_history）
            glucose_history: 历史血糖序列 (batch, ar_history) - 训练时使用真实标签
            aux: 辅助传感器数据（可选）
        
        Returns:
            output: 预测的血糖值 (batch,)
        """
        # 解析参数
        if len(args) == 1:
            if glucose_history is None:
                # 假设args[0]是glucose_history
                glucose_history = args[0]
            else:
                # 假设args[0]是aux
                aux = args[0]
        elif len(args) == 2:
            # args = (aux, glucose_history)
            aux = args[0]
            glucose_history = args[1]
        
        if glucose_history is None:
            raise ValueError("glucose_history不能为None")
        
        # 提取融合特征
        fused = self.extract_features(spectrum, aux, glucose_history)
        
        # 输出层
        out = self.output_head(fused)
        
        if self.task_type == 'regression':
            return out.squeeze(-1)
        return out
    
    def get_last_glucose(self, glucose_history):
        """
        获取历史血糖序列中最后一个值
        用于计算平滑性正则化
        
        Args:
            glucose_history: (batch, ar_history)
        
        Returns:
            last_glucose: (batch,)
        """
        return glucose_history[:, -1]
    
    @torch.no_grad()
    def predict_autoregressive(self, spectrum_sequence, initial_history, 
                                aux_sequence=None, return_all_histories=False):
        """
        真正的自回归推理：逐步预测，用预测值滚动更新历史
        
        这是推理时应该使用的方法！
        与forward不同，这里会用模型自己的预测值更新历史。
        
        Args:
            spectrum_sequence: (seq_len, window_size, n_features) 或 (seq_len, n_features)
                              - 按时间顺序排列的频谱数据序列
            initial_history: (ar_history,) - 初始的真实历史血糖值
            aux_sequence: (seq_len, aux_features) 或 (seq_len, window_size, aux_features)
                         - 按时间顺序排列的辅助传感器数据（可选）
            return_all_histories: bool - 是否返回每一步的历史（用于分析）
        
        Returns:
            predictions: (seq_len,) - 预测的血糖序列
            histories: (seq_len, ar_history) - 每一步使用的历史（仅当return_all_histories=True）
        
        使用示例:
            # 假设有10个连续时间点的频谱数据需要预测
            spectrum_seq = test_spectrums[100:110]  # (10, window, features)
            initial_hist = real_glucose[100-ar_history:100]  # 初始真实历史
            
            predictions = model.predict_autoregressive(spectrum_seq, initial_hist)
        """
        self.eval()  # 确保在评估模式
        
        device = next(self.parameters()).device
        seq_len = len(spectrum_sequence)
        
        # 转换为tensor并移到正确设备
        if isinstance(spectrum_sequence, np.ndarray):
            spectrum_sequence = torch.FloatTensor(spectrum_sequence)
        spectrum_sequence = spectrum_sequence.to(device)
        
        if isinstance(initial_history, np.ndarray):
            initial_history = torch.FloatTensor(initial_history)
        current_history = initial_history.to(device).clone()
        
        if aux_sequence is not None:
            if isinstance(aux_sequence, np.ndarray):
                aux_sequence = torch.FloatTensor(aux_sequence)
            aux_sequence = aux_sequence.to(device)
        
        predictions = []
        histories = [] if return_all_histories else None
        
        for t in range(seq_len):
            # 记录当前历史
            if return_all_histories:
                histories.append(current_history.cpu().numpy().copy())
            
            # 准备当前时间步的输入（添加batch维度）
            spectrum_t = spectrum_sequence[t].unsqueeze(0)  # (1, ...)
            history_t = current_history.unsqueeze(0)  # (1, ar_history)
            
            aux_t = None
            if aux_sequence is not None:
                aux_t = aux_sequence[t].unsqueeze(0)  # (1, ...)
            
            # 预测当前时间步
            pred = self.forward(spectrum_t, glucose_history=history_t, aux=aux_t)
            pred_value = pred.item()
            predictions.append(pred_value)
            
            # 更新历史：移除最旧的，添加新预测
            # [old1, old2, ..., oldN] -> [old2, ..., oldN, pred]
            current_history = torch.cat([
                current_history[1:], 
                torch.tensor([pred_value], device=device)
            ])
        
        predictions = np.array(predictions)
        
        if return_all_histories:
            histories = np.array(histories)
            return predictions, histories
        return predictions
    
    @torch.no_grad()
    def predict_batch_autoregressive(self, spectrum_sequences, initial_histories,
                                      aux_sequences=None):
        """
        批量自回归推理（更高效）
        
        Args:
            spectrum_sequences: (batch, seq_len, ...) - 批量频谱序列
            initial_histories: (batch, ar_history) - 批量初始历史
            aux_sequences: (batch, seq_len, ...) - 批量aux序列（可选）
        
        Returns:
            predictions: (batch, seq_len) - 批量预测结果
        """
        self.eval()
        
        device = next(self.parameters()).device
        batch_size, seq_len = spectrum_sequences.shape[:2]
        
        # 转换为tensor
        if isinstance(spectrum_sequences, np.ndarray):
            spectrum_sequences = torch.FloatTensor(spectrum_sequences)
        spectrum_sequences = spectrum_sequences.to(device)
        
        if isinstance(initial_histories, np.ndarray):
            initial_histories = torch.FloatTensor(initial_histories)
        current_histories = initial_histories.to(device).clone()
        
        if aux_sequences is not None:
            if isinstance(aux_sequences, np.ndarray):
                aux_sequences = torch.FloatTensor(aux_sequences)
            aux_sequences = aux_sequences.to(device)
        
        all_predictions = []
        
        for t in range(seq_len):
            # 获取当前时间步的输入
            spectrum_t = spectrum_sequences[:, t]  # (batch, ...)
            
            aux_t = None
            if aux_sequences is not None:
                aux_t = aux_sequences[:, t]
            
            # 批量预测
            preds = self.forward(spectrum_t, glucose_history=current_histories, aux=aux_t)
            all_predictions.append(preds)
            
            # 批量更新历史
            current_histories = torch.cat([
                current_histories[:, 1:],
                preds.unsqueeze(1)
            ], dim=1)
        
        # (seq_len, batch) -> (batch, seq_len)
        predictions = torch.stack(all_predictions, dim=1)
        return predictions.cpu().numpy()


def create_ar_model(base_model, config):
    """
    创建自回归模型
    
    Args:
        base_model: 基础TCN模型
        config: 配置对象，需包含:
            - ar_glucose_history: 历史血糖数量
            - data_fusion: 是否使用数据融合
    
    Returns:
        AutoRegressiveWrapper
    
    使用说明:
    ========
    训练时（使用真实历史 - Teacher Forcing）:
        model = create_ar_model(base_model, config)
        for spectrum, aux, glucose_history, target in dataloader:
            pred = model(spectrum, aux=aux, glucose_history=glucose_history)
            loss = criterion(pred, target)
    
    推理时（使用预测值滚动更新历史）:
        model.eval()
        # 单序列推理
        predictions = model.predict_autoregressive(
            spectrum_sequence,    # (seq_len, window, features)
            initial_history,      # (ar_history,) 初始真实历史
            aux_sequence          # (seq_len, aux_features) 可选
        )
        
        # 批量推理
        predictions = model.predict_batch_autoregressive(
            spectrum_sequences,   # (batch, seq_len, ...)
            initial_histories,    # (batch, ar_history)
            aux_sequences         # (batch, seq_len, ...) 可选
        )
    """
    ar_history = config.data.ar_glucose_history
    # 只有 Late Fusion 模式才需要 aux 参数
    # Early Fusion 的 aux 已经和 spectrum 合并，不需要单独处理
    has_aux = config.data.data_fusion and config.data.fusion_stage == 'late'
    dropout = config.model.dropout
    
    # 从config读取fusion策略和encoder维度（如果没有则使用默认值）
    fusion_strategy = getattr(config.data, 'ar_fusion_strategy', 'gating')
    glucose_encoder_dim = getattr(config.data, 'ar_glucose_encoder_dim', 128)
    
    return AutoRegressiveWrapper(
        base_model=base_model,
        ar_history_len=ar_history,
        fusion_strategy=fusion_strategy,
        glucose_encoder_dim=glucose_encoder_dim,
        dropout=dropout,
        has_aux=has_aux
    )


# 导出
__all__ = [
    'GlucoseHistoryEncoder',
    'AutoRegressiveWrapper', 
    'create_ar_model',
]

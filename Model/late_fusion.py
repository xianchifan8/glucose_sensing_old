"""
Late Fusion 模块
特征层融合：频谱特征和辅助传感器特征在模型内部分别处理后再融合

设计原则：
1. 低耦合：Late fusion逻辑完全独立，不影响early fusion
2. 可扩展：支持多种fusion方法（concat, attention, gated等）
3. 通用性：适配所有模型架构（MLP, CNN, TCN, Transformer）
4. 无硬编码：维度从config动态获取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _residual_scale_raw_from_effective(initial_scale: float, min_scale: float) -> torch.Tensor:
    """Return raw sigmoid parameter whose bounded value starts at initial_scale."""
    if min_scale >= initial_scale:
        normalized = 0.5
    else:
        normalized = (initial_scale - min_scale) / max(1.0 - min_scale, 1e-6)
    normalized = min(max(float(normalized), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(normalized, dtype=torch.float32))


class AuxFeatureEncoder(nn.Module):
    """
    辅助传感器特征编码器
    将低维aux特征编码为与主特征兼容的表示
    
    设计考量：
    - Aux特征维度较小（~14维），需要适当扩展
    - 使用LayerNorm而非BatchNorm，保持训练/测试一致性
    """
    def __init__(self, input_dim: int, output_dim: int = 32, dropout: float = 0.2):
        super(AuxFeatureEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 多层MLP编码器
        hidden_dim = max(input_dim * 2, output_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, aux_dim) for instant mode
               (batch_size, window_size, aux_dim) for window mode
        Returns:
            encoded: (batch_size, output_dim) or (batch_size, window_size, output_dim)
        """
        if x.dim() == 2:
            # Instant mode: (batch, aux_dim) -> (batch, output_dim)
            return self.encoder(x)
        elif x.dim() == 3:
            # Window mode: (batch, window, aux_dim) -> (batch, window, output_dim)
            batch_size, window_size, aux_dim = x.shape
            x_flat = x.view(-1, aux_dim)
            encoded = self.encoder(x_flat)
            return encoded.view(batch_size, window_size, self.output_dim)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")


class LateFusionHead(nn.Module):
    """
    Late Fusion 融合头
    将spectrum特征和aux特征融合后输出预测
    
    支持多种融合方法：
    - concat: 简单拼接 + FC (基线方法)
    - cross_attention: 交叉注意力融合，spectrum和aux互相关注
    - film: Feature-wise Linear Modulation，用aux调制spectrum特征
    
    方法详解：
    
    1. concat (基线):
       - 直接拼接两个特征向量
       - spectrum(128) + aux(32) → 160维 → MLP
       - 简单但无法建模特征间复杂交互
    
    2. cross_attention (推荐):
       - 原理：让两个模态互相关注，学习哪些特征相关
       - 实现：
         a) Spectrum关注Aux: Q_spec × K_aux → 注意力权重 → 加权V_aux
         b) Aux关注Spectrum: Q_aux × K_spec → 注意力权重 → 加权V_spec
         c) 拼接两个attended特征 → MLP输出
       - 优点：自适应学习模态间关系，效果通常最好
       - 适用：当aux传感器信号与spectrum有复杂相互作用时
    
    3. film (调制方法):
       - 原理：用aux特征生成调制参数(gamma, beta)，调制spectrum
       - 实现：
         a) aux → MLP → gamma(128), beta(128)
         b) spectrum_modulated = gamma ⊙ spectrum + beta
         c) spectrum_modulated → MLP → 输出
       - 优点：物理意义清晰(aux作为环境条件信号)
       - 适用：aux是环境因素(温度、运动状态)，spectrum是主要信号
    """
    def __init__(
        self,
        spectrum_dim: int,
        aux_dim: int,
        hidden_size: int = 256,
        fusion_method: str = 'concat',
        dropout: float = 0.2,
        task_type: str = 'regression',
        residual_scale_mode: str = 'sigmoid',
        residual_scale_min: float = 0.02,
        correction_scale_mode: str = 'none'
    ):
        super(LateFusionHead, self).__init__()
        
        self.spectrum_dim = spectrum_dim
        self.aux_dim = aux_dim
        self.fusion_method = fusion_method
        self.task_type = task_type
        self.residual_scale_mode = residual_scale_mode
        self.residual_scale_min = float(residual_scale_min)
        self.correction_scale_mode = correction_scale_mode
        self.collect_diagnostics = False
        self._diagnostic_batches = []
        self.last_correction_loss = None
        
        if fusion_method == 'concat':
            # ============================================================
            # 方法1: Concat (简单拼接基线)
            # ============================================================
            # 原理：直接拼接spectrum和aux特征，通过MLP学习融合
            # 优点：简单、快速、容易训练
            # 缺点：只是简单拼接，无法显式建模特征间的交互关系
            # ============================================================
            fused_dim = spectrum_dim + aux_dim
            
            self.output_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
        
        elif fusion_method == 'cross_attention':
            # ============================================================
            # 方法2: Cross-Attention (交叉注意力)
            # ============================================================
            # 原理：让spectrum和aux互相"关注"对方，学习模态间的关联
            # 
            # 详细步骤：
            # 1. Spectrum关注Aux（Spec → Query, Aux → Key/Value）：
            #    - 学习"aux中哪些信息对解释spectrum有用"
            #    - 例如：体温升高时关注到特定频谱变化
            # 
            # 2. Aux关注Spectrum（Aux → Query, Spec → Key/Value）：
            #    - 学习"spectrum中哪些特征受aux影响"
            #    - 例如：某些频段受运动状态影响
            # 
            # 3. 双向注意力结果拼接后预测
            # 
            # 数学公式：
            #   Attention(Q, K, V) = softmax(Q·K^T / √d) · V
            #   spectrum_attended = Attention(Q_spec, K_aux, V_aux)
            #   aux_attended = Attention(Q_aux, K_spec, V_spec)
            #   output = MLP([spectrum_attended; aux_attended])
            # 
            # 优点：
            # - 自适应学习模态关联，可解释性强
            # - 注意力权重可视化，了解哪些特征相互影响
            # - 效果通常优于简单拼接
            # ============================================================
            
            # Spectrum关注Aux的映射
            self.spectrum_to_query = nn.Linear(spectrum_dim, hidden_size)
            self.aux_to_key = nn.Linear(aux_dim, hidden_size)
            self.aux_to_value = nn.Linear(aux_dim, hidden_size)
            
            # Aux关注Spectrum的映射
            self.aux_to_query = nn.Linear(aux_dim, hidden_size)
            self.spectrum_to_key = nn.Linear(spectrum_dim, hidden_size)
            self.spectrum_to_value = nn.Linear(spectrum_dim, hidden_size)
            
            # 缩放因子，防止点积过大导致softmax梯度消失
            self.scale = hidden_size ** -0.5
            self.dropout_attn = nn.Dropout(dropout)
            
            # 融合后的输出头
            fused_dim = hidden_size * 2  # spectrum_attended + aux_attended
            self.output_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
        
        elif fusion_method == 'film':
            # ============================================================
            # 方法3: FiLM (Feature-wise Linear Modulation)
            # ============================================================
            # 原理：用aux特征调制spectrum特征，类似于"条件归一化"
            # 
            # 详细步骤：
            # 1. 用aux生成调制参数：
            #    aux → MLP → gamma(spectrum_dim维), beta(spectrum_dim维)
            # 
            # 2. 逐元素调制spectrum：
            #    spectrum_modulated[i] = gamma[i] × spectrum[i] + beta[i]
            #    其中i遍历spectrum的每个维度
            # 
            # 3. 调制后的spectrum通过MLP输出预测
            # 
            # 物理意义：
            # - gamma: 缩放因子，控制spectrum每个维度的"增益"
            #   例如：运动时某些频段被放大(gamma>1)或抑制(gamma<1)
            # 
            # - beta: 偏移量，控制spectrum每个维度的"基线"
            #   例如：温度影响导致频谱整体偏移
            # 
            # 优点：
            # - 物理意义清晰：aux作为"条件"调制主信号
            # - 参数效率高：只需要2×spectrum_dim个调制参数
            # - 适合aux是环境/状态变量的场景
            # 
            # 适用场景：
            # - aux代表环境因素(温度、湿度、运动)
            # - spectrum是主要测量信号
            # - 需要建模"在不同条件下spectrum如何变化"
            # ============================================================
            
            # aux → gamma (缩放参数)
            self.film_gamma = nn.Sequential(
                nn.Linear(aux_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, spectrum_dim)
            )
            
            # aux → beta (偏移参数)
            self.film_beta = nn.Sequential(
                nn.Linear(aux_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, spectrum_dim)
            )
            
            # 调制后的输出头
            self.output_head = nn.Sequential(
                nn.Linear(spectrum_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )

        elif fusion_method == 'gated_residual':
            # ============================================================
            # 方法4: Gated Residual
            # ============================================================
            # aux不直接覆盖主模态，而是生成一个有界残差修正。
            # 适合PPG/ICM这类质量/运动状态信号：有用时修正spectrum，
            # 无用或噪声大时门控会压低其影响。
            # ============================================================
            fused_dim = spectrum_dim + aux_dim

            self.residual_net = nn.Sequential(
                nn.Linear(fused_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, spectrum_dim),
                nn.Tanh()
            )
            self.gate_net = nn.Sequential(
                nn.Linear(fused_dim, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, spectrum_dim),
                nn.Sigmoid()
            )
            if self.residual_scale_mode == 'clamp':
                self.residual_scale = nn.Parameter(torch.tensor(0.1))
            else:
                self.residual_scale = nn.Parameter(
                    _residual_scale_raw_from_effective(0.1, self.residual_scale_min)
                )

            self.output_head = nn.Sequential(
                nn.Linear(spectrum_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
        
        else:
            raise NotImplementedError(f"Late fusion method '{fusion_method}' not implemented yet")
        
        # 记录融合后的特征维度（用于后续模块如AR Wrapper）
        if fusion_method == 'concat':
            self.output_dim = spectrum_dim + aux_dim
        elif fusion_method == 'cross_attention':
            self.output_dim = hidden_size * 2
        elif fusion_method == 'film':
            self.output_dim = spectrum_dim
        elif fusion_method == 'gated_residual':
            self.output_dim = spectrum_dim
        else:
            self.output_dim = hidden_size  # 默认值
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_diagnostics_enabled(self, enabled: bool):
        self.collect_diagnostics = bool(enabled)

    def reset_diagnostics(self):
        self._diagnostic_batches = []

    def get_diagnostics(self):
        if not self._diagnostic_batches:
            return {}
        keys = self._diagnostic_batches[0].keys()
        return {
            key: float(torch.tensor([batch[key] for batch in self._diagnostic_batches]).mean().item())
            for key in keys
        }

    def get_correction_loss(self):
        return self.last_correction_loss

    def _bounded_residual_scale(self):
        if self.residual_scale_mode == 'clamp':
            return torch.clamp(self.residual_scale, 0.0, 1.0)
        min_scale = min(max(self.residual_scale_min, 0.0), 0.99)
        return min_scale + (1.0 - min_scale) * torch.sigmoid(self.residual_scale)

    def _correction_amplifier(self, spectrum_features: torch.Tensor) -> torch.Tensor:
        if self.correction_scale_mode == 'spectrum_rms':
            return torch.sqrt(torch.mean(spectrum_features.detach().pow(2), dim=-1, keepdim=True) + 1e-8)
        if self.correction_scale_mode == 'multiplicative':
            return spectrum_features
        return torch.ones(
            spectrum_features.shape[0],
            1,
            dtype=spectrum_features.dtype,
            device=spectrum_features.device
        )

    def _record_gated_residual_diagnostics(
        self,
        spectrum_features: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
        correction: torch.Tensor,
        scale: torch.Tensor,
        scale_raw: Optional[torch.Tensor] = None,
        correction_amplifier: Optional[torch.Tensor] = None
    ):
        if not self.collect_diagnostics:
            return
        with torch.no_grad():
            eps = 1e-8
            gate_flat = gate.detach().reshape(-1)
            correction_norm = correction.detach().norm(dim=-1)
            spectrum_norm = spectrum_features.detach().norm(dim=-1)
            residual_norm = residual.detach().norm(dim=-1)
            correction_abs_mean = correction.detach().abs().mean()
            spectrum_abs_mean = spectrum_features.detach().abs().mean()
            residual_abs_mean = residual.detach().abs().mean()
            amp_mean = correction_amplifier.detach().abs().mean() if correction_amplifier is not None else torch.tensor(1.0)
            self._diagnostic_batches.append({
                'residual_scale': float(scale.detach().cpu().item()),
                'residual_scale_raw': float(scale_raw.detach().cpu().item()) if scale_raw is not None else 0.0,
                'residual_scale_min': float(self.residual_scale_min if self.residual_scale_mode != 'clamp' else 0.0),
                'correction_amplifier_mean': float(amp_mean.cpu().item()),
                'gate_mean': float(gate_flat.mean().cpu().item()),
                'gate_std': float(gate_flat.std(unbiased=False).cpu().item()),
                'gate_p25': float(torch.quantile(gate_flat, 0.25).cpu().item()),
                'gate_p50': float(torch.quantile(gate_flat, 0.50).cpu().item()),
                'gate_p75': float(torch.quantile(gate_flat, 0.75).cpu().item()),
                'residual_norm': float(residual_norm.mean().cpu().item()),
                'correction_norm': float(correction_norm.mean().cpu().item()),
                'spectrum_norm': float(spectrum_norm.mean().cpu().item()),
                'correction_to_spectrum_ratio': float((correction_norm / (spectrum_norm + eps)).mean().cpu().item()),
                'residual_abs_mean': float(residual_abs_mean.cpu().item()),
                'correction_abs_mean': float(correction_abs_mean.cpu().item()),
                'spectrum_abs_mean': float(spectrum_abs_mean.cpu().item()),
                'correction_abs_to_spectrum_abs': float((correction_abs_mean / (spectrum_abs_mean + eps)).cpu().item()),
            })
    
    def forward(
        self,
        spectrum_features: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spectrum_features: (batch_size, spectrum_dim) - 频谱编码后的特征
            aux_features: (batch_size, aux_dim) - aux编码后的特征
        Returns:
            output: (batch_size,) for regression
        """
        if self.fusion_method == 'concat':
            self.last_correction_loss = None
            # ============================================================
            # Concat融合：简单拼接
            # ============================================================
            fused = torch.cat([spectrum_features, aux_features], dim=-1)
            output = self.output_head(fused)
        
        elif self.fusion_method == 'cross_attention':
            self.last_correction_loss = None
            # ============================================================
            # Cross-Attention融合：双向注意力
            # ============================================================
            batch_size = spectrum_features.size(0)
            
            # 步骤1: Spectrum关注Aux
            # 计算："从aux中提取对spectrum有用的信息"
            q_spec = self.spectrum_to_query(spectrum_features).unsqueeze(1)  # (B, 1, hidden)
            k_aux = self.aux_to_key(aux_features).unsqueeze(1)  # (B, 1, hidden)
            v_aux = self.aux_to_value(aux_features).unsqueeze(1)  # (B, 1, hidden)
            
            # 注意力分数: Q·K^T / √d
            attn_weights_spec = torch.matmul(q_spec, k_aux.transpose(-2, -1)) * self.scale
            attn_weights_spec = torch.softmax(attn_weights_spec, dim=-1)  # 归一化
            attn_weights_spec = self.dropout_attn(attn_weights_spec)
            
            # 加权求和: Attention × V
            spectrum_attended = torch.matmul(attn_weights_spec, v_aux).squeeze(1)  # (B, hidden)
            
            # 步骤2: Aux关注Spectrum
            # 计算："从spectrum中提取受aux影响的部分"
            q_aux = self.aux_to_query(aux_features).unsqueeze(1)  # (B, 1, hidden)
            k_spec = self.spectrum_to_key(spectrum_features).unsqueeze(1)  # (B, 1, hidden)
            v_spec = self.spectrum_to_value(spectrum_features).unsqueeze(1)  # (B, 1, hidden)
            
            attn_weights_aux = torch.matmul(q_aux, k_spec.transpose(-2, -1)) * self.scale
            attn_weights_aux = torch.softmax(attn_weights_aux, dim=-1)
            attn_weights_aux = self.dropout_attn(attn_weights_aux)
            
            aux_attended = torch.matmul(attn_weights_aux, v_spec).squeeze(1)  # (B, hidden)
            
            # 步骤3: 融合两个attended特征
            fused = torch.cat([spectrum_attended, aux_attended], dim=-1)
            output = self.output_head(fused)
        
        elif self.fusion_method == 'film':
            self.last_correction_loss = None
            # ============================================================
            # FiLM融合：特征调制
            # ============================================================
            # 步骤1: 从aux生成调制参数
            gamma = self.film_gamma(aux_features)  # (B, spectrum_dim)
            beta = self.film_beta(aux_features)  # (B, spectrum_dim)
            
            # 步骤2: 逐元素调制spectrum
            # 公式: spectrum' = gamma ⊙ spectrum + beta
            # ⊙ 表示逐元素乘法(element-wise multiplication)
            modulated = gamma * spectrum_features + beta  # (B, spectrum_dim)
            
            # 步骤3: 调制后的特征输出预测
            output = self.output_head(modulated)

        elif self.fusion_method == 'gated_residual':
            fused_input = torch.cat([spectrum_features, aux_features], dim=-1)
            residual = self.residual_net(fused_input)
            gate = self.gate_net(fused_input)
            scale = self._bounded_residual_scale()
            correction_amplifier = self._correction_amplifier(spectrum_features)
            correction = scale * gate * residual * correction_amplifier
            self.last_correction_loss = torch.mean(correction.pow(2))
            self._record_gated_residual_diagnostics(
                spectrum_features, residual, gate, correction, scale, self.residual_scale, correction_amplifier
            )
            corrected = spectrum_features + correction
            output = self.output_head(corrected)
        
        else:
            raise NotImplementedError(f"Late fusion method '{self.fusion_method}' not implemented")
        
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output
    
    def fuse_features(
        self,
        spectrum_features: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        只进行特征融合，不经过最终输出层
        用于AR Wrapper等后续模块
        
        Args:
            spectrum_features: (batch_size, spectrum_dim)
            aux_features: (batch_size, aux_dim)
        Returns:
            fused: (batch_size, output_dim)
        """
        if self.fusion_method == 'concat':
            self.last_correction_loss = None
            fused = torch.cat([spectrum_features, aux_features], dim=-1)
        
        elif self.fusion_method == 'cross_attention':
            self.last_correction_loss = None
            batch_size = spectrum_features.size(0)
            
            # Spectrum关注Aux
            q_spec = self.spectrum_to_query(spectrum_features).unsqueeze(1)
            k_aux = self.aux_to_key(aux_features).unsqueeze(1)
            v_aux = self.aux_to_value(aux_features).unsqueeze(1)
            
            attn_weights_spec = torch.matmul(q_spec, k_aux.transpose(-2, -1)) * self.scale
            attn_weights_spec = torch.softmax(attn_weights_spec, dim=-1)
            attn_weights_spec = self.dropout_attn(attn_weights_spec)
            spectrum_attended = torch.matmul(attn_weights_spec, v_aux).squeeze(1)
            
            # Aux关注Spectrum
            q_aux = self.aux_to_query(aux_features).unsqueeze(1)
            k_spec = self.spectrum_to_key(spectrum_features).unsqueeze(1)
            v_spec = self.spectrum_to_value(spectrum_features).unsqueeze(1)
            
            attn_weights_aux = torch.matmul(q_aux, k_spec.transpose(-2, -1)) * self.scale
            attn_weights_aux = torch.softmax(attn_weights_aux, dim=-1)
            attn_weights_aux = self.dropout_attn(attn_weights_aux)
            aux_attended = torch.matmul(attn_weights_aux, v_spec).squeeze(1)
            
            fused = torch.cat([spectrum_attended, aux_attended], dim=-1)
        
        elif self.fusion_method == 'film':
            self.last_correction_loss = None
            gamma = self.film_gamma(aux_features)
            beta = self.film_beta(aux_features)
            fused = gamma * spectrum_features + beta

        elif self.fusion_method == 'gated_residual':
            fused_input = torch.cat([spectrum_features, aux_features], dim=-1)
            residual = self.residual_net(fused_input)
            gate = self.gate_net(fused_input)
            scale = self._bounded_residual_scale()
            correction_amplifier = self._correction_amplifier(spectrum_features)
            correction = scale * gate * residual * correction_amplifier
            self.last_correction_loss = torch.mean(correction.pow(2))
            self._record_gated_residual_diagnostics(
                spectrum_features, residual, gate, correction, scale, self.residual_scale, correction_amplifier
            )
            fused = spectrum_features + correction
        
        else:
            raise NotImplementedError(f"Late fusion method '{self.fusion_method}' not implemented")
        
        return fused


class CrossAttentionFiLMLateFusionHead(nn.Module):
    """
    混合融合头：Cross-Attention + FiLM

    设计目标：
    1) 先用“非温湿度”等动态辅助信号与主模态做 cross attention；
    2) 再用“温湿度等慢变环境信号”对合并后的特征进行 FiLM 调制。
    """
    def __init__(
        self,
        spectrum_dim: int,
        aux_cross_dim: int,
        aux_film_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        task_type: str = 'regression'
    ):
        super(CrossAttentionFiLMLateFusionHead, self).__init__()

        self.spectrum_dim = spectrum_dim
        self.aux_cross_dim = aux_cross_dim
        self.aux_film_dim = aux_film_dim
        self.hidden_size = hidden_size
        self.task_type = task_type

        # Stage-1: cross attention (spectrum <-> aux_cross)
        self.spectrum_to_query = nn.Linear(spectrum_dim, hidden_size)
        self.aux_cross_to_key = nn.Linear(aux_cross_dim, hidden_size)
        self.aux_cross_to_value = nn.Linear(aux_cross_dim, hidden_size)

        self.aux_cross_to_query = nn.Linear(aux_cross_dim, hidden_size)
        self.spectrum_to_key = nn.Linear(spectrum_dim, hidden_size)
        self.spectrum_to_value = nn.Linear(spectrum_dim, hidden_size)

        self.scale = hidden_size ** -0.5
        self.dropout_attn = nn.Dropout(dropout)

        # Cross-attention merged feature dim
        self.output_dim = hidden_size * 2

        # Stage-2: FiLM modulation on merged feature
        self.film_gamma = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.output_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.output_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(self.output_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _cross_attention_fuse(
        self,
        spectrum_features: torch.Tensor,
        aux_cross_features: torch.Tensor
    ) -> torch.Tensor:
        q_spec = self.spectrum_to_query(spectrum_features).unsqueeze(1)
        k_aux = self.aux_cross_to_key(aux_cross_features).unsqueeze(1)
        v_aux = self.aux_cross_to_value(aux_cross_features).unsqueeze(1)

        attn_weights_spec = torch.matmul(q_spec, k_aux.transpose(-2, -1)) * self.scale
        attn_weights_spec = torch.softmax(attn_weights_spec, dim=-1)
        attn_weights_spec = self.dropout_attn(attn_weights_spec)
        spectrum_attended = torch.matmul(attn_weights_spec, v_aux).squeeze(1)

        q_aux = self.aux_cross_to_query(aux_cross_features).unsqueeze(1)
        k_spec = self.spectrum_to_key(spectrum_features).unsqueeze(1)
        v_spec = self.spectrum_to_value(spectrum_features).unsqueeze(1)

        attn_weights_aux = torch.matmul(q_aux, k_spec.transpose(-2, -1)) * self.scale
        attn_weights_aux = torch.softmax(attn_weights_aux, dim=-1)
        attn_weights_aux = self.dropout_attn(attn_weights_aux)
        aux_attended = torch.matmul(attn_weights_aux, v_spec).squeeze(1)

        return torch.cat([spectrum_attended, aux_attended], dim=-1)

    def fuse_features(
        self,
        spectrum_features: torch.Tensor,
        aux_cross_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        # Stage-1: cross attention
        cross_fused = self._cross_attention_fuse(spectrum_features, aux_cross_features)

        # Stage-2: FiLM modulation by temperature/humidity-like signals
        gamma = self.film_gamma(aux_film_features)
        beta = self.film_beta(aux_film_features)
        return gamma * cross_fused + beta

    def forward(
        self,
        spectrum_features: torch.Tensor,
        aux_cross_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        fused = self.fuse_features(spectrum_features, aux_cross_features, aux_film_features)
        output = self.output_head(fused)
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output


class AuxCrossOnlyLateFusionHead(nn.Module):
    """
    仅使用aux-cross分支进行预测的融合头。

    说明：
    - 输入是“去除温湿度等FiLM分支后”的aux_cross特征；
    - 不使用spectrum主模态，主模态不参与预测结果。
    """
    def __init__(
        self,
        aux_cross_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        task_type: str = 'regression'
    ):
        super(AuxCrossOnlyLateFusionHead, self).__init__()

        self.aux_cross_dim = aux_cross_dim
        self.output_dim = aux_cross_dim
        self.task_type = task_type

        self.output_head = nn.Sequential(
            nn.Linear(self.output_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse_features(self, aux_cross_features: torch.Tensor) -> torch.Tensor:
        return aux_cross_features

    def forward(self, aux_cross_features: torch.Tensor) -> torch.Tensor:
        fused = self.fuse_features(aux_cross_features)
        output = self.output_head(fused)
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output


class FiLMAuxOnlyLateFusionHead(nn.Module):
    """
    仅使用aux-film分支进行FiLM调制的融合头。

    说明：
    - 输入是温湿度等慢变环境信号对应的aux_film特征；
    - 不使用aux的其他传感器特征，不进行cross-attention。
    """
    def __init__(
        self,
        spectrum_dim: int,
        aux_film_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        task_type: str = 'regression'
    ):
        super(FiLMAuxOnlyLateFusionHead, self).__init__()

        self.spectrum_dim = spectrum_dim
        self.aux_film_dim = aux_film_dim
        self.output_dim = spectrum_dim
        self.task_type = task_type

        self.film_gamma = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, spectrum_dim)
        )

        self.film_beta = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, spectrum_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse_features(
        self,
        spectrum_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.film_gamma(aux_film_features)
        beta = self.film_beta(aux_film_features)
        return gamma * spectrum_features + beta

    def forward(
        self,
        spectrum_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        fused = self.fuse_features(spectrum_features, aux_film_features)
        output = self.output_head(fused)
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output


class PPGICMGatedResidualEnvFiLMHead(nn.Module):
    """
    两阶段融合头：
    1) PPG/ICM 等动态信号生成门控残差，修正 spectrum 特征；
    2) BME680/T117 等环境温湿度信号对修正后的特征做 FiLM 调制。
    """
    def __init__(
        self,
        spectrum_dim: int,
        aux_gate_dim: int,
        aux_film_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        task_type: str = 'regression',
        residual_scale_mode: str = 'sigmoid',
        residual_scale_min: float = 0.02,
        correction_scale_mode: str = 'none'
    ):
        super(PPGICMGatedResidualEnvFiLMHead, self).__init__()

        self.spectrum_dim = spectrum_dim
        self.aux_gate_dim = aux_gate_dim
        self.aux_film_dim = aux_film_dim
        self.output_dim = spectrum_dim
        self.task_type = task_type
        self.residual_scale_mode = residual_scale_mode
        self.residual_scale_min = float(residual_scale_min)
        self.correction_scale_mode = correction_scale_mode
        self.collect_diagnostics = False
        self._diagnostic_batches = []
        self.last_correction_loss = None

        gated_input_dim = spectrum_dim + aux_gate_dim
        self.residual_net = nn.Sequential(
            nn.Linear(gated_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, spectrum_dim),
            nn.Tanh()
        )
        self.gate_net = nn.Sequential(
            nn.Linear(gated_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, spectrum_dim),
            nn.Sigmoid()
        )
        if self.residual_scale_mode == 'clamp':
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_scale = nn.Parameter(
                _residual_scale_raw_from_effective(0.1, self.residual_scale_min)
            )

        self.film_gamma = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, spectrum_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(aux_film_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, spectrum_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_diagnostics_enabled(self, enabled: bool):
        self.collect_diagnostics = bool(enabled)

    def reset_diagnostics(self):
        self._diagnostic_batches = []

    def get_diagnostics(self):
        if not self._diagnostic_batches:
            return {}
        keys = self._diagnostic_batches[0].keys()
        return {
            key: float(torch.tensor([batch[key] for batch in self._diagnostic_batches]).mean().item())
            for key in keys
        }

    def get_correction_loss(self):
        return self.last_correction_loss

    def _bounded_residual_scale(self):
        if self.residual_scale_mode == 'clamp':
            return torch.clamp(self.residual_scale, 0.0, 1.0)
        min_scale = min(max(self.residual_scale_min, 0.0), 0.99)
        return min_scale + (1.0 - min_scale) * torch.sigmoid(self.residual_scale)

    def _correction_amplifier(self, spectrum_features: torch.Tensor) -> torch.Tensor:
        if self.correction_scale_mode == 'spectrum_rms':
            return torch.sqrt(torch.mean(spectrum_features.detach().pow(2), dim=-1, keepdim=True) + 1e-8)
        if self.correction_scale_mode == 'multiplicative':
            return spectrum_features
        return torch.ones(
            spectrum_features.shape[0],
            1,
            dtype=spectrum_features.dtype,
            device=spectrum_features.device
        )

    def _record_gated_residual_diagnostics(
        self,
        spectrum_features: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
        correction: torch.Tensor,
        scale: torch.Tensor,
        scale_raw: Optional[torch.Tensor] = None,
        correction_amplifier: Optional[torch.Tensor] = None
    ):
        if not self.collect_diagnostics:
            return
        with torch.no_grad():
            eps = 1e-8
            gate_flat = gate.detach().reshape(-1)
            correction_norm = correction.detach().norm(dim=-1)
            spectrum_norm = spectrum_features.detach().norm(dim=-1)
            residual_norm = residual.detach().norm(dim=-1)
            correction_abs_mean = correction.detach().abs().mean()
            spectrum_abs_mean = spectrum_features.detach().abs().mean()
            residual_abs_mean = residual.detach().abs().mean()
            amp_mean = correction_amplifier.detach().abs().mean() if correction_amplifier is not None else torch.tensor(1.0)
            self._diagnostic_batches.append({
                'residual_scale': float(scale.detach().cpu().item()),
                'residual_scale_raw': float(scale_raw.detach().cpu().item()) if scale_raw is not None else 0.0,
                'residual_scale_min': float(self.residual_scale_min if self.residual_scale_mode != 'clamp' else 0.0),
                'correction_amplifier_mean': float(amp_mean.cpu().item()),
                'gate_mean': float(gate_flat.mean().cpu().item()),
                'gate_std': float(gate_flat.std(unbiased=False).cpu().item()),
                'gate_p25': float(torch.quantile(gate_flat, 0.25).cpu().item()),
                'gate_p50': float(torch.quantile(gate_flat, 0.50).cpu().item()),
                'gate_p75': float(torch.quantile(gate_flat, 0.75).cpu().item()),
                'residual_norm': float(residual_norm.mean().cpu().item()),
                'correction_norm': float(correction_norm.mean().cpu().item()),
                'spectrum_norm': float(spectrum_norm.mean().cpu().item()),
                'correction_to_spectrum_ratio': float((correction_norm / (spectrum_norm + eps)).mean().cpu().item()),
                'residual_abs_mean': float(residual_abs_mean.cpu().item()),
                'correction_abs_mean': float(correction_abs_mean.cpu().item()),
                'spectrum_abs_mean': float(spectrum_abs_mean.cpu().item()),
                'correction_abs_to_spectrum_abs': float((correction_abs_mean / (spectrum_abs_mean + eps)).cpu().item()),
            })

    def fuse_features(
        self,
        spectrum_features: torch.Tensor,
        aux_gate_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        gated_input = torch.cat([spectrum_features, aux_gate_features], dim=-1)
        residual = self.residual_net(gated_input)
        gate = self.gate_net(gated_input)
        scale = self._bounded_residual_scale()
        correction_amplifier = self._correction_amplifier(spectrum_features)
        correction = scale * gate * residual * correction_amplifier
        self.last_correction_loss = torch.mean(correction.pow(2))
        self._record_gated_residual_diagnostics(
            spectrum_features, residual, gate, correction, scale, self.residual_scale, correction_amplifier
        )
        corrected = spectrum_features + correction

        gamma = self.film_gamma(aux_film_features)
        beta = self.film_beta(aux_film_features)
        return gamma * corrected + beta

    def forward(
        self,
        spectrum_features: torch.Tensor,
        aux_gate_features: torch.Tensor,
        aux_film_features: torch.Tensor
    ) -> torch.Tensor:
        fused = self.fuse_features(spectrum_features, aux_gate_features, aux_film_features)
        output = self.output_head(fused)
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output


class SpectrumOnlyLateFusionHead(nn.Module):
    """
    Spectrum-only late head baseline.

    只使用base_model.extract_features得到的spectrum特征，再接与gated_residual
    相同规格的MLP预测头。它不编码aux、不生成gate/residual、不做correction，
    用于验证旧gated_residual收益是否主要来自late fusion output_head。
    """
    def __init__(
        self,
        spectrum_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        task_type: str = 'regression'
    ):
        super(SpectrumOnlyLateFusionHead, self).__init__()

        self.spectrum_dim = spectrum_dim
        self.output_dim = spectrum_dim
        self.task_type = task_type

        self.output_head = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse_features(self, spectrum_features: torch.Tensor) -> torch.Tensor:
        return spectrum_features

    def get_diagnostics(self):
        return {}

    def get_correction_loss(self):
        return None

    def forward(self, spectrum_features: torch.Tensor) -> torch.Tensor:
        output = self.output_head(spectrum_features)
        if self.task_type == 'regression':
            return output.squeeze(-1)
        return output


class LateFusionWrapper(nn.Module):
    """
    Late Fusion 包装器
    将任意基础模型包装为支持late fusion的模型
    
    工作流程：
    1. 基础模型处理spectrum特征 -> spectrum_encoded
    2. AuxEncoder处理aux特征 -> aux_encoded  
    3. FusionHead融合两者 -> prediction
    
    关键设计：
    - 基础模型需要暴露特征提取接口（不直接输出预测）
    - 包装器接管最终的预测输出
    """
    def __init__(
        self,
        base_model: nn.Module,
        spectrum_feature_dim: int,
        aux_input_dim: int,
        aux_encoded_dim: int = 32,
        hidden_size: int = 256,
        fusion_method: str = 'concat',
        film_aux_indices: Optional[List[int]] = None,
        dropout: float = 0.2,
        task_type: str = 'regression',
        residual_scale_mode: str = 'sigmoid',
        residual_scale_min: float = 0.02,
        correction_scale_mode: str = 'none'
    ):
        super(LateFusionWrapper, self).__init__()
        
        self.base_model = base_model
        self.spectrum_feature_dim = spectrum_feature_dim
        self.aux_input_dim = aux_input_dim
        self.aux_encoded_dim = aux_encoded_dim
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method
        self.residual_scale_mode = residual_scale_mode
        self.residual_scale_min = float(residual_scale_min)
        self.correction_scale_mode = correction_scale_mode
        self.film_aux_indices = None
        self.cross_aux_indices = None
        self.gate_aux_indices = None
        self.ppg_icm_aux_indices = None
        
        if fusion_method == 'spectrum_only_late_head':
            self.fusion_head = SpectrumOnlyLateFusionHead(
                spectrum_dim=spectrum_feature_dim,
                hidden_size=hidden_size,
                dropout=dropout,
                task_type=task_type
            )
        elif fusion_method in ['cross_attention_film', 'aux_cross_only', 'film_aux_only', 'ppg_icm_gated_residual_film', 'ppg_icm_gated_residual_only', 'ppg_icm_engineered_only', 'ppg_icm_segment_raw_only']:
            # 默认按data_loader.interpolate_aux_to_spectrum的真实顺序选FiLM通道：
            # [BME(0,1,2), PPG(...), T117(...), ICM(...)]
            # BME固定为前三维；T117索引受PPG格式影响：
            # - PPG raw(1维) -> t117约为索引4
            # - PPG red/ir(2维) -> t117约为索引5
            default_film_candidates = [i for i in [0, 1, 2] if i < aux_input_dim]
            if aux_input_dim >= 5:
                if fusion_method == 'ppg_icm_gated_residual_film':
                    inferred_t117_idx = self._infer_t117_index(aux_input_dim)
                else:
                    inferred_t117_idx = 4 if (aux_input_dim % 2 == 1) else 5
                if inferred_t117_idx < aux_input_dim:
                    default_film_candidates.append(inferred_t117_idx)

            requested = film_aux_indices if film_aux_indices is not None else default_film_candidates

            valid_film = sorted(set(int(i) for i in requested if 0 <= int(i) < aux_input_dim))
            if len(valid_film) == 0:
                valid_film = [0] if aux_input_dim > 0 else []

            cross_aux_indices = [i for i in range(aux_input_dim) if i not in valid_film]

            # 需要cross分支的方法：保证cross/film两路都非空
            if fusion_method in ['cross_attention_film', 'aux_cross_only']:
                if len(cross_aux_indices) == 0 and len(valid_film) > 1:
                    moved = valid_film.pop()
                    cross_aux_indices = [moved]
                elif len(cross_aux_indices) == 0 and len(valid_film) == 1 and aux_input_dim > 1:
                    cross_aux_indices = [i for i in range(aux_input_dim) if i != valid_film[0]]

                if len(valid_film) == 0 and len(cross_aux_indices) > 0:
                    valid_film = [cross_aux_indices[0]]
                    cross_aux_indices = cross_aux_indices[1:]

                # 最后兜底：若aux_input_dim=1，cross/film都使用同一通道
                if len(cross_aux_indices) == 0 and len(valid_film) == 1:
                    cross_aux_indices = [valid_film[0]]

            self.film_aux_indices = valid_film
            self.cross_aux_indices = cross_aux_indices

            if fusion_method == 'ppg_icm_gated_residual_film':
                gate_aux_indices = [i for i in range(aux_input_dim) if i not in valid_film]
                if len(gate_aux_indices) == 0 and len(valid_film) > 1:
                    gate_aux_indices = [valid_film.pop()]
                elif len(gate_aux_indices) == 0 and len(valid_film) == 1:
                    gate_aux_indices = [valid_film[0]]

                self.gate_aux_indices = gate_aux_indices
                self.aux_encoder_gate = AuxFeatureEncoder(
                    input_dim=len(self.gate_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )
                self.aux_encoder_film = AuxFeatureEncoder(
                    input_dim=len(self.film_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )
                self.fusion_head = PPGICMGatedResidualEnvFiLMHead(
                    spectrum_dim=spectrum_feature_dim,
                    aux_gate_dim=aux_encoded_dim,
                    aux_film_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type,
                    residual_scale_mode=residual_scale_mode,
                    residual_scale_min=residual_scale_min,
                    correction_scale_mode=correction_scale_mode
                )
            elif fusion_method == 'ppg_icm_gated_residual_only':
                self.ppg_icm_aux_indices = self._infer_ppg_icm_engineered_indices(aux_input_dim)
                self.aux_encoder_ppg_icm = AuxFeatureEncoder(
                    input_dim=len(self.ppg_icm_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )
                self.fusion_head = LateFusionHead(
                    spectrum_dim=spectrum_feature_dim,
                    aux_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    fusion_method='gated_residual',
                    dropout=dropout,
                    task_type=task_type,
                    residual_scale_mode=residual_scale_mode,
                    residual_scale_min=residual_scale_min,
                    correction_scale_mode=correction_scale_mode
                )
            elif fusion_method == 'ppg_icm_engineered_only':
                self.ppg_icm_aux_indices = self._infer_ppg_icm_engineered_indices(aux_input_dim)
                self.aux_encoder_ppg_icm = AuxFeatureEncoder(
                    input_dim=len(self.ppg_icm_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )
                self.fusion_head = AuxCrossOnlyLateFusionHead(
                    aux_cross_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type
                )
            elif fusion_method == 'ppg_icm_segment_raw_only':
                self.aux_encoder_ppg_icm = AuxFeatureEncoder(
                    input_dim=aux_input_dim,
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )
                self.fusion_head = AuxCrossOnlyLateFusionHead(
                    aux_cross_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type
                )
            elif fusion_method in ['cross_attention_film', 'aux_cross_only']:
                self.aux_encoder_cross = AuxFeatureEncoder(
                    input_dim=len(self.cross_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )

            if fusion_method == 'cross_attention_film':
                self.aux_encoder_film = AuxFeatureEncoder(
                    input_dim=len(self.film_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )

                self.fusion_head = CrossAttentionFiLMLateFusionHead(
                    spectrum_dim=spectrum_feature_dim,
                    aux_cross_dim=aux_encoded_dim,
                    aux_film_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type
                )
            elif fusion_method == 'aux_cross_only':
                self.fusion_head = AuxCrossOnlyLateFusionHead(
                    aux_cross_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type
                )
            elif fusion_method == 'film_aux_only':
                self.aux_encoder_film = AuxFeatureEncoder(
                    input_dim=len(self.film_aux_indices),
                    output_dim=aux_encoded_dim,
                    dropout=dropout
                )

                self.fusion_head = FiLMAuxOnlyLateFusionHead(
                    spectrum_dim=spectrum_feature_dim,
                    aux_film_dim=aux_encoded_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    task_type=task_type
                )
        else:
            # Aux特征编码器
            self.aux_encoder = AuxFeatureEncoder(
                input_dim=aux_input_dim,
                output_dim=aux_encoded_dim,
                dropout=dropout
            )

            # 融合头
            self.fusion_head = LateFusionHead(
                spectrum_dim=spectrum_feature_dim,
                aux_dim=aux_encoded_dim,
                hidden_size=hidden_size,
                fusion_method=fusion_method,
                dropout=dropout,
                task_type=task_type,
                residual_scale_mode=residual_scale_mode,
                residual_scale_min=residual_scale_min,
                correction_scale_mode=correction_scale_mode
            )

    def _infer_t117_index(self, aux_input_dim: int) -> int:
        """
        依据当前数据处理的aux拼接顺序推断T117索引。
        常见顺序为 BME(3) -> PPG raw/engineered -> T117(1) -> ICM。
        """
        if aux_input_dim >= 18:
            # engineered: BME3 + PPG engineered9 + T117 + ICM5 = 18
            # both常见为 BME3 + PPG raw1 + PPG engineered9 + T117 + ICM...
            return 13 if aux_input_dim >= 20 else 12
        return 4 if (aux_input_dim % 2 == 1) else 5

    def _infer_ppg_icm_engineered_indices(self, aux_input_dim: int) -> List[int]:
        """
        选择特征加工后的PPG与ICM维度，排除BME680与T117。

        aux_feature_mode='engineered' 时当前顺序为：
        BME(0,1,2) + PPG engineered(3..11) + T117(12) + ICM engineered(13..17)。
        对其他维度做保守兜底：去掉BME前三维和推断出的T117，其余作为PPG/ICM分支。
        """
        if aux_input_dim == 18:
            return list(range(3, 12)) + list(range(13, 18))

        t117_idx = self._infer_t117_index(aux_input_dim) if aux_input_dim > 3 else None
        indices = [
            i for i in range(aux_input_dim)
            if i >= 3 and (t117_idx is None or i != t117_idx)
        ]
        if len(indices) == 0 and aux_input_dim > 0:
            indices = list(range(aux_input_dim))
        return indices

    def _split_aux_branches(self, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将原始aux按索引分成cross-attention分支与FiLM分支。"""
        if aux.dim() == 3:
            aux_cross = aux[:, :, self.cross_aux_indices]
            aux_film = aux[:, :, self.film_aux_indices]
        else:
            aux_cross = aux[:, self.cross_aux_indices]
            aux_film = aux[:, self.film_aux_indices]
        return aux_cross, aux_film

    def _select_ppg_icm_aux(self, aux: torch.Tensor) -> torch.Tensor:
        """选择用于aux-only预测的PPG/ICM engineered特征。"""
        if aux.dim() == 3:
            return aux[:, :, self.ppg_icm_aux_indices]
        return aux[:, self.ppg_icm_aux_indices]

    def _split_gate_film_branches(self, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将原始aux分成PPG/ICM门控分支与BME680/T117 FiLM分支。"""
        if aux.dim() == 3:
            aux_gate = aux[:, :, self.gate_aux_indices]
            aux_film = aux[:, :, self.film_aux_indices]
        else:
            aux_gate = aux[:, self.gate_aux_indices]
            aux_film = aux[:, self.film_aux_indices]
        return aux_gate, aux_film
    
    def get_feature_dim(self):
        """
        返回融合后的特征维度
        用于后续模块（如AR Wrapper）获取输入维度
        """
        # 融合后的特征维度等于fusion_head的输入维度
        return self.fusion_head.output_dim

    def set_fusion_diagnostics_enabled(self, enabled: bool):
        if hasattr(self.fusion_head, 'set_diagnostics_enabled'):
            self.fusion_head.set_diagnostics_enabled(enabled)

    def reset_fusion_diagnostics(self):
        if hasattr(self.fusion_head, 'reset_diagnostics'):
            self.fusion_head.reset_diagnostics()

    def get_fusion_diagnostics(self):
        if hasattr(self.fusion_head, 'get_diagnostics'):
            return self.fusion_head.get_diagnostics()
        return {}

    def get_fusion_correction_loss(self):
        if hasattr(self.fusion_head, 'get_correction_loss'):
            return self.fusion_head.get_correction_loss()
        return None
    
    def extract_features(self, spectrum: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """
        提取融合后的特征（不经过最终输出层）
        用于AR Wrapper等后续模块
        
        Args:
            spectrum: (batch_size, spectrum_dim) or (batch_size, window_size, spectrum_dim)
            aux: (batch_size, aux_dim) or (batch_size, window_size, aux_dim)
        
        Returns:
            fused_features: (batch_size, feature_dim)
        """
        # 1. 基础模型提取spectrum特征
        if hasattr(self.base_model, 'extract_features'):
            spectrum_features = self.base_model.extract_features(spectrum)
        else:
            # 兼容模式
            spectrum_features = self.base_model(spectrum)
        
        if self.fusion_method == 'spectrum_only_late_head':
            fused = self.fusion_head.fuse_features(spectrum_features)
        elif aux is not None:
            if self.fusion_method == 'cross_attention_film':
                aux_cross, aux_film = self._split_aux_branches(aux)
                if aux.dim() == 3:
                    aux_cross_features = self.aux_encoder_cross(aux_cross).mean(dim=1)
                    aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
                else:
                    aux_cross_features = self.aux_encoder_cross(aux_cross)
                    aux_film_features = self.aux_encoder_film(aux_film)
                fused = self.fusion_head.fuse_features(
                    spectrum_features,
                    aux_cross_features,
                    aux_film_features
                )
            elif self.fusion_method == 'aux_cross_only':
                aux_cross, _ = self._split_aux_branches(aux)
                if aux.dim() == 3:
                    aux_cross_features = self.aux_encoder_cross(aux_cross).mean(dim=1)
                else:
                    aux_cross_features = self.aux_encoder_cross(aux_cross)
                fused = self.fusion_head.fuse_features(aux_cross_features)
            elif self.fusion_method == 'ppg_icm_engineered_only':
                aux_ppg_icm = self._select_ppg_icm_aux(aux)
                if aux.dim() == 3:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm).mean(dim=1)
                else:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm)
                fused = self.fusion_head.fuse_features(aux_ppg_icm_features)
            elif self.fusion_method == 'ppg_icm_gated_residual_only':
                aux_ppg_icm = self._select_ppg_icm_aux(aux)
                if aux.dim() == 3:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm).mean(dim=1)
                else:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm)
                fused = self.fusion_head.fuse_features(spectrum_features, aux_ppg_icm_features)
            elif self.fusion_method == 'ppg_icm_segment_raw_only':
                if aux.dim() == 3:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux).mean(dim=1)
                else:
                    aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux)
                fused = self.fusion_head.fuse_features(aux_ppg_icm_features)
            elif self.fusion_method == 'film_aux_only':
                _, aux_film = self._split_aux_branches(aux)
                if aux.dim() == 3:
                    aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
                else:
                    aux_film_features = self.aux_encoder_film(aux_film)
                fused = self.fusion_head.fuse_features(spectrum_features, aux_film_features)
            elif self.fusion_method == 'ppg_icm_gated_residual_film':
                aux_gate, aux_film = self._split_gate_film_branches(aux)
                if aux.dim() == 3:
                    aux_gate_features = self.aux_encoder_gate(aux_gate).mean(dim=1)
                    aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
                else:
                    aux_gate_features = self.aux_encoder_gate(aux_gate)
                    aux_film_features = self.aux_encoder_film(aux_film)
                fused = self.fusion_head.fuse_features(
                    spectrum_features,
                    aux_gate_features,
                    aux_film_features
                )
            else:
                if aux.dim() == 3:
                    # Window mode: (batch, window, aux_dim)
                    aux_encoded = self.aux_encoder(aux)
                    aux_features = aux_encoded.mean(dim=1)
                else:
                    # Instant mode: (batch, aux_dim)
                    aux_features = self.aux_encoder(aux)

                # 3. 融合特征（不经过输出层）
                fused = self.fusion_head.fuse_features(spectrum_features, aux_features)
        else:
            # 没有aux时，只返回spectrum特征
            fused = spectrum_features
        
        return fused

    def forward(
        self,
        spectrum: torch.Tensor,
        aux: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spectrum: (batch_size, spectrum_dim) for instant
                     (batch_size, window_size, spectrum_dim) for window
            aux: (batch_size, aux_dim) for instant
                (batch_size, window_size, aux_dim) for window
        Returns:
            prediction: (batch_size,)
        """
        # 1. 基础模型提取spectrum特征
        # 基础模型应该有 extract_features 方法
        if hasattr(self.base_model, 'extract_features'):
            spectrum_features = self.base_model.extract_features(spectrum)
        else:
            # 兼容模式：直接forward（假设输出是特征）
            spectrum_features = self.base_model(spectrum)
        
        # 2. 编码aux特征
        # 对于window模式，需要聚合aux特征
        if self.fusion_method == 'spectrum_only_late_head':
            output = self.fusion_head(spectrum_features)
        elif self.fusion_method == 'cross_attention_film':
            aux_cross, aux_film = self._split_aux_branches(aux)
            if aux.dim() == 3:
                aux_cross_features = self.aux_encoder_cross(aux_cross).mean(dim=1)
                aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
            else:
                aux_cross_features = self.aux_encoder_cross(aux_cross)
                aux_film_features = self.aux_encoder_film(aux_film)

            # 3. 混合融合并预测：cross-attention -> FiLM
            output = self.fusion_head(spectrum_features, aux_cross_features, aux_film_features)
        elif self.fusion_method == 'aux_cross_only':
            # 仅使用aux-cross分支进行预测，主模态spectrum不参与结果
            aux_cross, _ = self._split_aux_branches(aux)
            if aux.dim() == 3:
                aux_cross_features = self.aux_encoder_cross(aux_cross).mean(dim=1)
            else:
                aux_cross_features = self.aux_encoder_cross(aux_cross)
            output = self.fusion_head(aux_cross_features)
        elif self.fusion_method == 'ppg_icm_engineered_only':
            # 仅使用特征加工后的PPG与ICM分支预测，不使用spectrum/BME680/T117
            aux_ppg_icm = self._select_ppg_icm_aux(aux)
            if aux.dim() == 3:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm).mean(dim=1)
            else:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm)
            output = self.fusion_head(aux_ppg_icm_features)
        elif self.fusion_method == 'ppg_icm_gated_residual_only':
            # 仅使用PPG/ICM engineered特征生成门控残差来修正spectrum，不使用环境FiLM
            aux_ppg_icm = self._select_ppg_icm_aux(aux)
            if aux.dim() == 3:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm).mean(dim=1)
            else:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux_ppg_icm)
            output = self.fusion_head(spectrum_features, aux_ppg_icm_features)
        elif self.fusion_method == 'ppg_icm_segment_raw_only':
            # 仅使用PPG/ICM同一秒内原始序列槽位预测，不使用spectrum/BME680/T117
            if aux.dim() == 3:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux).mean(dim=1)
            else:
                aux_ppg_icm_features = self.aux_encoder_ppg_icm(aux)
            output = self.fusion_head(aux_ppg_icm_features)
        elif self.fusion_method == 'film_aux_only':
            # 仅使用温湿度等aux-film分支进行FiLM调制，不使用其他aux传感器
            _, aux_film = self._split_aux_branches(aux)
            if aux.dim() == 3:
                aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
            else:
                aux_film_features = self.aux_encoder_film(aux_film)
            output = self.fusion_head(spectrum_features, aux_film_features)
        elif self.fusion_method == 'ppg_icm_gated_residual_film':
            # 先用PPG/ICM动态分支做门控残差，再用BME680/T117环境分支做FiLM调制
            aux_gate, aux_film = self._split_gate_film_branches(aux)
            if aux.dim() == 3:
                aux_gate_features = self.aux_encoder_gate(aux_gate).mean(dim=1)
                aux_film_features = self.aux_encoder_film(aux_film).mean(dim=1)
            else:
                aux_gate_features = self.aux_encoder_gate(aux_gate)
                aux_film_features = self.aux_encoder_film(aux_film)
            output = self.fusion_head(spectrum_features, aux_gate_features, aux_film_features)
        else:
            if aux.dim() == 3:
                # Window mode: (batch, window, aux_dim)
                # 使用均值聚合（可以改为更复杂的聚合）
                aux_encoded = self.aux_encoder(aux)  # (batch, window, aux_encoded_dim)
                aux_features = aux_encoded.mean(dim=1)  # (batch, aux_encoded_dim)
            else:
                # Instant mode: (batch, aux_dim)
                aux_features = self.aux_encoder(aux)  # (batch, aux_encoded_dim)

            # 3. 融合并预测
            output = self.fusion_head(spectrum_features, aux_features)
        
        return output


def create_late_fusion_model(
    base_model: nn.Module,
    spectrum_feature_dim: int,
    aux_input_dim: int,
    config
) -> LateFusionWrapper:
    """
    工厂函数：创建late fusion模型
    
    Args:
        base_model: 基础模型（需要支持extract_features方法）
        spectrum_feature_dim: 基础模型输出的spectrum特征维度
        aux_input_dim: aux传感器输入维度
        config: 配置对象
    
    Returns:
        LateFusionWrapper实例
    """
    # 从config获取参数
    aux_encoded_dim = getattr(config.data, 'aux_encoded_dim', 32)
    fusion_method = config.data.fusion_method_late
    film_aux_indices = getattr(config.data, 'late_film_aux_indices', None)
    residual_scale_mode = getattr(config.training, 'gated_residual_scale_mode', 'sigmoid')
    residual_scale_min = getattr(config.training, 'gated_residual_min_scale', 0.02)
    correction_scale_mode = getattr(config.training, 'gated_residual_correction_scale', 'none')
    hidden_size = config.model.hidden_size
    dropout = config.model.dropout
    task_type = config.model.task_type
    
    return LateFusionWrapper(
        base_model=base_model,
        spectrum_feature_dim=spectrum_feature_dim,
        aux_input_dim=aux_input_dim,
        aux_encoded_dim=aux_encoded_dim,
        hidden_size=hidden_size,
        fusion_method=fusion_method,
        film_aux_indices=film_aux_indices,
        dropout=dropout,
        task_type=task_type,
        residual_scale_mode=residual_scale_mode,
        residual_scale_min=residual_scale_min,
        correction_scale_mode=correction_scale_mode
    )

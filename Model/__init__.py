"""
Model 模块
包含所有深度学习模型定义
"""

from .models import (
    GlucoseMLP,
    GlucoseCNN,
    GlucoseTransformer,
    RFImageCNN,
    create_model,
    get_model,
)

from .late_fusion import (
    AuxFeatureEncoder,
    LateFusionHead,
    CrossAttentionFiLMLateFusionHead,
    AuxCrossOnlyLateFusionHead,
    LateFusionWrapper,
    create_late_fusion_model,
)

from .ar_loss import (
    AutoRegressiveLoss,
    AdaptiveARLoss,
    create_ar_criterion,
)

__all__ = [
    'GlucoseMLP',
    'GlucoseCNN',
    'GlucoseTransformer',
    'RFImageCNN',
    'create_model',
    'get_model',
    # Late Fusion
    'AuxFeatureEncoder',
    'LateFusionHead',
    'CrossAttentionFiLMLateFusionHead',
    'AuxCrossOnlyLateFusionHead',
    'LateFusionWrapper',
    'create_late_fusion_model',
    # Auto-Regressive Loss
    'AutoRegressiveLoss',
    'AdaptiveARLoss',
    'create_ar_criterion',
]

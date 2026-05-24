"""
数据加载模块 - 从 Dataset 目录加载用户的频谱数据和血糖标签
支持跨平台（Windows/Linux）文件名大小写匹配

按照 Jupyter Notebook 的方法：
1. 从文件夹名称自动提取起始时间（如 11201828_Tao -> 11.20 18:28）
2. 加载所有 S*.BIN 文件并拼接
3. 使用 power_0 到 power_1000 作为特征（1001维）
4. 为每个频谱时间戳插值计算对应的血糖值
5. 可选：加载辅助传感器数据（BME680, PPG, T117, ICM）并融合到特征中

辅助传感器数据对齐策略：
==================
为确保aux sensor数据与spectrum数据在各种操作下完全对齐，采用以下策略：

1. **在数据加载阶段完成融合**：
   - Spectrum数据加载 → 得到spectrum_features和timestamps
   - (可选) 平滑处理 → spectrum_features保持相同timestamps
   - (可选) Down sampling → spectrum_features和timestamps同步下采样
   - Aux数据插值到最终的timestamps → aux_features与spectrum完全对齐
   - 融合 → features = [spectrum | aux]

2. **自动支持所有后续操作**：
   - Split划分：spectrum和aux作为统一的features一起split，自动对齐
   - Window创建：window内每个时间步的spectrum和aux都是对齐的
   - 归一化：对spectrum和aux分别归一化，但基于相同的样本集

3. **关键优势**：
   - ✓ 时间轴完全一致：aux插值到与spectrum相同的timestamps
   - ✓ Down sampling后对齐：aux在downsample后插值，适应新时间轴
   - ✓ Split模式通用：所有split模式（random, temporal, alternating等）自动对齐
   - ✓ Window模式对齐：窗口内每个时间步的spectrum和aux完全对应
   - ✓ 避免数据泄露：在单个实验内部融合，不跨实验共享信息
"""

import json
import struct
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np


def _aux_shuffle_group_for_feature(feature_name):
    name = str(feature_name).lower()
    if 'ppg' in name or 'heart' in name or name.startswith('hr') or 'pulse' in name:
        return 'ppg'
    if 'icm' in name or 'acc' in name or 'gyro' in name or 'gyr' in name or 'motion' in name:
        return 'icm'
    if 't117' in name:
        return 't117'
    if 'bme' in name or 'humid' in name or 'pressure' in name or 'gas' in name or 'temperature' in name or 'temp' in name:
        return 'bme'
    return 'other'


def shuffle_aux_features_for_truth_test(aux_features, feature_names=None, mode='sensor_groups', seed=42):
    """Shuffle aux only, preserving spectrum-label alignment for aux truth tests."""
    if aux_features is None:
        return aux_features

    shuffled = np.array(aux_features, copy=True)
    if shuffled.ndim != 2 or shuffled.shape[0] <= 1:
        return shuffled

    rng = np.random.default_rng(seed)
    n_samples, n_features = shuffled.shape

    if mode == 'rows':
        return shuffled[rng.permutation(n_samples)]

    if mode == 'features':
        for feature_idx in range(n_features):
            shuffled[:, feature_idx] = shuffled[rng.permutation(n_samples), feature_idx]
        return shuffled

    if mode == 'sensor_groups' and feature_names is not None and len(feature_names) == n_features:
        groups = {}
        for feature_idx, feature_name in enumerate(feature_names):
            groups.setdefault(_aux_shuffle_group_for_feature(feature_name), []).append(feature_idx)

        for group_indices in groups.values():
            perm = rng.permutation(n_samples)
            shuffled[:, group_indices] = shuffled[perm][:, group_indices]
        return shuffled

    for feature_idx in range(n_features):
        shuffled[:, feature_idx] = shuffled[rng.permutation(n_samples), feature_idx]
    return shuffled


import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler


# Auxiliary sensor file format constants
AUX_MAGIC = b"AUXDATA"
AUX_HEADER_STRUCT = struct.Struct('<8sII')  # magic[8], version, reserved

# Packed record layouts (little-endian)
REC_BME680 = struct.Struct('<d f f f')        # ts, tempC, pressure_hPa, humidity_pct
REC_PPG    = struct.Struct('<d i i h B B')    # ts, red, ir, heart_rate_bpm, spo2, reserved
REC_T117   = struct.Struct('<d f')            # ts, tempC
REC_ICM    = struct.Struct('<d f f f f f f')  # ts, acc_x,y,z, gyr_x,y,z

TAG_BME680 = 0x10
TAG_PPG    = 0x11
TAG_T117   = 0x12
TAG_ICM    = 0x13


def apply_data_fusion(spectrum_features: np.ndarray, 
                      aux_features: np.ndarray,
                      fusion_method: str = 'concat') -> np.ndarray:
    """
    应用数据融合策略，将光谱特征与辅助传感器特征融合
    
    针对血糖预测任务设计的融合方法：
    - Spectrum: 1001维光谱吸收特征，不同波段对血糖敏感性不同
    - Aux: ~10维环境/生理参数（温度、湿度、心率等），作为调制信号
    
    Args:
        spectrum_features: 光谱特征数组，shape=(n_samples, n_spectrum_features)
        aux_features: 辅助传感器特征数组，shape=(n_samples, n_aux_features)
        fusion_method: 融合方法，支持:
            - 'concat': 简单拼接，基线方法
            - 'film': Feature-wise Linear Modulation，全局调制
            - 'film_sensor_aware': 传感器感知FiLM，针对不同生理信号
            - 'attention_pool': 注意力加权池化，提取关键光谱信息
            - 'residual_film': 残差FiLM，保留原始spectrum避免过度调制
            - 'conditioned_rf': 条件化RF建模，aux仅作调制
            - 'attention_fusion': 交叉注意力融合，动态学习信任度
    
    Returns:
        fused_features: 融合后的特征数组
    """
    n_samples = spectrum_features.shape[0]
    n_spectrum = spectrum_features.shape[1]
    n_aux = aux_features.shape[1]
    
    if fusion_method == 'concat':
        # 方法1: 简单拼接 [spectrum | aux]
        # 适用场景: 基线方法，让模型自己学习特征关系
        fused = np.concatenate([spectrum_features, aux_features], axis=1)
        print(f"    ✓ Fusion: Concat - 输出维度: {fused.shape[1]}")
        return fused
    
    elif fusion_method == 'film':
        # 方法2: Feature-wise Linear Modulation
        # 用aux特征生成全局的缩放(gamma)和偏移(beta)参数来调制spectrum
        # 适用场景: 环境因素（温度、湿度）全局影响光谱响应
        
        # 计算aux的统计量作为调制参数
        aux_mean = np.mean(aux_features, axis=1, keepdims=True)  # (n_samples, 1)
        aux_std = np.std(aux_features, axis=1, keepdims=True) + 1e-8
        
        # 生成调制参数并广播到spectrum维度
        gamma = 1.0 + 0.1 * np.tile(aux_mean, (1, n_spectrum))  # 缩放因子
        beta = 0.1 * np.tile(aux_std, (1, n_spectrum))  # 偏移因子
        
        # FiLM调制
        modulated_spectrum = gamma * spectrum_features + beta
        
        # 拼接调制后的spectrum和原始aux
        fused = np.concatenate([modulated_spectrum, aux_features], axis=1)
        print(f"    ✓ Fusion: FiLM - 输出维度: {fused.shape[1]}")
        return fused
    
    elif fusion_method == 'attention_pool':
        # 方法5: 注意力加权池化 (Attention-based Pooling)
        # 用aux特征生成注意力权重，对spectrum进行加权池化，提取关键信息
        # 适用场景: 压缩spectrum维度同时保留重要信息
        
        # 检查输入数据是否有NaN
        if np.any(np.isnan(spectrum_features)):
            print(f"    ⚠ 警告: spectrum_features包含{np.sum(np.isnan(spectrum_features))}个NaN值")
            spectrum_features = np.nan_to_num(spectrum_features, nan=0.0)
        if np.any(np.isnan(aux_features)):
            print(f"    ⚠ 警告: aux_features包含{np.sum(np.isnan(aux_features))}个NaN值")
            aux_features = np.nan_to_num(aux_features, nan=0.0)
        
        # 用aux特征的每个维度生成一组注意力权重
        pooled_features = []
        for i in range(n_aux):
            # 用aux_i生成注意力权重（softmax归一化）
            aux_i = aux_features[:, i:i+1]
            attention_logits = spectrum_features * aux_i  # 加权
            
            # 数值稳定的softmax：减去最大值避免exp溢出
            attention_logits_stable = attention_logits - np.max(attention_logits, axis=1, keepdims=True)
            attention_weights = np.exp(attention_logits_stable)
            
            # 检查并处理无效值
            attention_sum = np.sum(attention_weights, axis=1, keepdims=True)
            attention_sum = np.where(attention_sum > 1e-10, attention_sum, 1.0)  # 避免除以0
            attention_weights = attention_weights / attention_sum
            
            # 检查NaN和Inf
            if np.any(np.isnan(attention_weights)) or np.any(np.isinf(attention_weights)):
                print(f"    ⚠ 警告: aux维度{i}的注意力权重包含NaN或Inf，使用均匀分布")
                attention_weights = np.nan_to_num(attention_weights, nan=1.0/n_spectrum, posinf=1.0/n_spectrum, neginf=0.0)
            
            # 加权池化
            pooled = np.sum(spectrum_features * attention_weights, axis=1, keepdims=True)
            
            # 检查池化结果
            if np.any(np.isnan(pooled)):
                print(f"    ⚠ 警告: aux维度{i}的池化结果包含NaN")
                pooled = np.nan_to_num(pooled, nan=0.0)
            
            pooled_features.append(pooled)
        
        # 拼接所有池化特征
        pooled_spectrum = np.concatenate(pooled_features, axis=1)  # (n_samples, n_aux)
        
        # 拼接原始spectrum、池化特征和aux
        fused = np.concatenate([spectrum_features, pooled_spectrum, aux_features], axis=1)
        
        # 最终检查
        if np.any(np.isnan(fused)):
            nan_count = np.sum(np.isnan(fused))
            print(f"    ⚠ 警告: 融合结果包含{nan_count}个NaN值，已替换为0")
            fused = np.nan_to_num(fused, nan=0.0)
        
        print(f"    ✓ Fusion: Attention-Pool - 输出维度: {fused.shape[1]} (池化到{n_aux}维)")
        print(f"      融合特征范围: [{fused.min():.3f}, {fused.max():.3f}]")
        return fused
    
    elif fusion_method == 'residual_film':
        # 方法6: 残差FiLM (Residual Feature-wise Linear Modulation)
        # 在FiLM的基础上添加残差连接，避免过度调制
        # 适用场景: 保守的调制策略，保留原始spectrum信息
        
        # 计算FiLM参数（使用更小的系数）
        aux_mean = np.mean(aux_features, axis=1, keepdims=True)
        aux_std = np.std(aux_features, axis=1, keepdims=True) + 1e-8
        
        # 生成调制参数（系数更小，避免过度调制）
        gamma = 1.0 + 0.05 * np.tile(aux_mean, (1, n_spectrum))
        beta = 0.05 * np.tile(aux_std, (1, n_spectrum))
        
        # FiLM调制
        modulated_spectrum = gamma * spectrum_features + beta
        
        # 残差连接：加权组合原始和调制后的spectrum
        residual_weight = 0.7  # 保留70%原始信息
        residual_spectrum = residual_weight * spectrum_features + (1 - residual_weight) * modulated_spectrum
        
        # 拼接residual spectrum和aux
        fused = np.concatenate([residual_spectrum, aux_features], axis=1)
        print(f"    ✓ Fusion: Residual-FiLM - 输出维度: {fused.shape[1]} (残差权重: {residual_weight})")
        return fused
    
    elif fusion_method == 'film_sensor_aware':
        # 方法6.5: 传感器感知FiLM (Sensor-Aware FiLM)
        # 根据不同传感器的物理特性，使用针对性的调制策略
        # 
        # 物理机制与调制策略：
        # - 温度(BME680+T117)：影响红外区域水分吸收峰 → 调制后半段光谱
        # - 湿度(BME680)：影响水分吸收强度 → 全局基线偏移
        # - 气压(BME680)：影响传感器物理环境 → 轻微全局调制
        # - PPG(红光+红外)：血流动力学变化 → 调制特定波段（绿光/红外区）
        # - 运动(ICM)：改变接触压力和光路 → 全局增益调制
        
        # 初始化调制参数
        gamma = np.ones((n_samples, n_spectrum))  # 缩放因子，默认1.0（不缩放）
        beta = np.zeros((n_samples, n_spectrum))   # 偏移因子，默认0.0（不偏移）
        
        # 特征顺序假设：[bme_temp, bme_pressure, bme_humidity, ppg_red, ppg_ir, 
        #                t117_temp, icm_features...]
        modulation_info = []
        
        # 1. 温度调制（BME680 + T117）
        # 物理原理：温度影响组织水分含量和红外吸收特性
        # 策略：主要调制后半段光谱（红外波段，索引500-1000）
        temp_features = []
        if n_aux >= 1:  # bme_temp
            temp_features.append(aux_features[:, 0:1])
        if n_aux >= 6:  # t117_temp
            temp_features.append(aux_features[:, 5:6])
        
        if temp_features:
            temp_avg = np.mean(np.concatenate(temp_features, axis=1), axis=1, keepdims=True)
            # 温度主要影响后500维（对应红外波段）
            temp_modulation = 0.08 * np.tanh(temp_avg)  # 限制在[-0.08, +0.08]
            gamma[:, 500:] *= (1.0 + temp_modulation)
            modulation_info.append(f"温度→红外区[500:] γ∈[{(1+temp_modulation).min():.3f},{(1+temp_modulation).max():.3f}]")
        
        # 2. 湿度调制（BME680）
        # 物理原理：湿度影响整体水分吸收强度
        # 策略：对全光谱添加偏移（基线漂移）
        if n_aux >= 3:  # bme_humidity
            humidity = aux_features[:, 2:3]
            humidity_offset = 0.05 * np.tanh(humidity)  # [-0.05, +0.05]
            beta += np.tile(humidity_offset, (1, n_spectrum))
            modulation_info.append(f"湿度→全谱基线 β∈[{humidity_offset.min():.3f},{humidity_offset.max():.3f}]")
        
        # 3. 气压调制（BME680）
        # 物理原理：气压影响传感器响应特性
        # 策略：全局小幅度缩放
        if n_aux >= 2:  # bme_pressure
            pressure = aux_features[:, 1:2]
            pressure_modulation = 0.02 * np.tanh(pressure)  # [-0.02, +0.02]
            gamma *= (1.0 + np.tile(pressure_modulation, (1, n_spectrum)))
            modulation_info.append(f"气压→全谱增益 γ×[{(1+pressure_modulation).min():.3f},{(1+pressure_modulation).max():.3f}]")
        
        # 4. PPG调制（心率、血氧）
        # 物理原理：血流动力学影响特定波长的光吸收
        # 策略：调制绿光区（前1/3）和红外区（后1/3）
        if n_aux >= 5:  # ppg_red, ppg_ir
            ppg_red = aux_features[:, 3:4]
            ppg_ir = aux_features[:, 4:5]
            
            # 绿光区域（前333维，对应可见光）受心率影响
            ppg_modulation_green = 0.06 * np.tanh(ppg_red)
            gamma[:, :333] *= (1.0 + ppg_modulation_green)
            
            # 红外区域（后334维）受血氧影响
            ppg_modulation_ir = 0.06 * np.tanh(ppg_ir)
            gamma[:, 667:] *= (1.0 + ppg_modulation_ir)
            
            modulation_info.append(f"PPG→绿光区[:333] γ×[{(1+ppg_modulation_green).min():.3f},{(1+ppg_modulation_green).max():.3f}]")
            modulation_info.append(f"PPG→红外区[667:] γ×[{(1+ppg_modulation_ir).min():.3f},{(1+ppg_modulation_ir).max():.3f}]")
        
        # 5. 运动调制（ICM）
        # 物理原理：运动改变接触压力和光路稳定性
        # 策略：基于运动强度调制全局增益（运动越大，信噪比越低）
        if n_aux >= 9:  # 有ICM数据（后6维或更多）
            icm_features = aux_features[:, -6:]
            # 计算运动强度（L2范数）
            motion_intensity = np.sqrt(np.sum(icm_features**2, axis=1, keepdims=True))
            # 运动越强，轻微降低增益（补偿信号质量下降）
            motion_modulation = -0.04 * np.tanh(motion_intensity)  # [-0.04, 0]
            gamma *= (1.0 + np.tile(motion_modulation, (1, n_spectrum)))
            modulation_info.append(f"运动→全谱抑制 γ×[{(1+motion_modulation).min():.3f},{(1+motion_modulation).max():.3f}]")
        
        # 应用FiLM调制
        modulated_spectrum = gamma * spectrum_features + beta
        
        # 拼接调制后的spectrum和原始aux
        fused = np.concatenate([modulated_spectrum, aux_features], axis=1)
        
        print(f"    ✓ Fusion: FiLM-SensorAware - 输出维度: {fused.shape[1]}")
        print(f"      总调制范围: γ∈[{gamma.min():.3f},{gamma.max():.3f}], β∈[{beta.min():.3f},{beta.max():.3f}]")
        for info in modulation_info:
            print(f"      • {info}")
        
        return fused
    
    elif fusion_method == 'conditioned_rf':
        # 方法7: 条件化RF建模 (Conditioned RF Modeling)
        # aux-sensor不直接参与预测，而是调制RF→glucose的mapping
        # 公式: z_RF = γ(aux) ⊙ z_RF + β(aux)
        # 直觉: 在不同温度/湿度/生理状态下，RF对glucose的sensitivity不一样
        # 适合 metasurface + bio-sensing 场景
        
        # 使用aux特征的非线性变换生成调制参数
        # 模拟 Aux → MLP → (γ, β) 的过程
        
        # 计算每个样本的aux特征统计量
        aux_mean = np.mean(aux_features, axis=1, keepdims=True)  # (n_samples, 1)
        aux_std = np.std(aux_features, axis=1, keepdims=True) + 1e-8
        aux_min = np.min(aux_features, axis=1, keepdims=True)
        aux_max = np.max(aux_features, axis=1, keepdims=True)
        
        # 生成调制参数γ和β (非线性映射)
        # γ: 控制RF特征的sensitivity/gain
        # β: 控制RF特征的baseline shift
        gamma = 1.0 + 0.2 * np.tanh(np.tile(aux_mean, (1, n_spectrum)))  # [0.8, 1.2]
        beta = 0.15 * np.tanh(np.tile(aux_std, (1, n_spectrum)))  # [-0.15, 0.15]
        
        # 可选: 添加更复杂的非线性映射（模拟MLP）
        # 这里使用简单的多项式组合
        aux_interaction = aux_mean * aux_std  # 特征交互
        gamma_adjustment = 0.05 * np.tanh(np.tile(aux_interaction, (1, n_spectrum)))
        gamma = gamma + gamma_adjustment
        
        # Conditioned RF Modeling: 只调制RF特征，不拼接aux
        conditioned_rf = gamma * spectrum_features + beta
        
        # 关键: 只输出conditioned RF，aux不直接参与后续预测
        # 这样aux的作用纯粹是调制RF→glucose的映射关系
        fused = conditioned_rf
        
        print(f"    ✓ Fusion: Conditioned-RF - 输出维度: {fused.shape[1]} (纯RF特征，aux仅调制)")
        print(f"      调制范围: γ ∈ [{gamma.min():.3f}, {gamma.max():.3f}], β ∈ [{beta.min():.3f}, {beta.max():.3f}]")
        return fused
    
    elif fusion_method == 'attention_fusion':
        # 方法8: 交叉注意力融合 (Cross-Attention Fusion)
        # 让模型自己学什么时候该信任aux，什么时候该信任RF
        # RF作为Query，Aux作为Key/Value
        # 问题：当前频谱形态下，哪些生理/环境变量是relevant的？
        # 对时间序列尤其有效
        
        # 简化版Cross-Attention (数据层面，不使用可训练参数)
        # 这里实现基于相似度的软注意力机制
        
        # 1. 计算RF特征的统计摘要作为Query
        rf_mean = np.mean(spectrum_features, axis=1, keepdims=True)  # (n_samples, 1)
        rf_std = np.std(spectrum_features, axis=1, keepdims=True) + 1e-8
        rf_max = np.max(spectrum_features, axis=1, keepdims=True)
        rf_min = np.min(spectrum_features, axis=1, keepdims=True)
        query = np.concatenate([rf_mean, rf_std, rf_max, rf_min], axis=1)  # (n_samples, 4)
        
        # 2. Aux特征作为Key和Value
        key = aux_features  # (n_samples, n_aux)
        value = aux_features
        
        # 3. 计算注意力权重：Query和Key的相似度
        # 使用余弦相似度的简化版本
        attention_scores = []
        for i in range(n_aux):
            # 计算query的每个维度与key的第i维的相关性
            score = np.sum(query * key[:, i:i+1], axis=1, keepdims=True)  # (n_samples, 1)
            attention_scores.append(score)
        
        attention_scores = np.concatenate(attention_scores, axis=1)  # (n_samples, n_aux)
        
        # Softmax归一化得到注意力权重
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_weights / (np.sum(attention_weights, axis=1, keepdims=True) + 1e-8)
        
        # 4. 使用注意力权重对aux进行加权
        attended_aux = attention_weights * aux_features  # (n_samples, n_aux)
        
        # 5. Gated Fusion: 基于attention生成门控信号
        # 计算overall attention strength (所有aux的重要性总和)
        attention_strength = np.sum(attention_weights, axis=1, keepdims=True) / n_aux  # (n_samples, 1)
        
        # 生成门控：attention强度高时更信任aux，低时更信任RF
        gate = np.clip(attention_strength, 0.3, 0.7)  # 限制在[0.3, 0.7]之间，避免极端情况
        
        # 6. 基于门控混合RF和Aux信息
        # 用aux的加权摘要调制RF
        aux_summary = np.mean(attended_aux, axis=1, keepdims=True)  # (n_samples, 1)
        rf_modulation = 1.0 + 0.3 * np.tanh(np.tile(aux_summary, (1, n_spectrum)))  # [0.7, 1.3]
        modulated_rf = rf_modulation * spectrum_features
        
        # 7. 融合输出：调制后的RF + 注意力加权的Aux
        fused = np.concatenate([modulated_rf, attended_aux], axis=1)
        
        avg_gate = np.mean(gate)
        avg_attention_entropy = -np.mean(np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1))
        
        print(f"    ✓ Fusion: Attention-Fusion - 输出维度: {fused.shape[1]}")
        print(f"      门控强度: {avg_gate:.3f} (越高越信任aux)")
        print(f"      注意力熵: {avg_attention_entropy:.3f} (越低越集中)")
        print(f"      RF调制范围: [{rf_modulation.min():.3f}, {rf_modulation.max():.3f}]")
        return fused
    
    else:
        raise ValueError(
            f"不支持的fusion方法: {fusion_method}. "
            f"支持的方法: concat, film, film_sensor_aware, attention_pool, residual_film, conditioned_rf, attention_fusion"
        )


@dataclass
class AuxSensorData:
    """辅助传感器数据容器"""
    bme680: pd.DataFrame  # temperature_c, pressure_hpa, humidity_pct
    ppg: pd.DataFrame     # raw (新格式) 或 red, ir (旧格式)
    t117: pd.DataFrame    # temperature_c
    icm: pd.DataFrame     # acc_x,y,z, gyr_x,y,z


class GlucoseDatasetLoader:
    """
    血糖传感数据集加载器
    
    功能：
    1. 从文件夹名称自动解析起始时间
    2. 加载用户的所有实验数据（多个日期文件夹）
    3. 读取 S*.BIN 频谱数据作为特征（power_0 到 power_1000）
    4. 读取 JSON 血糖数据作为标签
    5. 为每个频谱时间戳插值计算对应血糖值
    6. 数据归一化和预处理
    """
    
    def __init__(self, dataset_root: str = "./Dataset"):
        """
        Args:
            dataset_root: Dataset 根目录路径
        """
        self.dataset_root = Path(dataset_root)
    
    def _read_single_aux_file(self, path: Path) -> List[Tuple[int, tuple]]:
        """
        读取单个辅助传感器二进制文件 (A*.bin)
        
        Returns:
            List of (tag, data_tuple) pairs
        """
        records = []
        try:
            with open(path, 'rb') as f:
                # Read header
                hdr = f.read(AUX_HEADER_STRUCT.size)
                if len(hdr) < AUX_HEADER_STRUCT.size:
                    return records
                
                magic, version, reserved = AUX_HEADER_STRUCT.unpack(hdr)
                if magic.rstrip(b'\x00') != AUX_MAGIC:
                    return records
                
                # Stream parse
                while True:
                    tag_byte = f.read(1)
                    if not tag_byte:
                        break
                    tag = tag_byte[0]
                    
                    if tag == TAG_BME680:
                        blob = f.read(REC_BME680.size)
                        if len(blob) < REC_BME680.size:
                            break
                        records.append((tag, REC_BME680.unpack(blob)))
                    elif tag == TAG_PPG:
                        blob = f.read(REC_PPG.size)
                        if len(blob) < REC_PPG.size:
                            break
                        records.append((tag, REC_PPG.unpack(blob)))
                    elif tag == TAG_T117:
                        blob = f.read(REC_T117.size)
                        if len(blob) < REC_T117.size:
                            break
                        records.append((tag, REC_T117.unpack(blob)))
                    elif tag == TAG_ICM:
                        blob = f.read(REC_ICM.size)
                        if len(blob) < REC_ICM.size:
                            break
                        records.append((tag, REC_ICM.unpack(blob)))
                    else:
                        # Unknown tag, break to avoid infinite loop
                        break
        except Exception as e:
            print(f"  ⚠ 读取aux文件失败 {path.name}: {e}")
        
        return records
    
    def load_aux_sensor_files(self, directory: Path) -> AuxSensorData:
        """
        加载目录中所有辅助传感器文件 (A*.bin / a*.bin)
        
        Returns:
            AuxSensorData with separate DataFrames for each sensor
        """
        # Find all aux files (case insensitive)
        aux_files = []
        for pattern in ['a*.bin', 'A*.bin', 'A*.BIN']:
            aux_files.extend(sorted(directory.glob(pattern)))
        
        # Remove duplicates while preserving order
        aux_files = list(dict.fromkeys(aux_files))
        
        if not aux_files:
            # Return empty DataFrames
            empty = pd.DataFrame()
            return AuxSensorData(empty, empty, empty, empty)
        
        # Parse all records
        all_records = []
        for fp in aux_files:
            for tag, data in self._read_single_aux_file(fp):
                all_records.append((tag, data, fp.name))
        
        # Separate into sensor-specific lists
        bme_rows = []
        ppg_rows = []
        t117_rows = []
        icm_rows = []
        
        for tag, data, fname in all_records:
            if tag == TAG_BME680:
                ts, tC, p_hpa, h_pct = data
                bme_rows.append({
                    'timestamp': ts, 
                    'temperature_c': tC, 
                    'pressure_hpa': p_hpa, 
                    'humidity_pct': h_pct,
                    'file_source': fname
                })
            elif tag == TAG_PPG:
                ts, red, ir, hr, spo2, reserved = data
                ppg_rows.append({
                    'timestamp': ts,
                    'red': red,
                    'ir': ir,
                    'heart_rate_bpm': hr if hr >= 0 else np.nan,
                    'spo2_percent': spo2,
                    'file_source': fname
                })
            elif tag == TAG_T117:
                ts, tC = data
                t117_rows.append({
                    'timestamp': ts,
                    'temperature_c': tC,
                    'file_source': fname
                })
            elif tag == TAG_ICM:
                ts, ax, ay, az, gx, gy, gz = data
                icm_rows.append({
                    'timestamp': ts,
                    'acc_x': ax, 'acc_y': ay, 'acc_z': az,
                    'gyr_x': gx, 'gyr_y': gy, 'gyr_z': gz,
                    'file_source': fname
                })
        
        # Create DataFrames and sort by timestamp
        bme_df = pd.DataFrame(bme_rows).sort_values('timestamp').reset_index(drop=True) if bme_rows else pd.DataFrame()
        ppg_df = pd.DataFrame(ppg_rows).sort_values('timestamp').reset_index(drop=True) if ppg_rows else pd.DataFrame()
        t117_df = pd.DataFrame(t117_rows).sort_values('timestamp').reset_index(drop=True) if t117_rows else pd.DataFrame()
        icm_df = pd.DataFrame(icm_rows).sort_values('timestamp').reset_index(drop=True) if icm_rows else pd.DataFrame()
        
        return AuxSensorData(bme_df, ppg_df, t117_df, icm_df)
    
    def interpolate_aux_to_spectrum(
        self, 
        aux_data: AuxSensorData, 
        spectrum_timestamps: np.ndarray,
        normalize: bool = True,
        interpolation_method: str = 'pchip',
        icm_mode: str = 'raw',
        aux_feature_mode: str = 'raw'
    ) -> np.ndarray:
        """
        将辅助传感器数据插值到频谱时间戳，并可选地进行归一化
        
        改进版本：根据传感器特性使用更合适的插值方法
        
        Args:
            aux_data: 辅助传感器数据
            spectrum_timestamps: 频谱时间戳数组
            normalize: 是否对每个传感器特征进行 z-score 归一化（默认True）
            interpolation_method: 插值方法（pchip, akima, cubic, linear, nearest）
                - pchip: 推荐用于温度、压力、湿度等平缓变化的数据
                - linear: 快速但不平滑
                - cubic: 平滑但可能过冲
            icm_mode: ICM数据处理模式
                - 'raw': 使用原始加速度+陀螺仪数据（默认）
                - 'processed': 使用积分处理后的位移+角度数据（更平滑）
                - 'both': 同时使用原始和处理后的数据
            aux_feature_mode: PPG/ICM特征模式
                - 'raw': 保持旧逻辑，逐点插值原始特征
                - 'engineered': 使用按秒聚合的统计/质量/运动特征
                - 'both': 同时拼接raw与engineered特征
                - 'segment_raw': 使用用于engineered聚合的同一秒内PPG/ICM原始序列槽位
            
        Returns:
            aux_features: shape (n_samples, n_aux_features)
            feature_names: 特征名称列表
            特征顺序根据icm_mode而定：
                - raw: [bme_*, ppg_*, t117_*, icm_acc_x/y/z, icm_gyr_x/y/z]
                - processed: [bme_*, ppg_*, t117_*, icm_disp_x/y/z, icm_angle_x/y/z]
                - both: [bme_*, ppg_*, t117_*, icm_acc/gyr_*, icm_disp/angle_*]
        """
        from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
        
        n_samples = len(spectrum_timestamps)
        aux_features_list = []
        feature_names = []
        aux_feature_mode = aux_feature_mode or 'raw'
        
        def smart_interpolate(timestamps, values, target_timestamps, sensor_type='smooth'):
            """
            根据传感器类型智能选择插值方法
            
            Args:
                timestamps: 原始时间戳
                values: 原始值
                target_timestamps: 目标时间戳
                sensor_type: 传感器类型
                    - 'smooth': 平缓变化（温度、压力、湿度） -> 使用pchip或配置的方法
                    - 'rapid': 快速变化（加速度、陀螺仪） -> 使用linear避免过拟合
                    - 'signal': 信号类（PPG） -> 使用配置的方法
            """
            n_points = len(timestamps)
            
            # 排序和去重
            sort_idx = np.argsort(timestamps)
            timestamps = timestamps[sort_idx]
            values = values[sort_idx]
            
            # 去重：保留第一个值
            unique_mask = np.concatenate([[True], np.diff(timestamps) > 0])
            timestamps = timestamps[unique_mask]
            values = values[unique_mask]
            n_points = len(timestamps)
            
            if n_points < 2:
                # 数据点不足，返回常数
                return np.full(len(target_timestamps), np.mean(values) if len(values) > 0 else 0.0)
            
            # 根据传感器类型和数据点数量选择插值方法
            if sensor_type == 'rapid':
                # 快速变化的传感器（加速度、陀螺仪）使用线性插值
                # 避免过度平滑，保留快速变化特征
                interp_func = interp1d(
                    timestamps, values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                return interp_func(target_timestamps)
            
            # 对于平缓变化的传感器，使用配置的高级插值方法
            if interpolation_method == 'pchip' and n_points >= 3:
                interp_func = PchipInterpolator(timestamps, values, extrapolate=False)
                result = interp_func(target_timestamps)
                # 处理外推区域（设为NaN）
                result[target_timestamps < timestamps[0]] = np.nan
                result[target_timestamps > timestamps[-1]] = np.nan
                return result
            
            elif interpolation_method == 'akima' and n_points >= 5:
                interp_func = Akima1DInterpolator(timestamps, values)
                result = interp_func(target_timestamps, extrapolate=False)
                return result
            
            elif interpolation_method == 'cubic' and n_points >= 4:
                interp_func = interp1d(
                    timestamps, values,
                    kind='cubic',
                    bounds_error=False,
                    fill_value=np.nan
                )
                return interp_func(target_timestamps)
            
            else:
                # 降级为线性插值
                interp_func = interp1d(
                    timestamps, values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                return interp_func(target_timestamps)

        def aggregate_by_second(df, agg_specs, extra_columns=None):
            """按Unix秒聚合高频传感器，降低批量写入时间戳和相位噪声影响。"""
            if df.empty or 'timestamp' not in df.columns:
                return pd.DataFrame()
            extra_columns = extra_columns or {}
            valid_specs = {
                out_name: (col, func)
                for out_name, (col, func) in agg_specs.items()
                if col in df.columns or col in extra_columns
            }
            if not valid_specs:
                return pd.DataFrame()

            needed_cols = sorted({col for col, _ in valid_specs.values()})
            work = pd.DataFrame({
                'second': np.floor(df['timestamp'].values).astype(np.int64)
            })
            for col in needed_cols:
                if col in extra_columns:
                    work[col] = np.asarray(extra_columns[col])
                else:
                    work[col] = pd.to_numeric(df[col], errors='coerce').values

            grouped = work.groupby('second').agg(**valid_specs).reset_index()
            grouped = grouped.rename(columns={'second': 'timestamp'})
            return grouped

        def add_segment_raw_features(df, sensor_prefix, columns, slots_per_column):
            """
            按Unix秒取原始序列，填入固定槽位。

            这不是统计特征加工；每个槽位保留该秒内原始采样值。若极少数秒内
            样本超过槽位数，则均匀抽样到固定长度以保持模型输入维度一致。
            """
            if df.empty or 'timestamp' not in df.columns:
                return

            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                return

            work = df[['timestamp'] + valid_columns].copy()
            work = work.sort_values('timestamp').reset_index(drop=True)
            raw_timestamps = work['timestamp'].values
            raw_seconds = np.floor(raw_timestamps).astype(np.int64)
            target_seconds = np.floor(spectrum_timestamps).astype(np.int64)

            unique_seconds, second_starts = np.unique(raw_seconds, return_index=True)
            second_ends = np.r_[second_starts[1:], len(raw_seconds)]
            second_to_range = {
                int(sec): (int(start), int(end))
                for sec, start, end in zip(unique_seconds, second_starts, second_ends)
            }

            truncated_segments = 0
            max_seen = 0
            for col in valid_columns:
                values = pd.to_numeric(work[col], errors='coerce').fillna(0.0).values
                slot_matrix = np.zeros((n_samples, slots_per_column), dtype=np.float32)

                for row_idx, sec in enumerate(target_seconds):
                    value_range = second_to_range.get(int(sec))
                    if value_range is None:
                        continue

                    start, end = value_range
                    segment = values[start:end]
                    if len(segment) == 0:
                        continue

                    max_seen = max(max_seen, len(segment))
                    if len(segment) > slots_per_column:
                        pick_idx = np.linspace(0, len(segment) - 1, slots_per_column).astype(np.int64)
                        segment = segment[pick_idx]
                        truncated_segments += 1

                    slot_matrix[row_idx, :len(segment)] = segment

                for slot_idx in range(slots_per_column):
                    aux_features_list.append(slot_matrix[:, slot_idx])
                    feature_names.append(f'{sensor_prefix}_{col}_seg{slot_idx:03d}')

            print(
                f"    ✓ Segment Raw {sensor_prefix}: columns={len(valid_columns)}, "
                f"slots/column={slots_per_column}, max_samples/second={max_seen}, "
                f"truncated_segments={truncated_segments}"
            )

        if aux_feature_mode == 'segment_raw':
            # 与engineered特征使用相同的Unix秒段，但保留段内原始采样值。
            if not aux_data.ppg.empty:
                if 'raw' in aux_data.ppg.columns:
                    ppg_columns = ['raw']
                else:
                    ppg_columns = [col for col in ['red', 'ir'] if col in aux_data.ppg.columns]
                add_segment_raw_features(aux_data.ppg, 'ppg', ppg_columns, slots_per_column=256)

            if not aux_data.icm.empty:
                icm_columns = [
                    col for col in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
                    if col in aux_data.icm.columns
                ]
                add_segment_raw_features(aux_data.icm, 'icm', icm_columns, slots_per_column=128)

            if not aux_features_list:
                return np.array([]).reshape(n_samples, 0), []

            aux_features = np.column_stack(aux_features_list)
            if normalize:
                for i in range(aux_features.shape[1]):
                    col = aux_features[:, i]
                    col_mean = np.mean(col)
                    col_std = np.std(col)
                    if col_std > 1e-8:
                        aux_features[:, i] = (col - col_mean) / col_std
                    else:
                        aux_features[:, i] = col - col_mean
                aux_features = np.nan_to_num(aux_features, nan=0.0)
            return aux_features, feature_names
        
        # BME680: temperature, pressure, humidity (平缓变化)
        if not aux_data.bme680.empty:
            bme_timestamps = aux_data.bme680['timestamp'].values
            for col in ['temperature_c', 'pressure_hpa', 'humidity_pct']:
                if col in aux_data.bme680.columns:
                    values = aux_data.bme680[col].values
                    interpolated = smart_interpolate(
                        bme_timestamps, values, spectrum_timestamps, 
                        sensor_type='smooth'
                    )
                    aux_features_list.append(interpolated)
                    feature_names.append(f'bme_{col}')
        
        # PPG: raw (单通道原始信号)
        if not aux_data.ppg.empty and aux_feature_mode in ['raw', 'both']:
            ppg_timestamps = aux_data.ppg['timestamp'].values
            # 支持新格式(raw)和旧格式(red, ir)
            if 'raw' in aux_data.ppg.columns:
                # 新格式：只有一维raw数据
                values = aux_data.ppg['raw'].values
                interpolated = smart_interpolate(
                    ppg_timestamps, values, spectrum_timestamps,
                    sensor_type='signal'
                )
                aux_features_list.append(interpolated)
                feature_names.append('ppg_raw')
            else:
                # 旧格式兼容：red, ir两列
                for col in ['red', 'ir']:
                    if col in aux_data.ppg.columns:
                        values = aux_data.ppg[col].values
                        interpolated = smart_interpolate(
                            ppg_timestamps, values, spectrum_timestamps,
                            sensor_type='signal'
                        )
                        aux_features_list.append(interpolated)
                        feature_names.append(f'ppg_{col}')

        if not aux_data.ppg.empty and aux_feature_mode in ['engineered', 'both']:
            ppg_extra_columns = {}
            if 'raw' in aux_data.ppg.columns:
                raw_values = pd.to_numeric(aux_data.ppg['raw'], errors='coerce')
                ppg_extra_columns['raw_clip_flag'] = ((raw_values <= 2) | (raw_values >= 4093)).astype(float).values
            agg_specs = {
                'ppg_raw_mean_1s': ('raw', 'mean'),
                'ppg_raw_std_1s': ('raw', 'std'),
                'ppg_raw_p2_1s': ('raw', lambda x: np.nanpercentile(x, 2)),
                'ppg_raw_p98_1s': ('raw', lambda x: np.nanpercentile(x, 98)),
                'ppg_filter_std_1s': ('filter', 'std'),
                'ppg_hr_median_1s': ('hr', 'median'),
                'ppg_hrv_median_1s': ('hrv', 'median'),
                'ppg_peak_rate_1s': ('peak', 'mean'),
                'ppg_clip_frac_1s': ('raw_clip_flag', 'mean'),
            }
            ppg_agg = aggregate_by_second(aux_data.ppg, agg_specs, extra_columns=ppg_extra_columns)
            if not ppg_agg.empty:
                ppg_timestamps = ppg_agg['timestamp'].values
                for col in [c for c in ppg_agg.columns if c != 'timestamp']:
                    values = ppg_agg[col].fillna(0.0).values
                    interpolated = smart_interpolate(
                        ppg_timestamps, values, spectrum_timestamps,
                        sensor_type='signal'
                    )
                    aux_features_list.append(interpolated)
                    feature_names.append(col)
        
        # T117: temperature (平缓变化)
        if not aux_data.t117.empty:
            t117_timestamps = aux_data.t117['timestamp'].values
            values = aux_data.t117['temperature_c'].values
            interpolated = smart_interpolate(
                t117_timestamps, values, spectrum_timestamps,
                sensor_type='smooth'
            )
            aux_features_list.append(interpolated)
            feature_names.append('t117_temperature_c')
        
        # ICM: accelerometer (x, y, z) and gyroscope (x, y, z)
        # 支持三种模式：raw(原始), processed(处理后), both(两者都用)
        if not aux_data.icm.empty:
            icm_timestamps = aux_data.icm['timestamp'].values
            
            # 提取原始数据
            acc_x = aux_data.icm['acc_x'].values if 'acc_x' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            acc_y = aux_data.icm['acc_y'].values if 'acc_y' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            acc_z = aux_data.icm['acc_z'].values if 'acc_z' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            gyr_x = aux_data.icm['gyr_x'].values if 'gyr_x' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            gyr_y = aux_data.icm['gyr_y'].values if 'gyr_y' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            gyr_z = aux_data.icm['gyr_z'].values if 'gyr_z' in aux_data.icm.columns else np.zeros(len(icm_timestamps))
            
            # 插值到spectrum时间戳
            acc_x_interp = smart_interpolate(icm_timestamps, acc_x, spectrum_timestamps, sensor_type='rapid')
            acc_y_interp = smart_interpolate(icm_timestamps, acc_y, spectrum_timestamps, sensor_type='rapid')
            acc_z_interp = smart_interpolate(icm_timestamps, acc_z, spectrum_timestamps, sensor_type='rapid')
            gyr_x_interp = smart_interpolate(icm_timestamps, gyr_x, spectrum_timestamps, sensor_type='rapid')
            gyr_y_interp = smart_interpolate(icm_timestamps, gyr_y, spectrum_timestamps, sensor_type='rapid')
            gyr_z_interp = smart_interpolate(icm_timestamps, gyr_z, spectrum_timestamps, sensor_type='rapid')
            
            if aux_feature_mode in ['raw', 'both'] and icm_mode in ['raw', 'both']:
                # 添加原始加速度和陀螺仪数据
                aux_features_list.extend([acc_x_interp, acc_y_interp, acc_z_interp])
                feature_names.extend(['icm_acc_x', 'icm_acc_y', 'icm_acc_z'])
                aux_features_list.extend([gyr_x_interp, gyr_y_interp, gyr_z_interp])
                feature_names.extend(['icm_gyr_x', 'icm_gyr_y', 'icm_gyr_z'])
            
            if aux_feature_mode in ['raw', 'both'] and icm_mode in ['processed', 'both']:
                # 处理成统计特征（更鲁棒的方法，避免积分漂移）
                def process_imu_to_motion_features(acc_data, gyr_data, timestamps):
                    """
                    将加速度和陀螺仪数据转换为运动特征
                    使用统计和频域特征而非积分，避免漂移问题
                    
                    Args:
                        acc_data: 加速度数据 (m/s^2 或 g)
                        gyr_data: 角速度数据 (rad/s 或 deg/s)
                        timestamps: 时间戳
                    
                    Returns:
                        acc_feature: 加速度特征（RMS或滤波后的信号）
                        gyr_feature: 角速度特征（RMS或滤波后的信号）
                    """
                    # 处理NaN值
                    valid_mask = ~np.isnan(acc_data) & ~np.isnan(gyr_data)
                    if not np.any(valid_mask):
                        return np.zeros_like(acc_data), np.zeros_like(gyr_data)
                    
                    # 方法1: 去除静态偏置后直接使用（最简单，效果往往更好）
                    # 去除静态偏置
                    acc_bias = np.nanmedian(acc_data)
                    acc_feature = acc_data - acc_bias
                    
                    gyr_bias = np.nanmedian(gyr_data)
                    gyr_feature = gyr_data - gyr_bias
                    
                    # 方法2: 应用轻度平滑滤波（可选）
                    try:
                        from scipy.signal import butter, filtfilt
                        
                        dt = np.median(np.diff(timestamps[valid_mask])) if np.sum(valid_mask) > 1 else 1.0
                        fs = 1.0 / dt if dt > 0 else 1.0
                        
                        # 低通滤波器（截止频率: 5Hz，保留更多运动信息）
                        if fs > 10 and len(acc_feature) > 15:
                            b_low, a_low = butter(2, 5.0/(fs/2), btype='low')
                            acc_feature = filtfilt(b_low, a_low, acc_feature, padlen=min(len(acc_feature)//2, 15))
                            gyr_feature = filtfilt(b_low, a_low, gyr_feature, padlen=min(len(gyr_feature)//2, 15))
                    except Exception:
                        # 滤波失败时使用去偏置后的数据
                        pass
                    
                    # 方法3: 计算运动强度（RMS）- 对于window模式，可以在模型中使用
                    # window_size = 10  # 10秒窗口
                    # acc_rms = np.sqrt(np.convolve(acc_feature**2, np.ones(window_size)/window_size, mode='same'))
                    # gyr_rms = np.sqrt(np.convolve(gyr_feature**2, np.ones(window_size)/window_size, mode='same'))
                    
                    return acc_feature, gyr_feature
                
                # 处理每个轴 - 使用改进的统计特征方法
                acc_x_feat, gyr_x_feat = process_imu_to_motion_features(acc_x_interp, gyr_x_interp, spectrum_timestamps)
                acc_y_feat, gyr_y_feat = process_imu_to_motion_features(acc_y_interp, gyr_y_interp, spectrum_timestamps)
                acc_z_feat, gyr_z_feat = process_imu_to_motion_features(acc_z_interp, gyr_z_interp, spectrum_timestamps)
                
                # 添加处理后的加速度和陀螺仪特征
                aux_features_list.extend([acc_x_feat, acc_y_feat, acc_z_feat])
                feature_names.extend(['icm_acc_x_proc', 'icm_acc_y_proc', 'icm_acc_z_proc'])
                aux_features_list.extend([gyr_x_feat, gyr_y_feat, gyr_z_feat])
                feature_names.extend(['icm_gyr_x_proc', 'icm_gyr_y_proc', 'icm_gyr_z_proc'])

            if aux_feature_mode in ['engineered', 'both']:
                def numeric_icm_column(col_name):
                    if col_name in aux_data.icm.columns:
                        return pd.to_numeric(aux_data.icm[col_name], errors='coerce').values
                    return np.zeros(len(aux_data.icm), dtype=np.float32)

                acc_norm = np.sqrt(
                    np.square(numeric_icm_column('acc_x')) +
                    np.square(numeric_icm_column('acc_y')) +
                    np.square(numeric_icm_column('acc_z'))
                )
                gyro_norm = np.sqrt(
                    np.square(numeric_icm_column('gyr_x')) +
                    np.square(numeric_icm_column('gyr_y')) +
                    np.square(numeric_icm_column('gyr_z'))
                )
                icm_engineered = pd.DataFrame({
                    'timestamp': aux_data.icm['timestamp'].values,
                    'acc_norm': np.asarray(acc_norm),
                    'gyro_norm': np.asarray(gyro_norm),
                    'acc_dynamic': np.asarray(acc_norm) - np.nanmedian(acc_norm),
                })
                agg_specs = {
                    'icm_acc_norm_mean_1s': ('acc_norm', 'mean'),
                    'icm_acc_dynamic_rms_1s': ('acc_dynamic', lambda x: np.sqrt(np.nanmean(np.square(x)))),
                    'icm_acc_norm_std_1s': ('acc_norm', 'std'),
                    'icm_gyro_energy_1s': ('gyro_norm', lambda x: np.sqrt(np.nanmean(np.square(x)))),
                    'icm_gyro_std_1s': ('gyro_norm', 'std'),
                }
                icm_agg = aggregate_by_second(icm_engineered, agg_specs)
                if not icm_agg.empty:
                    icm_timestamps_agg = icm_agg['timestamp'].values
                    for col in [c for c in icm_agg.columns if c != 'timestamp']:
                        values = icm_agg[col].fillna(0.0).values
                        interpolated = smart_interpolate(
                            icm_timestamps_agg, values, spectrum_timestamps,
                            sensor_type='rapid'
                        )
                        aux_features_list.append(interpolated)
                        feature_names.append(col)
        
        if not aux_features_list:
            # No aux sensor data available, return empty array
            return np.array([]).reshape(n_samples, 0), []
        
        # Stack features
        aux_features = np.column_stack(aux_features_list)
        
        # Handle NaN values using forward fill strategy
        # 使用前向填充：用最后一个有效值填充后续缺失数据（适用于PPG等中途信号丢失的传感器）
        for i in range(aux_features.shape[1]):
            col = aux_features[:, i]
            if np.isnan(col).any():
                # Forward fill: 用前面最近的有效值填充
                last_valid = None
                for j in range(len(col)):
                    if not np.isnan(col[j]):
                        last_valid = col[j]
                    elif last_valid is not None:
                        col[j] = last_valid
                
                # 如果开头有NaN（还没有有效值），用后面第一个有效值填充
                if np.isnan(col[0]):
                    first_valid_idx = np.where(~np.isnan(col))[0]
                    if len(first_valid_idx) > 0:
                        first_valid = col[first_valid_idx[0]]
                        for j in range(first_valid_idx[0]):
                            col[j] = first_valid
                    else:
                        # 如果全是NaN，填充0
                        col[:] = 0.0
                
                aux_features[:, i] = col
        
        # Apply z-score normalization to each aux sensor feature
        if normalize:
            for i in range(aux_features.shape[1]):
                col = aux_features[:, i]
                col_mean = np.mean(col)
                col_std = np.std(col)
                
                # Avoid division by zero
                if col_std > 1e-8:
                    aux_features[:, i] = (col - col_mean) / col_std
                else:
                    # 如果标准差太小，只做去均值，不做缩放
                    aux_features[:, i] = col - col_mean
                    
            # 最终检查归一化后是否有NaN
            if np.any(np.isnan(aux_features)):
                nan_count = np.sum(np.isnan(aux_features))
                print(f"    ⚠ 警告: 归一化后aux_features包含{nan_count}个NaN值")
                # 找出哪些列有NaN
                for i in range(aux_features.shape[1]):
                    if np.any(np.isnan(aux_features[:, i])):
                        print(f"      - {feature_names[i]}: {np.sum(np.isnan(aux_features[:, i]))}个NaN")
                # 替换为0
                aux_features = np.nan_to_num(aux_features, nan=0.0)
        
        return aux_features, feature_names
    
    @staticmethod
    def parse_start_time_from_dirname(dirname: str) -> Optional[str]:
        """
        从目录名提取起始时间
        
        例如: "11201828_Tao" -> "11.20 18:28"
              "11141404_Tao" -> "11.14 14:04"
        
        Args:
            dirname: 目录名称
            
        Returns:
            时间字符串 "MM.DD HH:MM" 或 None
        """
        # 匹配格式: MMDDHHMM_User
        pattern = r'^(\d{2})(\d{2})(\d{2})(\d{2})_'
        match = re.match(pattern, dirname)
        
        if match:
            month, day, hour, minute = match.groups()
            return f"{month}.{day} {hour}:{minute}"
        
        return None
        
    def load_spectrum_file(self, file_path: Path) -> pd.DataFrame:
        """
        加载单个频谱 BIN 文件
        
        文件格式:
        - Header: Magic(8字节) + version(4字节) + record_header_size(4字节) + max_points(4字节)
        - Records: timestamp(8字节double) + num_points(2字节) + power_data(num_points*4字节)
        
        Returns:
            DataFrame with columns: timestamp, num_points, power_0, power_1, ...
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        records = []
        
        with open(file_path, 'rb') as f:
            # 读取文件头
            header = f.read(20)
            if len(header) < 20:
                raise ValueError(f'{file_path.name}: Header too short')
            
            magic, version, record_header_size, max_points = struct.unpack('<8s3I', header)
            
            if magic != b'SPECDATA':
                raise ValueError(f'{file_path.name}: Invalid magic {magic}')
            
            # 读取记录
            while True:
                rec_hdr = f.read(10)
                if len(rec_hdr) < 10:
                    break
                
                timestamp, num_points = struct.unpack('<dH', rec_hdr)
                
                # 跳过无效记录
                if num_points == 0xFFFF:
                    continue
                
                # 读取功率数据
                data_bytes = f.read(num_points * 4)
                if len(data_bytes) < num_points * 4:
                    break
                
                powers = struct.unpack(f'<{num_points}i', data_bytes)
                
                # 构建记录
                record = {'timestamp': timestamp, 'num_points': num_points}
                for i, power in enumerate(powers):
                    record[f'power_{i}'] = power
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def load_spectrum_files(self, directory: Path, pattern: str = 's*.bin') -> pd.DataFrame:
        """
        加载目录中所有匹配的频谱文件并拼接（支持大小写不敏感）
        
        按照 notebook 的做法：直接拼接所有 S*.BIN 文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            合并后的 DataFrame，只保留 power_0 到 power_1000 列
        """
        dfs = []
        
        # 支持大小写不敏感：尝试小写和大写模式
        patterns = [pattern]
        if pattern != pattern.upper():
            patterns.append(pattern.upper())
        
        all_files = []
        for p in patterns:
            files = sorted(directory.glob(p))
            all_files.extend(files)
        
        # 去重
        all_files = list(dict.fromkeys(all_files))
        
        for file_path in all_files:
            try:
                df = self.load_spectrum_file(file_path)
                if not df.empty:
                    df['file_source'] = file_path.name
                    dfs.append(df)
            except Exception as e:
                print(f"  ⚠ 跳过 {file_path.name}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        # 合并并按时间戳排序
        combined = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
        combined = combined.reset_index(drop=True)
        
        # 只保留 power_0 到 power_1000 列（1001维）
        power_cols = [f'power_{i}' for i in range(1001)]
        available_power_cols = [col for col in power_cols if col in combined.columns]
        
        if len(available_power_cols) < 1001:
            print(f"  ⚠ 警告: 只找到 {len(available_power_cols)} 个功率列，期望 1001")
        
        # 保留时间戳和功率列
        result_cols = ['timestamp'] + available_power_cols
        combined = combined[result_cols]
        
        return combined
    
    def load_glucose_json(self, json_path: Path) -> pd.DataFrame:
        """
        加载血糖 JSON 文件
        
        JSON 格式: [{"time": "MM.DD HH:MM", "value": float}, ...]
        
        Returns:
            DataFrame with columns: time, value, datetime, rel_seconds
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            glucose_data = json.load(f)
        
        df = pd.DataFrame(glucose_data)
        
        # 解析时间字符串 (假设年份为2025)
        df['datetime'] = pd.to_datetime('2025.' + df['time'], format='%Y.%m.%d %H:%M')
        
        return df
    
    def align_spectrum_glucose(
        self,
        spectrum_df: pd.DataFrame,
        glucose_df: pd.DataFrame,
        start_time_str: str,
        interpolation_method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对齐频谱数据和血糖数据的时间戳，并插值
        
        改进的插值方法，针对CGM低采样率数据进行优化
        
        Args:
            spectrum_df: 频谱数据 DataFrame (包含 timestamp 和 power_* 列)
            glucose_df: 血糖数据 DataFrame (包含 datetime 和 value 列)
            start_time_str: 实验开始时间字符串 (格式: "MM.DD HH:MM")
            interpolation_method: 插值方法
                - 'linear': 线性插值（默认，快速但可能不够平滑）
                - 'cubic': 三次样条插值（平滑，适合CGM数据）
                - 'pchip': 保形分段三次插值（单调性保持，避免过冲）
                - 'akima': Akima插值（避免过冲和振荡）
                - 'makima': 修改的Akima插值（更稳定）
                - 'nearest': 最近邻（适合步进式变化）
            
        Returns:
            (spectrum_features, glucose_labels, spectrum_timestamps)
            - spectrum_features: shape (n_samples, 1001)
            - glucose_labels: shape (n_samples,)
            - spectrum_timestamps: shape (n_samples,)
        """
        from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
        
        # 解析开始时间
        start_dt = datetime.strptime('2025.' + start_time_str, '%Y.%m.%d %H:%M')
        
        # 提取功率列作为特征 (power_0 到 power_1000)
        power_cols = [col for col in spectrum_df.columns if col.startswith('power_')]
        spectrum_features = spectrum_df[power_cols].to_numpy()
        spectrum_timestamps = spectrum_df['timestamp'].to_numpy()
        
        # 血糖数据转换为相对时间（秒）
        glucose_df = glucose_df.copy()
        glucose_df['rel_seconds'] = (glucose_df['datetime'] - start_dt).dt.total_seconds()
        
        # 排序并去重（确保时间戳单调递增）
        glucose_df = glucose_df.sort_values('rel_seconds').drop_duplicates('rel_seconds')
        
        # 过滤：只保留有效时间范围内的血糖数据
        glucose_df = glucose_df[
            (glucose_df['rel_seconds'] >= spectrum_timestamps.min()) &
            (glucose_df['rel_seconds'] <= spectrum_timestamps.max())
        ]
        
        glucose_times = glucose_df['rel_seconds'].values
        glucose_values = glucose_df['value'].values
        
        # 根据数据点数量和方法选择合适的插值器
        n_glucose_points = len(glucose_df)
        
        if n_glucose_points < 2:
            raise ValueError(f"血糖数据点不足（需要至少2个点进行插值，当前只有{n_glucose_points}个点）")
        
        # 计算血糖数据的平均采样间隔（用于分析）
        if n_glucose_points > 1:
            glucose_intervals = np.diff(glucose_times)
            avg_glucose_interval = np.mean(glucose_intervals)
            print(f"    血糖数据: {n_glucose_points}个采样点, 平均间隔: {avg_glucose_interval/60:.1f}分钟")
        
        # 选择插值方法
        if interpolation_method == 'pchip':
            # PCHIP: 保形分段三次Hermite插值
            # 优点: 保持单调性，避免过冲，适合CGM数据
            if n_glucose_points < 3:
                print(f"    ⚠ 数据点不足，PCHIP需要至少3个点，降级为线性插值")
                interpolation_method = 'linear'
            else:
                glucose_interp = PchipInterpolator(glucose_times, glucose_values, extrapolate=True)
                glucose_labels = glucose_interp(spectrum_timestamps)
                print(f"    ✓ 使用PCHIP插值（保形三次插值，避免过冲）")
                return spectrum_features, glucose_labels, spectrum_timestamps
        
        elif interpolation_method == 'akima':
            # Akima插值: 避免振荡和过冲
            # 优点: 在数据点处连续，对异常值不敏感
            if n_glucose_points < 5:
                print(f"    ⚠ 数据点不足，Akima需要至少5个点，降级为PCHIP或线性插值")
                if n_glucose_points >= 3:
                    glucose_interp = PchipInterpolator(glucose_times, glucose_values, extrapolate=True)
                    glucose_labels = glucose_interp(spectrum_timestamps)
                    print(f"    ✓ 使用PCHIP插值（降级）")
                    return spectrum_features, glucose_labels, spectrum_timestamps
                else:
                    interpolation_method = 'linear'
            else:
                glucose_interp = Akima1DInterpolator(glucose_times, glucose_values)
                glucose_labels = glucose_interp(spectrum_timestamps, extrapolate=True)
                print(f"    ✓ 使用Akima插值（避免振荡）")
                return spectrum_features, glucose_labels, spectrum_timestamps
        
        elif interpolation_method == 'makima':
            # Modified Akima插值: scipy 1.10+
            try:
                from scipy.interpolate import make_interp_spline
                # 使用 Akima 作为替代
                if n_glucose_points < 5:
                    print(f"    ⚠ 数据点不足，降级为PCHIP插值")
                    glucose_interp = PchipInterpolator(glucose_times, glucose_values, extrapolate=True)
                else:
                    glucose_interp = Akima1DInterpolator(glucose_times, glucose_values)
                glucose_labels = glucose_interp(spectrum_timestamps, extrapolate=True)
                print(f"    ✓ 使用Modified Akima插值")
                return spectrum_features, glucose_labels, spectrum_timestamps
            except ImportError:
                print(f"    ⚠ scipy版本不支持makima，降级为Akima插值")
                interpolation_method = 'akima'
        
        elif interpolation_method == 'cubic':
            # 三次样条插值
            if n_glucose_points < 4:
                print(f"    ⚠ 数据点不足，cubic需要至少4个点，降级为线性插值")
                interpolation_method = 'linear'
            else:
                glucose_interp = interp1d(
                    glucose_times, glucose_values,
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                glucose_labels = glucose_interp(spectrum_timestamps)
                print(f"    ✓ 使用三次样条插值")
                return spectrum_features, glucose_labels, spectrum_timestamps
        
        # 默认或降级：线性插值或最近邻
        glucose_interp = interp1d(
            glucose_times, glucose_values,
            kind=interpolation_method if interpolation_method in ['linear', 'nearest'] else 'linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        glucose_labels = glucose_interp(spectrum_timestamps)
        
        method_name = 'linear' if interpolation_method not in ['linear', 'nearest'] else interpolation_method
        print(f"    ✓ 使用{method_name}插值")
        
        return spectrum_features, glucose_labels, spectrum_timestamps
    
    def load_user_experiment(
        self,
        experiment_dir: Path,
        interpolation_method: str = 'linear',
        data_fusion: bool = False,
        fusion_method: str = 'concat',
        fusion_stage: str = 'early',  # 新增参数：'early' 或 'late'
        downsample: bool = False,
        downsample_interval: float = 1.0,
        smooth_config: Optional[dict] = None,
        icm_mode: str = 'raw',
        fusion_config: Optional[dict] = None,  # 融合配置（包括aux_override等）
        user_name: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        加载单个实验的数据
        
        改进：自动从目录名提取起始时间，支持辅助传感器数据融合，支持数据下采样，支持数据平滑
        
        数据处理流程（保证aux与spectrum完全对齐）：
        1. 加载spectrum数据 → spectrum_features, timestamps
        2. 插值glucose到spectrum的timestamps → labels
        3. (可选) 平滑处理 → spectrum_features (timestamps不变)
        4. (可选) 下采样 → spectrum_features, labels, timestamps 同步下采样
        5. (如果data_fusion) 插值aux数据到当前的timestamps → aux_features
        6. 融合策略:
           - fusion_stage='early': features = apply_data_fusion(spectrum, aux) 数据层融合
           - fusion_stage='late': 返回分离的(spectrum, aux)，在模型层融合
        
        关键特性：
        - Aux数据插值到与spectrum完全相同的timestamps（经过所有预处理后）
        - 后续的split、window创建等操作会同时处理spectrum+aux，自动保持对齐
        - Down sampling在aux融合前完成，aux数据自动适应downsampled的时间轴
        - 支持多种fusion方法：concat, film, weighted_sum, multiply, attention
        
        Args:
            experiment_dir: 实验目录路径 (例如 11201828_Tao)
            interpolation_method: 插值方法
            data_fusion: 是否融合辅助传感器数据
            fusion_method: 融合方法 (concat, film, weighted_sum, multiply, attention)
            fusion_stage: 融合阶段 ('early'=数据层, 'late'=特征层/模型层)
            downsample: 是否对数据进行下采样
            downsample_interval: 下采样时间间隔（秒）
            smooth_config: 平滑配置字典，包含所有平滑相关参数
            
        Returns:
            (features, labels, timestamps, metadata) 或 None（如果加载失败）
            - features: shape (n_samples, n_features)
                       如果data_fusion=False: n_features=1001 (spectrum only)
                       如果data_fusion=True: n_features取决于fusion_method
            - labels: shape (n_samples,)
            - timestamps: shape (n_samples,)
            - metadata: 实验元数据字典
        """
        try:
            print(f"\n加载实验: {experiment_dir.name}")
            
            # 从目录名自动提取起始时间
            start_time_str = self.parse_start_time_from_dirname(experiment_dir.name)
            if not start_time_str:
                print(f"  ✗ 无法从目录名 {experiment_dir.name} 解析起始时间")
                return None
            
            print(f"  - 起始时间: {start_time_str}")
            
            # 1. 加载频谱数据
            print("  - 加载频谱数据...")
            spectrum_df = self.load_spectrum_files(experiment_dir, pattern='s*.bin')
            
            if spectrum_df.empty:
                print(f"  ✗ 未找到频谱数据")
                return None
            
            print(f"    ✓ {len(spectrum_df)} 条频谱记录")
            
            # 2. 加载血糖数据
            print("  - 加载血糖数据...")
            # 查找 JSON 文件 (支持不同命名格式)
            json_files = list(experiment_dir.glob("*.json"))
            
            if not json_files:
                print(f"  ✗ 未找到 JSON 文件")
                return None
            
            json_path = json_files[0]
            glucose_df = self.load_glucose_json(json_path)
            print(f"    ✓ {len(glucose_df)} 条血糖记录")
            
            # 3. 时间对齐和插值
            print("  - 对齐时间戳并插值...")
            spectrum_features, labels, timestamps = self.align_spectrum_glucose(
                spectrum_df,
                glucose_df,
                start_time_str,
                interpolation_method
            )
            
            # 3.5 数据平滑处理（可选 - 必须在下采样前进行，作为抗混叠滤波器）
            if smooth_config and smooth_config.get('smooth_data', False):
                print("  - 应用数据平滑处理（下采样前的抗混叠预处理）...")
                from .smoothing import apply_smoothing_pipeline
                
                # 提取平滑参数
                time_smooth_method = smooth_config.get('time_smooth_method', 'none')
                spectrum_smooth_method = smooth_config.get('spectrum_smooth_method', 'none')
                
                # 构建时间轴平滑参数
                time_smooth_params = {}
                if time_smooth_method in ['moving_average', 'median', 'savgol']:
                    time_smooth_params['window_size'] = smooth_config.get('time_smooth_window', 5)
                if time_smooth_method == 'savgol':
                    time_smooth_params['poly_order'] = 3
                if time_smooth_method == 'gaussian':
                    time_smooth_params['sigma'] = smooth_config.get('time_smooth_sigma', 2.0)
                if time_smooth_method == 'ema':
                    time_smooth_params['alpha'] = smooth_config.get('time_smooth_alpha', 0.3)
                
                # 构建频谱轴平滑参数
                spectrum_smooth_params = {}
                if spectrum_smooth_method in ['moving_average', 'median', 'savgol']:
                    spectrum_smooth_params['window_size'] = smooth_config.get('spectrum_smooth_window', 5)
                if spectrum_smooth_method == 'savgol':
                    spectrum_smooth_params['poly_order'] = 3
                if spectrum_smooth_method == 'gaussian':
                    spectrum_smooth_params['sigma'] = smooth_config.get('spectrum_smooth_sigma', 2.0)
                
                # 应用平滑处理（包括异常值检测和处理）
                spectrum_features, smooth_info = apply_smoothing_pipeline(
                    spectrum_features,
                    time_smooth_method=time_smooth_method,
                    time_smooth_params=time_smooth_params,
                    spectrum_smooth_method=spectrum_smooth_method,
                    spectrum_smooth_params=spectrum_smooth_params,
                    detect_outliers=smooth_config.get('detect_outliers', False),
                    outlier_method=smooth_config.get('outlier_method', 'zscore'),
                    outlier_threshold=smooth_config.get('outlier_threshold', 3.0),
                    handle_outliers=smooth_config.get('handle_outliers', False),
                    outlier_handle_method=smooth_config.get('outlier_handle_method', 'interpolate')
                )
                
                # 打印平滑信息
                if 'outliers_detected' in smooth_info:
                    print(f"    ✓ 检测到 {smooth_info['outliers_detected']} 个异常值 "
                          f"({smooth_info['outlier_ratio']*100:.2f}%)")
                    if smooth_info.get('outliers_handled'):
                        print(f"    ✓ 异常值已使用 {smooth_config.get('outlier_handle_method', 'interpolate')} 方法处理")
                
                if time_smooth_method != 'none':
                    print(f"    ✓ 时间轴平滑: {time_smooth_method}")
                if spectrum_smooth_method != 'none':
                    print(f"    ✓ 频谱轴平滑: {spectrum_smooth_method}")
            
            # 3.6 下采样（可选 - 必须在平滑后进行，利用平滑结果避免混叠效应）
            if downsample:
                print(f"  - 下采样数据（间隔: {downsample_interval}秒）...")
                original_count = len(timestamps)
                original_duration = timestamps[-1] - timestamps[0]
                
                # 计算下采样索引：从第一个时间点开始，每隔 downsample_interval 秒取一个样本
                # 创建新的时间网格
                start_time = timestamps[0]
                end_time = timestamps[-1]
                new_timestamps = np.arange(start_time, end_time, downsample_interval)
                
                # 找到最接近新时间网格的原始样本索引
                downsample_indices = []
                for target_time in new_timestamps:
                    # 找到最接近 target_time 的原始时间戳索引
                    idx = np.argmin(np.abs(timestamps - target_time))
                    downsample_indices.append(idx)
                
                # 去重（防止相邻的目标时间点映射到同一个原始样本）
                downsample_indices = sorted(list(set(downsample_indices)))
                
                # 应用下采样
                spectrum_features = spectrum_features[downsample_indices]
                labels = labels[downsample_indices]
                timestamps = timestamps[downsample_indices]
                
                new_duration = timestamps[-1] - timestamps[0]
                avg_interval = np.mean(np.diff(timestamps))
                print(f"    ✓ 下采样完成: {original_count} -> {len(timestamps)} 样本")
                print(f"    ✓ 时长: {original_duration:.1f}秒 -> {new_duration:.1f}秒")
                print(f"    ✓ 平均采样间隔: {avg_interval:.2f}秒")
            
            # 4. 辅助传感器数据融合（可选）
            # 关键：aux数据在此处插值到与spectrum完全相同的timestamps
            # 这确保了在后续的split、window创建等操作中，aux和spectrum完全对齐
            aux_feature_names = []
            if data_fusion:
                # 只有启用data_fusion时才读取辅助传感器数据
                print("  - 加载辅助传感器数据...")
                aux_data = self.load_aux_sensor_files(experiment_dir)
                
                # 统计各传感器数据
                aux_counts = {
                    'BME680': len(aux_data.bme680),
                    'PPG': len(aux_data.ppg),
                    'T117': len(aux_data.t117),
                    'ICM': len(aux_data.icm)
                }
                available_sensors = [k for k, v in aux_counts.items() if v > 0]
                
                if available_sensors:
                    print(f"    ✓ 找到传感器: {', '.join(available_sensors)}")
                    for sensor, count in aux_counts.items():
                        if count > 0:
                            print(f"      - {sensor}: {count} 条记录")
                    print(f"\n    🎉 辅助传感器数据加载成功! 共{len(available_sensors)}种传感器")
                    
                    # 插值到频谱时间戳（经过下采样和平滑处理后的timestamps）
                    # 使用与血糖数据相同的插值方法，并进行 z-score 归一化
                    # 重要：此时timestamps已经是最终的时间轴，包含了所有预处理（平滑、下采样）
                    print(f"    辅助传感器插值到spectrum的{len(timestamps)}个时间点")
                    print(f"    辅助传感器插值方法: {interpolation_method}")
                    print(f"      - 温度/压力/湿度: {interpolation_method}（平缓变化）")
                    print(f"      - 加速度/陀螺仪: linear（快速变化，保留细节）")
                    aux_features, aux_feature_names = self.interpolate_aux_to_spectrum(
                        aux_data, timestamps, 
                        normalize=True,
                        interpolation_method=interpolation_method,
                        icm_mode=icm_mode
                    )
                    
                    # [可选] 覆盖aux sensor数值（用于消融实验）
                    if fusion_config and fusion_config.get('aux_override', False):
                        aux_override_value = fusion_config.get('aux_override_value', 0.0)
                        print(f"    ⚠ 覆盖aux sensor数值为: {aux_override_value} (消融实验模式)")
                        aux_features = np.full_like(aux_features, aux_override_value)

                    if fusion_config and fusion_config.get('aux_shuffle', False):
                        aux_shuffle_mode = fusion_config.get('aux_shuffle_mode', 'sensor_groups')
                        aux_shuffle_seed = int(fusion_config.get('aux_shuffle_seed', 42))
                        experiment_seed = aux_shuffle_seed + sum(ord(ch) for ch in experiment_dir.name)
                        aux_features = shuffle_aux_features_for_truth_test(
                            aux_features,
                            aux_feature_names,
                            mode=aux_shuffle_mode,
                            seed=experiment_seed
                        )
                        print(
                            f"    ⚠ Aux Shuffle真值验证: mode={aux_shuffle_mode}, seed={experiment_seed} "
                            f"(仅打乱aux，spectrum-血糖对应保持不变)"
                        )
                    
                    if aux_features.shape[1] > 0:
                        # 根据fusion_stage决定融合策略
                        if fusion_stage == 'early':
                            # Early Fusion: 在数据层融合spectrum和aux
                            features = apply_data_fusion(spectrum_features, aux_features, fusion_method)
                            print(f"    ✓ Early Fusion - 融合后特征维度: {features.shape[1]} (spectrum: {spectrum_features.shape[1]}, aux: {aux_features.shape[1]})")
                            print(f"    ✓ 辅助传感器已进行 z-score 归一化")
                            print(f"    ✓ Aux数据与spectrum数据完全对齐（相同的{len(timestamps)}个timestamps）")
                            print(f"\n    ✅ 数据层融合成功! 方法: {fusion_method}, 最终维度: {features.shape[1]}\n")
                        else:
                            # Late Fusion: 不在数据层融合，保持分离
                            # 返回拼接的数据，但在metadata中记录分割点
                            # 这样后续处理可以正确分离spectrum和aux
                            features = np.concatenate([spectrum_features, aux_features], axis=1)
                            print(f"    ✓ Late Fusion - 数据层不融合，保持分离")
                            print(f"    ✓ 特征维度: spectrum={spectrum_features.shape[1]}, aux={aux_features.shape[1]}")
                            print(f"    ✓ 拼接后总维度: {features.shape[1]} (将在模型层分离并融合)")
                            print(f"\n    ✅ Late Fusion准备完成! 将在模型特征层进行融合\n")
                    else:
                        features = spectrum_features
                        aux_features = None
                        print(f"    ⚠ 未能提取辅助传感器特征，仅使用频谱数据")
                else:
                    features = spectrum_features
                    aux_features = None
                    print(f"    ⚠ 未找到辅助传感器数据，仅使用频谱数据")
            else:
                # 不使用数据融合时，直接使用频谱特征，不加载aux sensor文件
                features = spectrum_features
                aux_features = None
            
            print(f"    ✓ 最终特征shape: {features.shape}, 标签shape: {labels.shape}")
            print(f"    ✓ 血糖范围: {labels.min():.2f} - {labels.max():.2f} mmol/L")
            
            # 5. 元数据
            metadata = {
                'experiment_dir': experiment_dir.name,
                'experiment_name': experiment_dir.name,  # 添加这个字段用于后续处理
                'user_name': user_name,
                'start_time': start_time_str,
                'n_samples': len(features),
                'n_features': features.shape[1],
                'n_spectrum_features': spectrum_features.shape[1],
                'n_aux_features': len(aux_feature_names) if data_fusion and aux_features is not None else 0,
                'aux_feature_names': aux_feature_names if data_fusion and aux_features is not None else [],
                'data_fusion': data_fusion,
                'fusion_stage': fusion_stage,  # 记录融合阶段
                'glucose_mean': float(np.mean(labels)),
                'glucose_std': float(np.std(labels)),
                'glucose_min': float(np.min(labels)),
                'glucose_max': float(np.max(labels)),
            }
            
            return features, labels, timestamps, metadata
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_user_all_experiments(
        self,
        user_name: str,
        interpolation_method: str = 'linear',
        experiment_filter: Optional[List[str]] = None,
        data_fusion: bool = False,
        fusion_method: str = 'concat',
        fusion_stage: str = 'early',  # 新增参数
        downsample: bool = False,
        downsample_interval: float = 1.0,
        smooth_config: Optional[dict] = None,
        icm_mode: str = 'raw',
        fusion_config: Optional[dict] = None  # 融合配置（包括aux_override等）
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
        """
        加载用户的所有实验数据
        
        改进：不再需要手动配置每个实验的起始时间，自动从目录名提取，支持辅助传感器数据融合、数据下采样和数据平滑
        
        Args:
            user_name: 用户名称 (例如 "Tao")
            interpolation_method: 插值方法
            experiment_filter: 可选的实验目录名列表，用于只加载特定实验
            data_fusion: 是否融合辅助传感器数据
            fusion_method: 融合方法 (concat, film, weighted_sum, multiply, attention)
            downsample: 是否对数据进行下采样
            downsample_interval: 下采样时间间隔（秒）
            smooth_config: 平滑配置字典
            
        Returns:
            (all_features, all_labels, all_timestamps, metadata_list, experiment_indices)
            - all_features: 所有实验的特征，shape (total_samples, n_features)
            - all_labels: 所有实验的标签，shape (total_samples,)
            - all_timestamps: 所有实验的时间戳，shape (total_samples,)
            - metadata_list: 每个实验的元数据列表
            - experiment_indices: 每个样本所属的实验索引，shape (total_samples,)
        """
        print(f"\n{'='*70}")
        print(f"加载用户 {user_name} 的所有实验数据")
        if data_fusion:
            stage_str = '数据层(Early)' if fusion_stage == 'early' else '特征层(Late)'
            print(f"✓ 启用辅助传感器数据融合（阶段: {stage_str}, 方法: {fusion_method}）")
        if downsample:
            print(f"✓ 启用数据下采样（间隔: {downsample_interval}秒）")
        if smooth_config and smooth_config.get('smooth_data', False):
            print("✓ 启用数据平滑处理")
        print(f"{'='*70}")
        
        all_features_list = []
        all_labels_list = []
        all_timestamps_list = []
        metadata_list = []
        experiment_indices_list = []
        
        user_dir = self.dataset_root / user_name
        
        if not user_dir.exists():
            raise FileNotFoundError(f"用户目录不存在: {user_dir}")
        
        # 获取所有实验目录
        experiment_dirs = sorted([d for d in user_dir.iterdir() if d.is_dir()])
        
        if experiment_filter:
            experiment_dirs = [d for d in experiment_dirs if d.name in experiment_filter]
        
        print(f"找到 {len(experiment_dirs)} 个实验目录")
        
        for exp_idx, exp_dir in enumerate(experiment_dirs):
            result = self.load_user_experiment(
                exp_dir,
                interpolation_method,
                data_fusion=data_fusion,
                fusion_method=fusion_method,
                fusion_stage=fusion_stage,  # 传递fusion_stage参数
                downsample=downsample,
                downsample_interval=downsample_interval,
                smooth_config=smooth_config,
                icm_mode=icm_mode,
                fusion_config=fusion_config,  # 传递fusion_config参数
                user_name=user_name
            )
            
            if result is not None:
                features, labels, timestamps, metadata = result
                all_features_list.append(features)
                all_labels_list.append(labels)
                all_timestamps_list.append(timestamps)
                metadata_list.append(metadata)
        
        if not all_features_list:
            raise ValueError(f"未能加载用户 {user_name} 的任何实验数据")
        
        # 合并所有实验（使用float32减少内存）
        # 先转换各个实验为float32再合并
        all_features_list = [f.astype(np.float32) if f.dtype != np.float32 else f for f in all_features_list]
        all_features = np.vstack(all_features_list)
        all_labels = np.concatenate(all_labels_list)
        all_timestamps = np.concatenate(all_timestamps_list)
        
        # 创建实验分组索引：记录每个样本属于哪个实验
        experiment_indices = []
        for exp_idx, features in enumerate(all_features_list):
            experiment_indices.extend([exp_idx] * len(features))
        experiment_indices = np.array(experiment_indices)
        
        print(f"\n{'='*70}")
        print(f"✓ 成功加载 {len(metadata_list)} 个实验")
        print(f"  总样本数: {len(all_features)}")
        print(f"  特征维度: {all_features.shape[1]}")
        if fusion_stage == 'late' and data_fusion:
            n_spectrum = metadata_list[0].get('n_spectrum_features', 1001)
            n_aux = metadata_list[0].get('n_aux_features', 0)
            print(f"  Late Fusion: spectrum={n_spectrum}, aux={n_aux}")
        print(f"  血糖范围: {all_labels.min():.2f} - {all_labels.max():.2f} mmol/L")
        print(f"  血糖均值: {all_labels.mean():.2f} ± {all_labels.std():.2f} mmol/L")
        print(f"{'='*70}\n")
        
        return all_features, all_labels, all_timestamps, metadata_list, experiment_indices
    
    def normalize_features(
        self,
        features: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
        归一化特征 - 使用全局统计量
        
        新方法：使用训练集中所有时间步、所有频点的能量的全局均值和标准差
        对训练集、验证集、测试集都使用相同的归一化参数
        
        Args:
            features: 原始特征
                - instant模式: shape (n_samples, n_features)
                - window模式: shape (n_samples, window_size, n_features)
            scaler: 已有的 scaler（用于验证集和测试集）
            fit: 是否 fit scaler（训练集为 True，验证集和测试集为 False）
            
        Returns:
            (normalized_features, scaler)
        """
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class GlobalSpectrumNormalizer(BaseEstimator, TransformerMixin):
            """
            全局归一化器：使用所有训练数据的全局均值和标准差
            """
            def __init__(self):
                self.mean_ = None
                self.std_ = None
            
            def fit(self, X, y=None):
                """
                计算全局均值和标准差 - 内存高效版本（批次增量计算）
                X可以是2D (instant模式) 或3D (window模式)
                """
                data_size_gb = X.nbytes / 1e9
                print(f"    计算归一化统计量（数组: {data_size_gb:.2f} GB）...")
                
                # 小数据直接计算
                if data_size_gb < 2.0:
                    self.mean_ = float(np.mean(X))
                    self.std_ = float(np.std(X))
                else:
                    # 大数据使用批次增量计算避免内存溢出
                    print(f"    使用批次增量计算（数据过大: {data_size_gb:.2f} GB）...")
                    n_samples = X.shape[0]
                    batch_size = max(1, n_samples // 20)  # 分20批
                    
                    # 两遍算法：第一遍计算均值，第二遍计算方差
                    # 比单遍 Welford 更快（向量化操作）
                    
                    # 第一遍：计算均值
                    total_sum = 0.0
                    total_count = 0
                    for i in range(0, n_samples, batch_size):
                        end_idx = min(i + batch_size, n_samples)
                        batch = X[i:end_idx]
                        total_sum += np.sum(batch, dtype=np.float64)
                        total_count += batch.size
                    
                    self.mean_ = total_sum / total_count
                    print(f"      均值计算完成: {self.mean_:.4f}")
                    
                    # 第二遍：计算方差
                    total_sq_diff = 0.0
                    for i in range(0, n_samples, batch_size):
                        end_idx = min(i + batch_size, n_samples)
                        batch = X[i:end_idx].astype(np.float64)
                        diff = batch - self.mean_
                        total_sq_diff += np.sum(diff * diff)
                        del diff, batch  # 及时释放内存
                    
                    self.std_ = np.sqrt(total_sq_diff / total_count)
                    print(f"      标准差计算完成: {self.std_:.4f}")
                
                # 避免除零
                if self.std_ < 1e-8:
                    self.std_ = 1.0
                
                print(f"    全局归一化统计量: 均值={self.mean_:.2f}, 标准差={self.std_:.2f}")
                return self
            
            def transform(self, X):
                """
                使用全局统计量进行归一化 - 内存高效版本
                """
                if self.mean_ is None or self.std_ is None:
                    raise ValueError("必须先调用 fit() 方法")
                
                # 小数据直接处理
                if X.nbytes < 2e9:  # <2GB
                    if X.dtype != np.float32:
                        X = X.astype(np.float32)
                    normalized = (X - np.float32(self.mean_)) / np.float32(self.std_)
                    return normalized
                
                # 大数据分批处理避免内存溢出
                print(f"    分批归一化大数据 ({X.nbytes/1e9:.2f} GB)...")
                batch_size = max(1, X.shape[0] // 20)  # 分20批
                
                # 先转换类型
                if X.dtype != np.float32:
                    X = X.astype(np.float32)
                
                # 原地归一化避免额外分配
                for i in range(0, X.shape[0], batch_size):
                    end_idx = min(i + batch_size, X.shape[0])
                    X[i:end_idx] = (X[i:end_idx] - np.float32(self.mean_)) / np.float32(self.std_)
                
                return X
            
            def fit_transform(self, X, y=None):
                return self.fit(X, y).transform(X)
        
        # 创建或使用已有的scaler
        if scaler is None:
            scaler = GlobalSpectrumNormalizer()
        
        if fit:
            # 训练集：fit + transform
            normalized = scaler.fit_transform(features)
        else:
            # 验证集/测试集：只transform（使用训练集的统计量）
            normalized = scaler.transform(features)
        
        return normalized, scaler


# 示例使用
if __name__ == "__main__":
    # 创建加载器
    loader = GlucoseDatasetLoader(dataset_root="./Dataset")
    
    # 加载 Tao 用户的所有实验数据（自动从目录名提取时间）
    features, labels, metadata = loader.load_user_all_experiments(
        user_name="Tao",
        interpolation_method='linear'
    )
    
    # 归一化特征
    features_normalized, scaler = loader.normalize_features(features)
    
    print("数据加载完成！")
    print(f"归一化后特征 shape: {features_normalized.shape}")
    print(f"标签 shape: {labels.shape}")
    print(f"\n实验详情:")
    for i, meta in enumerate(metadata, 1):
        print(f"  {i}. {meta['experiment_dir']}: {meta['n_samples']} 样本, "
              f"血糖 {meta['glucose_min']:.2f}-{meta['glucose_max']:.2f} mmol/L")

"""
数据处理工具函数
包含数据预处理、增强、归一化等功能
整合了从 Dataset 加载真实数据的功能

改进：
- 自动从目录名提取起始时间，无需手动配置
- 使用 power_0 到 power_1000 作为特征（1001维）
- 为每个频谱时间戳插值计算血糖标签
- 支持 DB 格式数据源（SQLite数据库）
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import gc

# 导入数据加载模块
from .data_loader import GlucoseDatasetLoader
from .db_loader import DBGlucoseDatasetLoader


def _fmt_nbytes(nbytes):
    if nbytes >= 1024 ** 3:
        return f"{nbytes / (1024 ** 3):.2f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / (1024 ** 2):.2f} MB"
    return f"{nbytes / 1024:.2f} KB"


class GlucoseDataset(Dataset):
    """
    Glucose Sensing 数据集类
    支持两种模式:
    1. instant模式: data shape (n_samples, n_features), 当前spectrum预测当前血糖
    2. window模式: data shape (n_samples, window_size, n_features), 历史窗口预测当前血糖
    """
    def __init__(self, data, labels, transform=None, mode='instant'):
        """
        Args:
            data: numpy array
                - instant模式: shape (n_samples, n_features)
                - window模式: shape (n_samples, window_size, n_features)
            labels: numpy array, shape (n_samples,)
            transform: 数据变换函数
            mode: 'instant' 或 'window'
        """
        self.mode = mode
        data_np = np.asarray(data, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.float32)

        # 使用from_numpy避免不必要拷贝（float32时基本零拷贝）
        self.data = torch.from_numpy(data_np)
        self.labels = torch.from_numpy(labels_np)  # 回归任务，使用 float32
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class LazyWindowGlucoseDataset(Dataset):
    """
    按需窗口数据集：不预先展开3D窗口，__getitem__时动态切片历史窗口。
    仅支持drop模式（当前样本索引必须满足有window_size个历史样本）。
    """
    def __init__(self, base_features, current_indices, labels, window_size, transform=None):
        self.base_features = np.asarray(base_features, dtype=np.float32)
        self.current_indices = np.asarray(current_indices, dtype=np.int64)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.window_size = int(window_size)
        self.transform = transform

        if len(self.current_indices) != len(self.labels):
            raise ValueError(
                f"LazyWindowGlucoseDataset长度不一致: indices={len(self.current_indices)}, labels={len(self.labels)}"
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size必须为正整数，当前为 {self.window_size}")

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        current_idx = int(self.current_indices[idx])
        start_idx = current_idx - self.window_size
        if start_idx < 0:
            raise IndexError(
                f"当前索引 {current_idx} 历史窗口不足 window_size={self.window_size}"
            )

        window_np = self.base_features[start_idx:current_idx]
        sample = torch.from_numpy(window_np)
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class LazyWindowLateFusionDataset(Dataset):
    """
    按需窗口 + Late Fusion 数据集。
    不预展开3D窗口，在 __getitem__ 时动态切片并拆分为 spectrum/aux。
    仅支持drop模式（current_index 之前必须有 window_size 个历史样本）。
    """
    def __init__(self, base_features, current_indices, labels, window_size, n_spectrum_features, transform=None):
        self.base_features = np.asarray(base_features, dtype=np.float32)
        self.current_indices = np.asarray(current_indices, dtype=np.int64)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.window_size = int(window_size)
        self.n_spectrum = int(n_spectrum_features)
        self.transform = transform

        if len(self.current_indices) != len(self.labels):
            raise ValueError(
                f"LazyWindowLateFusionDataset长度不一致: indices={len(self.current_indices)}, labels={len(self.labels)}"
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size必须为正整数，当前为 {self.window_size}")
        if self.n_spectrum <= 0 or self.n_spectrum > self.base_features.shape[1]:
            raise ValueError(
                f"n_spectrum_features非法: {self.n_spectrum}, 总特征维度: {self.base_features.shape[1]}"
            )

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        current_idx = int(self.current_indices[idx])
        start_idx = current_idx - self.window_size
        if start_idx < 0:
            raise IndexError(
                f"当前索引 {current_idx} 历史窗口不足 window_size={self.window_size}"
            )

        window_np = self.base_features[start_idx:current_idx]
        spectrum = torch.from_numpy(window_np[:, :self.n_spectrum])
        aux = torch.from_numpy(window_np[:, self.n_spectrum:])
        label = self.labels[idx]

        if self.transform:
            spectrum = self.transform(spectrum)

        return spectrum, aux, label


class LateFusionDataset(Dataset):
    """
    Late Fusion 数据集类
    返回分离的spectrum和aux数据，供Late Fusion模型使用
    
    支持两种模式:
    1. instant模式: spectrum (n_samples, n_spectrum), aux (n_samples, n_aux)
    2. window模式: spectrum (n_samples, window, n_spectrum), aux (n_samples, window, n_aux)
    """
    def __init__(self, data, labels, n_spectrum_features, mode='instant', transform=None):
        """
        Args:
            data: numpy array, 拼接后的特征 [spectrum | aux]
                - instant: (n_samples, n_spectrum + n_aux)
                - window: (n_samples, window_size, n_spectrum + n_aux)
            labels: numpy array, shape (n_samples,)
            n_spectrum_features: spectrum特征的维度(1001)
            mode: 'instant' 或 'window'
            transform: 可选的数据变换
        """
        self.mode = mode
        self.n_spectrum = n_spectrum_features
        self.transform = transform
        
        # 分离spectrum和aux
        data_np = np.asarray(data, dtype=np.float32)

        if mode == 'window':
            # Window模式: (batch, window, features)
            self.spectrum = torch.from_numpy(data_np[:, :, :n_spectrum_features])
            self.aux = torch.from_numpy(data_np[:, :, n_spectrum_features:])
        else:
            # Instant模式: (batch, features)
            self.spectrum = torch.from_numpy(data_np[:, :n_spectrum_features])
            self.aux = torch.from_numpy(data_np[:, n_spectrum_features:])
        
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        
        # 记录aux维度
        self.n_aux = self.aux.shape[-1]
    
    def __len__(self):
        return len(self.spectrum)
    
    def __getitem__(self, idx):
        spectrum = self.spectrum[idx]
        aux = self.aux[idx]
        label = self.labels[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, aux, label
    
    def get_aux_dim(self):
        """返回aux特征维度"""
        return self.n_aux
    
    def get_spectrum_dim(self):
        """返回spectrum特征维度"""
        return self.n_spectrum


class AutoRegressiveDataset(Dataset):
    """
    自回归模式数据集
    返回: (spectrum, aux(可选), glucose_history, glucose_target)
    
    glucose_history: 历史血糖值序列，用作模型输入
    glucose_target: 当前时刻的真实血糖值，用于计算loss
    
    支持两种对齐模式:
    1. aligned=True: history是过去ar_history个时刻的血糖，不包含当前(真正未知)
    2. aligned=False: history包含当前时刻血糖，但预测目标仍是当前血糖
       (用于训练时增强学习历史趋势的能力)
    """
    def __init__(self, data, labels, glucose_history, n_spectrum_features=None, 
                 mode='instant', has_aux=False, transform=None):
        """
        Args:
            data: numpy array - spectrum数据，或者[spectrum | aux]拼接数据
                - instant: (n_samples, n_features)
                - window: (n_samples, window_size, n_features)
            labels: numpy array, shape (n_samples,) - 目标血糖值
            glucose_history: numpy array, shape (n_samples, ar_history) - 历史血糖序列
            n_spectrum_features: spectrum特征维度，若None则不分离aux
            mode: 'instant' 或 'window'
            has_aux: 是否包含aux数据（需配合n_spectrum_features使用）
            transform: 可选的数据变换
        """
        self.mode = mode
        self.n_spectrum = n_spectrum_features
        self.has_aux = has_aux and n_spectrum_features is not None
        self.transform = transform
        
        # 转换数据为tensor
        data_np = np.asarray(data, dtype=np.float32)
        history_np = np.asarray(glucose_history, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.float32)

        if mode == 'window':
            if self.has_aux:
                self.spectrum = torch.from_numpy(data_np[:, :, :n_spectrum_features])
                self.aux = torch.from_numpy(data_np[:, :, n_spectrum_features:])
            else:
                self.spectrum = torch.from_numpy(data_np)
                self.aux = None
        else:
            if self.has_aux:
                self.spectrum = torch.from_numpy(data_np[:, :n_spectrum_features])
                self.aux = torch.from_numpy(data_np[:, n_spectrum_features:])
            else:
                self.spectrum = torch.from_numpy(data_np)
                self.aux = None
        
        self.glucose_history = torch.from_numpy(history_np)  # (n_samples, ar_history)
        self.labels = torch.from_numpy(labels_np)  # (n_samples,)
        
        # 记录维度信息
        self.ar_history_len = glucose_history.shape[1]
        self.n_aux = self.aux.shape[-1] if self.aux is not None else 0
    
    def __len__(self):
        return len(self.spectrum)
    
    def __getitem__(self, idx):
        spectrum = self.spectrum[idx]
        glucose_hist = self.glucose_history[idx]  # (ar_history,)
        label = self.labels[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        if self.has_aux:
            aux = self.aux[idx]
            return spectrum, aux, glucose_hist, label
        else:
            return spectrum, glucose_hist, label
    
    def get_spectrum_dim(self):
        """返回spectrum特征维度"""
        if self.mode == 'window':
            return self.spectrum.shape[-1]
        return self.spectrum.shape[1] if self.n_spectrum is None else self.n_spectrum
    
    def get_aux_dim(self):
        """返回aux特征维度"""
        return self.n_aux
    
    def get_ar_history_len(self):
        """返回历史血糖序列长度"""
        return self.ar_history_len


class DataAugmentation:
    """
    数据增强类
    """
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    @staticmethod
    def scale(data, scale_range=(0.9, 1.1)):
        """随机缩放"""
        scale_factor = torch.FloatTensor(1).uniform_(*scale_range)
        return data * scale_factor
    
    @staticmethod
    def shift(data, shift_range=(-0.1, 0.1)):
        """随机偏移"""
        shift = torch.FloatTensor(1).uniform_(*shift_range)
        return data + shift


def normalize_data(data, method='standard'):
    """
    数据归一化
    
    Args:
        data: numpy array
        method: 'standard', 'minmax', 'robust'
    
    Returns:
        normalized_data, scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


def split_data_random(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     random_state=42):
    """
    随机划分数据集（默认策略）
    
    Args:
        data: numpy array
        labels: numpy array
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为1"
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    # 先划分出测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_state, shuffle=True
    )
    
    # 再从剩余索引中划分训练集和验证集
    if val_ratio > 0:
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_adjusted,
            random_state=random_state, shuffle=True
        )
    else:
        # 没有验证集，所有train_val_indices都作为训练集
        train_indices = train_val_indices
        val_indices = np.array([], dtype=int)
    
    # 使用索引获取数据
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices] if len(val_indices) > 0 else np.array([]).reshape(0, data.shape[1])
    val_labels = labels[val_indices] if len(val_indices) > 0 else np.array([])
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_temporal(data, labels, metadata_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按时序划分数据集（时间顺序）
    前70%时间的数据作为训练集，中15%作为验证集，后15%作为测试集
    
    Args:
        data: numpy array
        labels: numpy array
        metadata_list: list of dicts with 'start_time' info
        train_ratio: 训练集比例
        val_ratio: 验证集比例  
        test_ratio: 测试集比例
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为1"
    
    n_samples = len(data)
    
    # 计算划分点
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 生成索引
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n_samples)
    
    # 按顺序划分（不打乱）
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    val_data = data[val_indices]
    val_labels = labels[val_indices]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_stratified(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                          random_state=42, n_bins=5):
    """
    分层划分数据集（保持各血糖区间比例一致）
    
    Args:
        data: numpy array
        labels: numpy array
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        n_bins: 分层的区间数
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为1"
    
    # 将连续的血糖值离散化为分类标签
    bins = np.linspace(labels.min(), labels.max(), n_bins + 1)
    stratify_labels = np.digitize(labels, bins[:-1]) - 1
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    # 先划分出测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_state,
        stratify=stratify_labels, shuffle=True
    )
    
    # 再从剩余索引中划分训练集和验证集
    train_val_stratify = stratify_labels[train_val_indices]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio_adjusted,
        random_state=random_state, stratify=train_val_stratify, shuffle=True
    )
    
    # 使用索引获取数据
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices]
    val_labels = labels[val_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_by_experiment(data, labels, metadata_list, train_ratio=0.7, val_ratio=0.15, 
                            test_ratio=0.15, random_state=42):
    """
    按实验分组划分数据集（整个实验作为一个单位）
    避免同一实验的数据分散到不同集合中
    
    Args:
        data: numpy array
        labels: numpy array
        metadata_list: list of dicts with experiment info
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为1"
    
    # 提取每个样本所属的实验ID
    experiment_ids = []
    for meta in metadata_list:
        n_samples = meta['n_samples']
        exp_name = meta['experiment_name']
        experiment_ids.extend([exp_name] * n_samples)
    
    experiment_ids = np.array(experiment_ids)
    unique_experiments = np.unique(experiment_ids)
    
    # 随机打乱实验顺序
    np.random.seed(random_state)
    shuffled_experiments = np.random.permutation(unique_experiments)
    
    # 计算每个集合应该包含的实验数量
    n_experiments = len(unique_experiments)
    n_train = max(1, int(n_experiments * train_ratio))
    n_val = max(1, int(n_experiments * val_ratio))
    
    train_experiments = shuffled_experiments[:n_train]
    val_experiments = shuffled_experiments[n_train:n_train+n_val]
    test_experiments = shuffled_experiments[n_train+n_val:]
    
    # 根据实验分配样本
    train_mask = np.isin(experiment_ids, train_experiments)
    val_mask = np.isin(experiment_ids, val_experiments)
    test_mask = np.isin(experiment_ids, test_experiments)
    
    # 获取索引
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    val_data = data[val_indices]
    val_labels = labels[val_indices]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_alternating(data, labels, n_splits=10, use_val_set=True, metadata_list=None, experiment_indices=None):
    """
    交叉时序划分：将时序数据n等分，奇数段作为训练集，偶数段作为测试集
    
    改进版本：如果提供 experiment_indices，则对每个实验分别进行分段，
    这样可以更好地评估模型在同一受试者不同时间段的泛化能力
    
    Args:
        data: numpy array, shape (n_samples, n_features)
        labels: numpy array, shape (n_samples,)
        n_splits: int, 每个实验的分段数量（默认10）
        use_val_set: bool, 是否使用验证集（如果True，偶数段的一半作为验证集）
        metadata_list: list of dicts, 实验元数据（用于打印信息）
        experiment_indices: numpy array, 每个样本所属的实验ID（如果提供，则对每个实验分别分段）
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels,
        train_indices, val_indices, test_indices
        
    示例：
        n_splits=10时，每个实验分为10段
        - 训练集：每个实验的段1, 3, 5, 7, 9（奇数段）
        - 测试集：每个实验的段2, 4, 6, 8, 10（偶数段）
        - 如果use_val_set=True，段2, 6, 10作为验证集，段4, 8作为测试集
    """
    n_samples = len(data)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 如果提供了experiment_indices，对每个实验分别进行alternating分段
    if experiment_indices is not None:
        print(f"  交叉时序划分: 对每个实验分别分为{n_splits}段")
        
        # 获取唯一的实验ID（保持顺序）
        unique_exp_ids = []
        seen = set()
        for exp_id in experiment_indices:
            if exp_id not in seen:
                unique_exp_ids.append(exp_id)
                seen.add(exp_id)
        
        # 创建实验名称映射（如果有metadata_list）
        exp_name_map = {}
        if metadata_list is not None:
            for idx, meta in enumerate(metadata_list):
                exp_name_map[idx] = meta['experiment_name']
        
        # 对每个实验分别分段
        for exp_id in unique_exp_ids:
            # 获取当前实验的所有样本索引
            exp_mask = (experiment_indices == exp_id)
            exp_indices = np.where(exp_mask)[0]
            exp_n_samples = len(exp_indices)
            
            segment_size = exp_n_samples // n_splits
            
            exp_name = exp_name_map.get(exp_id, f"Exp_{exp_id}")
            
            if segment_size < 1:
                print(f"    ⚠️  实验 {exp_name} 数据量({exp_n_samples})太小，无法分成{n_splits}段，跳过")
                continue
            
            print(f"    - {exp_name}: {exp_n_samples}个样本 → 每段约{segment_size}个样本")
            
            # 对当前实验进行分段
            for i in range(n_splits):
                seg_start_local = i * segment_size
                # 最后一段包含所有剩余样本
                seg_end_local = (i + 1) * segment_size if i < n_splits - 1 else exp_n_samples
                
                # 获取当前段的实际索引
                segment_indices = exp_indices[seg_start_local:seg_end_local].tolist()
                
                if (i + 1) % 2 == 1:  # 奇数段（1, 3, 5, ...）
                    train_indices.extend(segment_indices)
                else:  # 偶数段（2, 4, 6, ...）
                    if use_val_set:
                        # 偶数段中，每3个中有1个作为验证集，2个作为测试集
                        even_segment_num = (i + 1) // 2  # 1, 2, 3, 4, 5...
                        if even_segment_num % 3 == 1:  # 第1, 4, 7...个偶数段作为验证集
                            val_indices.extend(segment_indices)
                        else:  # 其他偶数段作为测试集
                            test_indices.extend(segment_indices)
                    else:
                        test_indices.extend(segment_indices)
    
    else:
        # 如果没有metadata_list，使用旧版本的全局分段方式
        segment_size = n_samples // n_splits
        
        if segment_size < 1:
            raise ValueError(f"数据量({n_samples})太小，无法分成{n_splits}段")
        
        print(f"  交叉时序划分: 将{n_samples}个样本分为{n_splits}段，每段约{segment_size}个样本")
        
        for i in range(n_splits):
            start_idx = i * segment_size
            # 最后一段包含所有剩余样本
            end_idx = (i + 1) * segment_size if i < n_splits - 1 else n_samples
            
            segment_indices = list(range(start_idx, end_idx))
            
            if (i + 1) % 2 == 1:  # 奇数段（1, 3, 5, ...）
                train_indices.extend(segment_indices)
            else:  # 偶数段（2, 4, 6, ...）
                if use_val_set:
                    # 偶数段中，每3个中有1个作为验证集，2个作为测试集
                    even_segment_num = (i + 1) // 2  # 1, 2, 3, 4, 5...
                    if even_segment_num % 3 == 1:  # 第1, 4, 7...个偶数段作为验证集
                        val_indices.extend(segment_indices)
                    else:  # 其他偶数段作为测试集
                        test_indices.extend(segment_indices)
                else:
                    test_indices.extend(segment_indices)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    val_data = data[val_indices] if len(val_indices) > 0 else data[test_indices[:0]]
    val_labels = labels[val_indices] if len(val_indices) > 0 else labels[test_indices[:0]]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # 输出统计信息
    total_samples = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"\n  划分结果统计:")
    print(f"  奇数段(训练集): {len(train_indices)}个样本 ({len(train_indices)/total_samples*100:.1f}%)")
    if use_val_set and len(val_indices) > 0:
        print(f"  偶数段(验证集): {len(val_indices)}个样本 ({len(val_indices)/total_samples*100:.1f}%)")
        print(f"  偶数段(测试集): {len(test_indices)}个样本 ({len(test_indices)/total_samples*100:.1f}%)")
    else:
        print(f"  偶数段(测试集): {len(test_indices)}个样本 ({len(test_indices)/total_samples*100:.1f}%)")
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_hybrid_alternating(data, labels, n_splits=10, use_val_set=True, metadata_list=None, experiment_indices=None):
    """
    混合交叉划分：前一半数据全部用于训练，后一半数据使用alternating划分
    
    策略说明：
    - 数据分为n段
    - 前 n//2 段：全部作为训练集
    - 后 n//2 段：奇数段训练，偶数段测试
    
    优势：
    - 模型在前期数据上充分训练
    - 在后期数据上评估时间泛化能力
    - 适合评估模型对时间漂移的鲁棒性
    
    Args:
        data: numpy array, shape (n_samples, n_features)
        labels: numpy array, shape (n_samples,)
        n_splits: int, 总分段数（必须是偶数，默认10）
        use_val_set: bool, 是否在后半部分使用验证集
        metadata_list: list of dicts, 实验元数据
        experiment_indices: numpy array, 每个样本所属的实验ID
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels,
        train_indices, val_indices, test_indices
        
    示例：
        n_splits=10时：前5段训练，后5段alternating
        n_splits=9时：前4段训练，后5段alternating
        n_splits=11时：前5段训练，后6段alternating
    """
    n_samples = len(data)
    half_splits = n_splits // 2  # 前半部分段数
    second_half_splits = n_splits - half_splits  # 后半部分段数
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    if experiment_indices is not None:
        print(f"  混合交叉划分: 对每个实验分{n_splits}段，前{half_splits}段训练，后{second_half_splits}段alternating")
        
        # 获取唯一的实验ID
        unique_exp_ids = []
        seen = set()
        for exp_id in experiment_indices:
            if exp_id not in seen:
                unique_exp_ids.append(exp_id)
                seen.add(exp_id)
        
        # 创建实验名称映射
        exp_name_map = {}
        if metadata_list is not None:
            for idx, meta in enumerate(metadata_list):
                exp_name_map[idx] = meta['experiment_name']
        
        # 对每个实验分别处理
        for exp_id in unique_exp_ids:
            exp_mask = (experiment_indices == exp_id)
            exp_indices = np.where(exp_mask)[0]
            exp_n_samples = len(exp_indices)
            
            segment_size = exp_n_samples // n_splits
            exp_name = exp_name_map.get(exp_id, f"Exp_{exp_id}")
            
            if segment_size < 1:
                print(f"    ⚠️  实验 {exp_name} 数据量({exp_n_samples})太小，无法分成{n_splits}段，跳过")
                continue
            
            print(f"    - {exp_name}: {exp_n_samples}个样本 → 前{half_splits}段训练, 后{second_half_splits}段alternating")
            
            # 分段处理
            for i in range(n_splits):
                seg_start_local = i * segment_size
                seg_end_local = (i + 1) * segment_size if i < n_splits - 1 else exp_n_samples
                segment_indices = exp_indices[seg_start_local:seg_end_local].tolist()
                
                if i < half_splits:
                    # 前一协：全部用于训练
                    train_indices.extend(segment_indices)
                else:
                    # 后一协：alternating划分
                    segment_in_second_half = i - half_splits  # 0-based index in second half
                    if segment_in_second_half % 2 == 0:  # 偶数位置（第1个在后半部分）-> 训练
                        train_indices.extend(segment_indices)
                    else:  # 奇数位置 -> 测试或验证
                        if use_val_set:
                            # 分配到验证集或测试集
                            test_segment_num = (segment_in_second_half + 1) // 2
                            if test_segment_num % 3 == 1:
                                val_indices.extend(segment_indices)
                            else:
                                test_indices.extend(segment_indices)
                        else:
                            test_indices.extend(segment_indices)
    else:
        # 全局分段
        segment_size = n_samples // n_splits
        
        if segment_size < 1:
            raise ValueError(f"数据量({n_samples})太小，无法分成{n_splits}段")
        
        print(f"  混合交叉划分: {n_samples}个样本分{n_splits}段, 前{half_splits}段训练, 后{second_half_splits}段alternating")
        
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_splits - 1 else n_samples
            segment_indices = list(range(start_idx, end_idx))
            
            if i < half_splits:
                # 前一协
                train_indices.extend(segment_indices)
            else:
                # 后一协
                segment_in_second_half = i - half_splits
                if segment_in_second_half % 2 == 0:
                    train_indices.extend(segment_indices)
                else:
                    if use_val_set:
                        test_segment_num = (segment_in_second_half + 1) // 2
                        if test_segment_num % 3 == 1:
                            val_indices.extend(segment_indices)
                        else:
                            test_indices.extend(segment_indices)
                    else:
                        test_indices.extend(segment_indices)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    val_data = data[val_indices] if len(val_indices) > 0 else data[test_indices[:0]]
    val_labels = labels[val_indices] if len(val_indices) > 0 else labels[test_indices[:0]]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # 输出统计信息
    total_samples = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"\n  划分结果统计:")
    print(f"  训练集: {len(train_indices)}个样本 ({len(train_indices)/total_samples*100:.1f}%) - 前{half_splits}段+后半alternating奇数位")
    if use_val_set and len(val_indices) > 0:
        print(f"  验证集: {len(val_indices)}个样本 ({len(val_indices)/total_samples*100:.1f}%) - 后半alternating偶数位")
        print(f"  测试集: {len(test_indices)}个样本 ({len(test_indices)/total_samples*100:.1f}%) - 后半alternating偶数位")
    else:
        print(f"  测试集: {len(test_indices)}个样本 ({len(test_indices)/total_samples*100:.1f}%) - 后半alternating偶数位")
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def _extract_date_info_from_metadata(meta):
    """从实验metadata中提取日期标签和排序键。"""
    start_time = str(meta.get('start_time', '')).strip()
    experiment_name = str(meta.get('experiment_name', '')).strip()

    # 1) YYYY-MM-DD HH:MM:SS -> date_label=YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})', start_time)
    if m:
        y, mm, dd = map(int, m.groups())
        return f"{y:04d}-{mm:02d}-{dd:02d}", (y, mm, dd)

    # 2) MM.DD HH:MM 或 MM-DD HH:MM -> date_label=MM.DD
    m = re.match(r'^(\d{2})[.\-/](\d{2})\b', start_time)
    if m:
        mm, dd = map(int, m.groups())
        return f"{mm:02d}.{dd:02d}", (2000, mm, dd)

    # 3) DB实验名: 260121_1703_Tao -> 2026-01-21
    m = re.match(r'^(\d{2})(\d{2})(\d{2})_', experiment_name)
    if m:
        yy, mm, dd = map(int, m.groups())
        year = 2000 + yy
        return f"{year:04d}-{mm:02d}-{dd:02d}", (year, mm, dd)

    # 4) BIN实验名: 11201828_Tao -> 11.20
    m = re.match(r'^(\d{2})(\d{2})\d{4}_', experiment_name)
    if m:
        mm, dd = map(int, m.groups())
        return f"{mm:02d}.{dd:02d}", (2000, mm, dd)

    raise ValueError(
        f"无法从metadata提取日期信息: start_time='{start_time}', experiment_name='{experiment_name}'"
    )


def split_data_by_date(data, labels, metadata_list, experiment_indices,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按日期划分数据集（日期级分组，不拆分同一天的数据）。

    说明:
    - 先按 experiment -> date 聚合，再按日期时间顺序切分 train/val/test。
    - 同一天的所有样本会落在同一个集合中，避免日期泄漏。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    if metadata_list is None or experiment_indices is None:
        raise ValueError("按日期划分需要 metadata_list 和 experiment_indices 参数")

    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("空数据集无法划分")

    # 实验ID -> 日期
    exp_to_date = {}
    exp_to_sort_key = {}
    for exp_id, meta in enumerate(metadata_list):
        date_label, sort_key = _extract_date_info_from_metadata(meta)
        exp_to_date[exp_id] = date_label
        exp_to_sort_key[exp_id] = sort_key

    # 日期 -> 样本索引
    date_to_indices = {}
    date_to_sort_key = {}
    for sample_idx, exp_id in enumerate(experiment_indices):
        exp_id_int = int(exp_id)
        if exp_id_int not in exp_to_date:
            raise ValueError(f"experiment_indices中存在无效实验ID: {exp_id_int}")
        date_label = exp_to_date[exp_id_int]
        date_to_indices.setdefault(date_label, []).append(sample_idx)
        date_to_sort_key[date_label] = exp_to_sort_key[exp_id_int]

    sorted_dates = sorted(date_to_indices.keys(), key=lambda d: date_to_sort_key[d])
    date_counts = np.array([len(date_to_indices[d]) for d in sorted_dates], dtype=int)

    if len(sorted_dates) < 2:
        raise ValueError(
            f"按日期划分至少需要2个不同日期，当前仅有{len(sorted_dates)}个日期"
        )

    total = int(date_counts.sum())
    train_target = total * train_ratio
    val_target = total * (train_ratio + val_ratio)

    cum = np.cumsum(date_counts)

    # cut1: 训练集结束的日期位置（左闭右开）
    cut1 = int(np.searchsorted(cum, train_target, side='left') + 1)
    # cut2: 验证集结束的位置
    cut2 = int(np.searchsorted(cum, val_target, side='left') + 1)

    # 边界保护：在可行范围内保证集合非空
    max_cut1 = len(sorted_dates) - (1 if test_ratio > 0 else 0) - (1 if val_ratio > 0 else 0)
    max_cut1 = max(1, max_cut1)
    cut1 = max(1, min(cut1, max_cut1))

    if val_ratio > 0:
        min_cut2 = cut1 + 1
        max_cut2 = len(sorted_dates) - (1 if test_ratio > 0 else 0)
        max_cut2 = max(min_cut2, max_cut2)
        cut2 = max(min_cut2, min(cut2, max_cut2))
    else:
        cut2 = cut1

    train_dates = sorted_dates[:cut1]
    val_dates = sorted_dates[cut1:cut2] if val_ratio > 0 else []
    test_dates = sorted_dates[cut2:] if val_ratio > 0 else sorted_dates[cut1:]

    train_indices = np.array([i for d in train_dates for i in date_to_indices[d]], dtype=int)
    val_indices = np.array([i for d in val_dates for i in date_to_indices[d]], dtype=int)
    test_indices = np.array([i for d in test_dates for i in date_to_indices[d]], dtype=int)

    # 保证索引按时间顺序（原始顺序）
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices] if len(val_indices) > 0 else data[:0]
    val_labels = labels[val_indices] if len(val_indices) > 0 else labels[:0]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    print("  日期划分详情:")
    print(f"  - 训练日期: {train_dates}")
    if len(val_dates) > 0:
        print(f"  - 验证日期: {val_dates}")
    print(f"  - 测试日期: {test_dates}")

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_by_date_random(data, labels, metadata_list, experiment_indices,
                              train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                              random_state=42, train_days=None):
    """
    随机按日期划分数据集（日期级分组，不拆分同一天的数据）。

    与 split_data_by_date 的主要区别:
    - split_data_by_date: 按日期先后顺序划分
    - split_data_by_date_random: 在日期集合中随机抽取训练日期，再从剩余日期中划分验证/测试
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    if metadata_list is None or experiment_indices is None:
        raise ValueError("随机按日期划分需要 metadata_list 和 experiment_indices 参数")

    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("空数据集无法划分")

    exp_to_date = {}
    exp_to_sort_key = {}
    for exp_id, meta in enumerate(metadata_list):
        date_label, sort_key = _extract_date_info_from_metadata(meta)
        exp_to_date[exp_id] = date_label
        exp_to_sort_key[exp_id] = sort_key

    date_to_indices = {}
    date_to_sort_key = {}
    for sample_idx, exp_id in enumerate(experiment_indices):
        exp_id_int = int(exp_id)
        if exp_id_int not in exp_to_date:
            raise ValueError(f"experiment_indices中存在无效实验ID: {exp_id_int}")
        date_label = exp_to_date[exp_id_int]
        date_to_indices.setdefault(date_label, []).append(sample_idx)
        date_to_sort_key[date_label] = exp_to_sort_key[exp_id_int]

    sorted_dates = sorted(date_to_indices.keys(), key=lambda d: date_to_sort_key[d])
    n_dates = len(sorted_dates)
    if n_dates < 2:
        raise ValueError(f"随机按日期划分至少需要2个不同日期，当前仅有{n_dates}个日期")

    rng = np.random.default_rng(random_state)

    if train_days is None:
        train_days_count = int(round(n_dates * train_ratio))
    else:
        train_days_count = int(train_days)

    reserved_for_val = 1 if val_ratio > 0 else 0
    max_train_days = n_dates - 1 - reserved_for_val
    max_train_days = max(1, max_train_days)
    train_days_count = max(1, min(train_days_count, max_train_days))

    permuted_dates = [sorted_dates[i] for i in rng.permutation(n_dates)]
    train_dates = sorted(
        permuted_dates[:train_days_count],
        key=lambda d: date_to_sort_key[d]
    )

    remaining_dates = sorted(
        [d for d in sorted_dates if d not in set(train_dates)],
        key=lambda d: date_to_sort_key[d]
    )

    if len(remaining_dates) == 0:
        raise ValueError("随机按日期划分失败：训练日期覆盖了全部日期，无法构造测试集")

    if val_ratio > 0:
        # 在剩余日期中按比例切分验证/测试，保证两者尽量非空
        val_days_count = int(round(len(remaining_dates) * (val_ratio / (val_ratio + test_ratio))))
        val_days_count = max(1, min(val_days_count, len(remaining_dates) - 1))
        val_dates = remaining_dates[:val_days_count]
        test_dates = remaining_dates[val_days_count:]
    else:
        val_dates = []
        test_dates = remaining_dates

    train_indices = np.array([i for d in train_dates for i in date_to_indices[d]], dtype=int)
    val_indices = np.array([i for d in val_dates for i in date_to_indices[d]], dtype=int)
    test_indices = np.array([i for d in test_dates for i in date_to_indices[d]], dtype=int)

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices] if len(val_indices) > 0 else data[:0]
    val_labels = labels[val_indices] if len(val_indices) > 0 else labels[:0]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    print("  随机日期划分详情:")
    print(f"  - train_days={train_days_count}, random_state={random_state}")
    print(f"  - 训练日期: {train_dates}")
    if len(val_dates) > 0:
        print(f"  - 验证日期: {val_dates}")
    print(f"  - 测试日期: {test_dates}")

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_max_day_train_min_day_test(data, labels, metadata_list, experiment_indices):
    """
    简单对照划分：
    - 训练集使用样本数最多的一天
    - 测试集使用样本数最少的一天
    - 不使用验证集

    适用于快速检查“单日训练 -> 另一日测试”的最小实验。
    """
    if metadata_list is None or experiment_indices is None:
        raise ValueError("max_day_train_min_day_test 需要 metadata_list 和 experiment_indices 参数")

    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("空数据集无法划分")

    exp_to_date = {}
    exp_to_sort_key = {}
    for exp_id, meta in enumerate(metadata_list):
        date_label, sort_key = _extract_date_info_from_metadata(meta)
        exp_to_date[exp_id] = date_label
        exp_to_sort_key[exp_id] = sort_key

    date_to_indices = {}
    date_to_sort_key = {}
    for sample_idx, exp_id in enumerate(experiment_indices):
        exp_id_int = int(exp_id)
        if exp_id_int not in exp_to_date:
            raise ValueError(f"experiment_indices中存在无效实验ID: {exp_id_int}")
        date_label = exp_to_date[exp_id_int]
        date_to_indices.setdefault(date_label, []).append(sample_idx)
        date_to_sort_key[date_label] = exp_to_sort_key[exp_id_int]

    if len(date_to_indices) < 2:
        raise ValueError(f"max_day_train_min_day_test 至少需要2个不同日期，当前仅有{len(date_to_indices)}个日期")

    counts = {d: len(v) for d, v in date_to_indices.items()}
    sorted_dates = sorted(date_to_indices.keys(), key=lambda d: date_to_sort_key[d])

    max_count = max(counts.values())
    min_count = min(counts.values())

    max_dates = sorted([d for d in sorted_dates if counts[d] == max_count], key=lambda d: date_to_sort_key[d])
    min_dates = sorted([d for d in sorted_dates if counts[d] == min_count], key=lambda d: date_to_sort_key[d])

    train_date = max_dates[0]
    test_date = min_dates[0]

    # 若所有日期样本数相同，避免 train/test 落在同一天
    if train_date == test_date:
        if len(sorted_dates) < 2:
            raise ValueError("无法构造不同的训练/测试日期")
        train_date = sorted_dates[0]
        test_date = sorted_dates[-1]

    train_indices = np.array(date_to_indices[train_date], dtype=int)
    test_indices = np.array(date_to_indices[test_date], dtype=int)
    val_indices = np.array([], dtype=int)

    train_indices.sort()
    test_indices.sort()

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[:0]
    val_labels = labels[:0]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    print("  最大/最小日期划分详情:")
    print(f"  - 训练日期(最多样本): {train_date} (n={len(train_indices)})")
    print(f"  - 测试日期(最少样本): {test_date} (n={len(test_indices)})")

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data_by_db_file_temporal_80_20(data, labels, timestamps, db_file_indices, db_file_names=None):
    """
    按每个源 .db 文件内部时间顺序划分:
    - 每个 .db 的前80%时间戳样本进入训练集
    - 每个 .db 的后20%时间戳样本进入测试集
    - 不使用验证集
    """
    if timestamps is None or db_file_indices is None:
        raise ValueError("db_file_temporal_80_20 需要 timestamps 和 db_file_indices 参数")

    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("空数据集无法划分")
    if len(timestamps) != n_samples or len(db_file_indices) != n_samples:
        raise ValueError(
            "db_file_temporal_80_20 输入长度不一致: "
            f"data={n_samples}, timestamps={len(timestamps)}, db_file_indices={len(db_file_indices)}"
        )

    train_indices_parts = []
    test_indices_parts = []
    db_file_names = db_file_names or {}

    print("  DB文件时间顺序80/20划分详情:")
    for db_id in np.unique(db_file_indices):
        db_sample_indices = np.where(db_file_indices == db_id)[0]
        if len(db_sample_indices) < 2:
            raise ValueError(f"DB文件 {db_file_names.get(int(db_id), db_id)} 样本数不足2，无法80/20划分")

        sort_order = np.argsort(timestamps[db_sample_indices])
        sorted_indices = db_sample_indices[sort_order]
        cut = int(len(sorted_indices) * 0.8)
        cut = max(1, min(cut, len(sorted_indices) - 1))

        train_part = sorted_indices[:cut]
        test_part = sorted_indices[cut:]
        train_indices_parts.append(train_part)
        test_indices_parts.append(test_part)

        db_name = db_file_names.get(int(db_id), f"db_file_{int(db_id)}")
        print(f"    - {db_name}: total={len(sorted_indices)}, train={len(train_part)}, test={len(test_part)}")

    train_indices = np.concatenate(train_indices_parts).astype(int)
    test_indices = np.concatenate(test_indices_parts).astype(int)
    val_indices = np.array([], dtype=int)

    train_indices.sort()
    test_indices.sort()

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[:0]
    val_labels = labels[:0]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices


def split_data(data, labels, metadata_list=None, strategy='random', 
               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42,
               n_splits=10, return_indices=False, experiment_indices=None,
               date_train_days=None, timestamps=None, db_file_indices=None, db_file_names=None):
    """
    统一的数据划分接口，支持多种划分策略
    
    Args:
        data: numpy array, shape (n_samples, n_features)
        labels: numpy array, shape (n_samples,)
        metadata_list: list of dicts, 实验元数据（某些策略需要）
        strategy: str, 划分策略
            - 'random': 随机划分（默认）
            - 'temporal': 按时序划分
            - 'stratified': 分层划分（保持血糖分布）
            - 'experiment': 按实验分组划分
            - 'date': 按日期分组划分（整天不拆分）
            - 'date_random': 随机按日期分组划分（整天不拆分）
            - 'max_day_train_min_day_test': 最多样本日期训练，最少样本日期测试（单日训练快速对照）
            - 'alternating': 交叉时序划分（奇数段训练，偶数段测试）
            - 'multi_user_independent': 多用户各自独立划分后合并训练集
            - 'db_file_temporal_80_20': 每个DB文件内按时间前80%训练、后20%测试
            - 'tao_db_label_shuffle_debug': Tao_db专用debug策略（在load_and_preprocess_data中处理）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        n_splits: 交叉时序划分的分段数（仅用于alternating策略）
        return_indices: 是否同时返回索引（用于同步划分timestamps等）
        experiment_indices: numpy array, 每个样本所属的实验ID（用于alternating策略）
    
    Returns:
        如果return_indices=False:
            train_data, val_data, test_data, train_labels, val_labels, test_labels
        如果return_indices=True:
            train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices
    """
    print(f"\n使用划分策略: {strategy}")
    
    if strategy == 'alternating':
        use_val_set = val_ratio > 0
        result = split_data_alternating(data, labels, n_splits=n_splits, use_val_set=use_val_set, 
                                       metadata_list=metadata_list, experiment_indices=experiment_indices)
    
    elif strategy == 'hybrid_alternating':
        use_val_set = val_ratio > 0
        result = split_data_hybrid_alternating(data, labels, n_splits=n_splits, use_val_set=use_val_set,
                                               metadata_list=metadata_list, experiment_indices=experiment_indices)
    
    elif strategy == 'random':
        result = split_data_random(data, labels, train_ratio, val_ratio, test_ratio, random_state)
    
    elif strategy == 'temporal':
        if metadata_list is None:
            raise ValueError("时序划分需要 metadata_list 参数")
        result = split_data_temporal(data, labels, metadata_list, train_ratio, val_ratio, test_ratio)
    
    elif strategy == 'stratified':
        result = split_data_stratified(data, labels, train_ratio, val_ratio, test_ratio, random_state)
    
    elif strategy == 'experiment':
        if metadata_list is None:
            raise ValueError("按实验划分需要 metadata_list 参数")
        result = split_data_by_experiment(data, labels, metadata_list, train_ratio, val_ratio, test_ratio, random_state)

    elif strategy == 'date':
        if metadata_list is None or experiment_indices is None:
            raise ValueError("按日期划分需要 metadata_list 和 experiment_indices 参数")
        result = split_data_by_date(
            data, labels, metadata_list, experiment_indices,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    elif strategy == 'date_random':
        if metadata_list is None or experiment_indices is None:
            raise ValueError("随机按日期划分需要 metadata_list 和 experiment_indices 参数")
        result = split_data_by_date_random(
            data, labels, metadata_list, experiment_indices,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            train_days=date_train_days,
        )

    elif strategy == 'max_day_train_min_day_test':
        if metadata_list is None or experiment_indices is None:
            raise ValueError("max_day_train_min_day_test 需要 metadata_list 和 experiment_indices 参数")
        result = split_data_max_day_train_min_day_test(
            data, labels, metadata_list, experiment_indices
        )

    elif strategy == 'cross_user':
        raise ValueError("split_data 不直接支持 cross_user，请在 load_and_preprocess_data 中使用该策略")

    elif strategy == 'multi_user_independent':
        raise ValueError("split_data 不直接支持 multi_user_independent，请在 load_and_preprocess_data 中使用该策略")

    elif strategy == 'db_file_temporal_80_20':
        result = split_data_by_db_file_temporal_80_20(
            data, labels, timestamps, db_file_indices, db_file_names=db_file_names
        )

    elif strategy == 'tao_db_label_shuffle_debug':
        raise ValueError("split_data 不直接支持 tao_db_label_shuffle_debug，请在 load_and_preprocess_data 中使用该策略")
    
    else:
        raise ValueError(f"不支持的划分策略: {strategy}. 可选: random, temporal, stratified, experiment, date, date_random, max_day_train_min_day_test, alternating, hybrid_alternating, cross_user, multi_user_independent, db_file_temporal_80_20, tao_db_label_shuffle_debug")
    
    # 所有策略现在都返回9个值（包括indices）
    if return_indices:
        return result
    else:
        # 只返回前6个值（数据和标签）
        return result[:6]


def create_dataloaders(train_data, val_data, test_data, 
                       train_labels, val_labels, test_labels,
                       batch_size=32, num_workers=4, pin_memory=True, mode='instant',
                       shuffle_train=True, late_fusion=False, n_spectrum_features=1001):
    """
    创建DataLoader
    
    Args:
        mode: 'instant' 或 'window'
        shuffle_train: 是否打乱训练集顺序（对于时序窗口模式应设为False）
        late_fusion: 是否使用Late Fusion数据集（分离spectrum和aux）
        n_spectrum_features: spectrum特征维度（Late Fusion时使用）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 根据是否使用Late Fusion选择数据集类
    if late_fusion:
        # Late Fusion: 使用LateFusionDataset分离spectrum和aux
        train_dataset = LateFusionDataset(train_data, train_labels, n_spectrum_features, mode=mode)
        val_dataset = LateFusionDataset(val_data, val_labels, n_spectrum_features, mode=mode)
        test_dataset = LateFusionDataset(test_data, test_labels, n_spectrum_features, mode=mode)
    else:
        # Early Fusion或无融合: 使用普通数据集
        train_dataset = GlucoseDataset(train_data, train_labels, mode=mode)
        val_dataset = GlucoseDataset(val_data, val_labels, mode=mode)
        test_dataset = GlucoseDataset(test_data, test_labels, mode=mode)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,  # 可配置是否打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_ar_dataloaders(train_data, val_data, test_data, 
                          train_labels, val_labels, test_labels,
                          train_glucose_history, val_glucose_history, test_glucose_history,
                          batch_size=32, num_workers=4, pin_memory=True, mode='instant',
                          shuffle_train=True, has_aux=False, n_spectrum_features=1001):
    """
    创建Auto-Regressive模式的DataLoader
    
    Args:
        train_data, val_data, test_data: 特征数据
        train_labels, val_labels, test_labels: 目标血糖值
        train_glucose_history, val_glucose_history, test_glucose_history: 历史血糖序列
        batch_size: 批次大小
        num_workers: DataLoader工作进程数
        pin_memory: 是否使用固定内存
        mode: 'instant' 或 'window'
        shuffle_train: 是否打乱训练集顺序
        has_aux: 是否包含aux数据
        n_spectrum_features: spectrum特征维度（has_aux=True时使用）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建AR数据集
    train_dataset = AutoRegressiveDataset(
        train_data, train_labels, train_glucose_history,
        n_spectrum_features=n_spectrum_features if has_aux else None,
        mode=mode, has_aux=has_aux
    )
    
    val_dataset = AutoRegressiveDataset(
        val_data, val_labels, val_glucose_history,
        n_spectrum_features=n_spectrum_features if has_aux else None,
        mode=mode, has_aux=has_aux
    ) if len(val_data) > 0 else None
    
    test_dataset = AutoRegressiveDataset(
        test_data, test_labels, test_glucose_history,
        n_spectrum_features=n_spectrum_features if has_aux else None,
        mode=mode, has_aux=has_aux
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_window_data(features, labels, timestamps, experiment_indices, window_size, padding_mode='drop',
                       return_source_indices=False):
    """
    创建窗口数据：使用过去window_size个时间步的数据预测当前血糖
    重要：窗口不会跨越不同的实验组
    
    Args:
        features: 原始特征数据, shape (n_samples, n_features)
        labels: 原始标签数据, shape (n_samples,)
        timestamps: 时间戳数据, shape (n_samples,)
        experiment_indices: 实验分组索引, shape (n_samples,), 标记每个样本属于哪个实验
        window_size: 窗口大小（历史时间步数）
        padding_mode: 对于不足window_size的前期样本的处理方式
            - 'drop': 丢弃前window_size-1个样本（当前默认方式）
            - 'zero': 零填充不足的时间步
            - 'repeat': 用第一个时间步重复填充
            - 'edge': 用边缘值填充（第一个可用样本）
    
    Returns:
        window_features: shape (n_valid_samples, window_size, n_features)
        window_labels: shape (n_valid_samples,) - 对应当前时刻的血糖值
        window_timestamps: shape (n_valid_samples,) - 对应当前时刻的时间戳
        window_experiment_indices: shape (n_valid_samples,) - 对应的实验索引
        window_source_indices: shape (n_valid_samples,) - 对应原始2D样本索引（return_source_indices=True时返回）
    """
    n_samples, n_features = features.shape

    # 预计算样本数，避免 list -> np.array 的双倍峰值内存
    expected_windows = 0
    unique_experiments = np.unique(experiment_indices)
    for exp_id in unique_experiments:
        exp_indices = np.where(experiment_indices == exp_id)[0]
        if len(exp_indices) < 1:
            continue
        if padding_mode == 'drop':
            expected_windows += max(0, len(exp_indices) - window_size)
        else:
            expected_windows += max(0, len(exp_indices) - 1)

    if expected_windows <= 0:
        raise ValueError(f"无法创建窗口数据: 所有实验的样本数都小于window_size({window_size})")

    window_features = np.empty((expected_windows, window_size, n_features), dtype=np.float32)
    window_labels = np.empty((expected_windows,), dtype=np.float32)
    window_timestamps = np.empty((expected_windows,), dtype=np.float64)
    window_experiment_indices = np.empty((expected_windows,), dtype=np.int32)
    window_source_indices = np.empty((expected_windows,), dtype=np.int64) if return_source_indices else None

    estimated_bytes = expected_windows * window_size * n_features * 4
    
    print(f"    在 {len(unique_experiments)} 个实验内部分别创建窗口...")
    if padding_mode != 'drop':
        print(f"    使用填充模式: {padding_mode}（保留前期样本）")
    else:
        print(f"    使用丢弃模式（前 {window_size-1} 个样本将被丢弃）")
    print(f"    预计窗口样本: {expected_windows}, 预计窗口数组占用: {_fmt_nbytes(estimated_bytes)}")
    
    total_dropped = 0
    total_padded = 0
    
    write_idx = 0

    # 在每个实验内部分别创建窗口
    for exp_id in unique_experiments:
        # 找到属于当前实验的所有样本索引
        exp_mask = experiment_indices == exp_id
        exp_indices = np.where(exp_mask)[0]
        
        if len(exp_indices) < 1:
            continue
        
        if len(exp_indices) < window_size and padding_mode == 'drop':
            print(f"      警告: 实验{exp_id}样本数({len(exp_indices)})不足window_size({window_size})，跳过")
            total_dropped += len(exp_indices)
            continue
        
        # 确定起始位置
        if padding_mode == 'drop':
            start_idx = window_size  # 从第window_size个样本开始（索引从0开始，所以是第window_size+1个）
        else:
            start_idx = 1  # 从第2个样本开始（第1个样本没有历史，跳过）
            total_padded += min(window_size - 1, len(exp_indices) - 1)
        
        # 在当前实验内创建窗口
        for i in range(start_idx, len(exp_indices)):
            # 计算实际可用的历史长度
            available_history = i  # 在当前样本之前有i个样本可用
            
            if available_history >= window_size:
                # 有足够的历史数据，正常创建窗口
                global_window_indices = exp_indices[i-window_size:i]
                history_window = features[global_window_indices]
            else:
                # 历史数据不足，需要填充
                available_data = features[exp_indices[:i]]  # 所有可用的历史数据
                padding_needed = window_size - available_history
                
                if padding_mode == 'zero':
                    # 零填充
                    padding = np.zeros((padding_needed, n_features))
                    history_window = np.vstack([padding, available_data])
                
                elif padding_mode == 'repeat':
                    # 重复第一个样本
                    first_sample = features[exp_indices[0]]
                    padding = np.tile(first_sample, (padding_needed, 1))
                    history_window = np.vstack([padding, available_data])
                
                elif padding_mode == 'edge':
                    # 边缘填充（用第一个可用样本填充）
                    edge_sample = features[exp_indices[0]]
                    padding = np.tile(edge_sample, (padding_needed, 1))
                    history_window = np.vstack([padding, available_data])
                
                else:
                    raise ValueError(f"未知的填充模式: {padding_mode}")
            
            global_current_index = exp_indices[i]
            current_label = labels[global_current_index]
            current_timestamp = timestamps[global_current_index]
            
            window_features[write_idx] = history_window
            window_labels[write_idx] = current_label
            window_timestamps[write_idx] = current_timestamp
            window_experiment_indices[write_idx] = exp_id
            if return_source_indices:
                window_source_indices[write_idx] = global_current_index
            write_idx += 1

    # 实际样本数可能因为跳过短实验小于预估，做一次裁剪
    if write_idx != expected_windows:
        window_features = window_features[:write_idx]
        window_labels = window_labels[:write_idx]
        window_timestamps = window_timestamps[:write_idx]
        window_experiment_indices = window_experiment_indices[:write_idx]
        if return_source_indices:
            window_source_indices = window_source_indices[:write_idx]
    
    # 打印统计信息
    if total_dropped > 0:
        print(f"    ✓ 丢弃了 {total_dropped} 个前期样本（不足窗口大小）")
    if total_padded > 0:
        print(f"    ✓ 填充了 {total_padded} 个前期样本的历史窗口")
    print(f"    ✓ 最终窗口样本数: {len(window_features)}")
    print(f"    ✓ 数据大小: {window_features.nbytes / 1e9:.2f} GB")
    
    if return_source_indices:
        return window_features, window_labels, window_timestamps, window_experiment_indices, window_source_indices
    return window_features, window_labels, window_timestamps, window_experiment_indices


def create_window_sample_indices(experiment_indices, window_size, padding_mode='drop', n_features=None):
    """
    创建窗口样本对应的“当前时刻索引”，用于按需窗口Dataset。

    Returns:
        current_indices: 每个窗口样本对应原始2D特征中的当前时刻索引
        window_experiment_indices: 与窗口样本对齐的实验索引
    """
    if padding_mode != 'drop':
        raise ValueError("当前按需窗口仅支持 padding_mode='drop'")

    unique_experiments = np.unique(experiment_indices)
    expected_windows = 0
    for exp_id in unique_experiments:
        exp_indices = np.where(experiment_indices == exp_id)[0]
        expected_windows += max(0, len(exp_indices) - window_size)

    if expected_windows <= 0:
        raise ValueError(f"无法创建窗口索引: 所有实验的样本数都小于window_size({window_size})")

    current_indices = np.empty((expected_windows,), dtype=np.int64)
    window_experiment_indices = np.empty((expected_windows,), dtype=np.int32)

    estimated_bytes = expected_windows * window_size * (n_features if n_features is not None else 0) * 4

    print(f"    在 {len(unique_experiments)} 个实验内部分别创建窗口索引...")
    print(f"    使用丢弃模式（前 {window_size-1} 个样本将被丢弃）")
    if n_features is not None:
        print(f"    预计窗口样本: {expected_windows}, 若预展开3D数组将占用: {_fmt_nbytes(estimated_bytes)}")

    write_idx = 0
    total_dropped = 0
    for exp_id in unique_experiments:
        exp_indices = np.where(experiment_indices == exp_id)[0]
        n_valid = len(exp_indices) - window_size
        if n_valid <= 0:
            total_dropped += len(exp_indices)
            continue

        valid_curr = exp_indices[window_size:]
        end_idx = write_idx + n_valid
        current_indices[write_idx:end_idx] = valid_curr
        window_experiment_indices[write_idx:end_idx] = np.int32(exp_id)
        write_idx = end_idx

    if write_idx != expected_windows:
        current_indices = current_indices[:write_idx]
        window_experiment_indices = window_experiment_indices[:write_idx]

    if total_dropped > 0:
        print(f"    ✓ 丢弃了 {total_dropped} 个前期样本（不足窗口大小）")
    print(f"    ✓ 最终窗口样本数: {len(current_indices)}")
    print("    ✓ 按需窗口模式：未预分配3D窗口数组")

    return current_indices, window_experiment_indices


def create_time_window_data(features, labels, timestamps, experiment_indices, 
                            window_duration, downsample_interval=None, padding_mode='drop',
                            return_source_indices=False):
    """
    基于时间长度创建窗口数据：使用过去window_duration秒的数据预测当前血糖
    支持在每个窗口内进行下采样
    
    Args:
        features: 原始特征数据, shape (n_samples, n_features)
        labels: 原始标签数据, shape (n_samples,)
        timestamps: 时间戳数据, shape (n_samples,) - 单位：秒
        experiment_indices: 实验分组索引, shape (n_samples,)
        window_duration: 窗口时间长度（秒），例如30表示使用过去30秒的数据
        downsample_interval: 窗口内下采样间隔（秒），None表示不下采样
        padding_mode: 对于不足window_duration的前期样本的处理方式
            - 'drop': 丢弃前期样本（默认）
            - 'zero': 零填充不足的时间步
            - 'repeat': 用第一个时间步重复填充
            - 'edge': 用边缘值填充
    
    Returns:
        window_features: shape (n_valid_samples, window_size, n_features)
        window_labels: shape (n_valid_samples,)
        window_timestamps: shape (n_valid_samples,)
        window_experiment_indices: shape (n_valid_samples,)
        window_glucose_history: shape (n_valid_samples, window_size) - 窗口内下采样后的glucose历史
        window_source_indices: shape (n_valid_samples,) - 对应原始2D样本索引（return_source_indices=True时返回）
    """
    n_samples, n_features = features.shape
    
    window_features_list = []
    window_labels_list = []
    window_timestamps_list = []
    window_experiment_indices_list = []
    window_glucose_history_list = []  # 新增：保存窗口内glucose历史
    window_source_indices_list = []
    
    unique_experiments = np.unique(experiment_indices)
    
    print(f"    在 {len(unique_experiments)} 个实验内部分别创建时间窗口...")
    print(f"    窗口时长: {window_duration}秒")
    if downsample_interval:
        expected_samples = int(window_duration / downsample_interval)
        print(f"    窗口内下采样间隔: {downsample_interval}秒 (每窗口约{expected_samples}个样本)")
    else:
        print(f"    窗口内不进行下采样")
    
    total_dropped = 0
    total_windows = 0
    
    # 在每个实验内部分别创建窗口
    for exp_id in unique_experiments:
        exp_mask = experiment_indices == exp_id
        exp_indices = np.where(exp_mask)[0]
        
        if len(exp_indices) < 1:
            continue
        
        exp_timestamps = timestamps[exp_indices]
        exp_start_time = exp_timestamps[0]
        
        # 遍历每个时间点，创建向前看window_duration的窗口
        for i in range(len(exp_indices)):
            current_global_idx = exp_indices[i]
            current_time = timestamps[current_global_idx]
            current_label = labels[current_global_idx]
            
            # 计算窗口的起始时间
            window_start_time = current_time - window_duration
            
            # 找到窗口内的所有样本
            # 窗口范围: [current_time - window_duration, current_time)
            window_mask = (exp_timestamps >= window_start_time) & (exp_timestamps < current_time)
            window_sample_indices = exp_indices[window_mask]
            
            if len(window_sample_indices) == 0:
                if padding_mode == 'drop':
                    total_dropped += 1
                    continue
                else:
                    # 如果没有历史数据，根据padding模式处理
                    if padding_mode in ['zero', 'repeat', 'edge']:
                        # 需要估计窗口应该有多少样本
                        if downsample_interval:
                            target_window_size = int(window_duration / downsample_interval)
                        else:
                            # 估计原始采样率
                            if i > 0:
                                avg_interval = np.mean(np.diff(exp_timestamps[:i+1]))
                                target_window_size = max(1, int(window_duration / avg_interval))
                            else:
                                target_window_size = 1
                        
                        if padding_mode == 'zero':
                            history_window = np.zeros((target_window_size, n_features))
                        elif padding_mode == 'repeat' and i > 0:
                            first_sample = features[exp_indices[0]]
                            history_window = np.tile(first_sample, (target_window_size, 1))
                        elif padding_mode == 'edge' and i > 0:
                            edge_sample = features[exp_indices[0]]
                            history_window = np.tile(edge_sample, (target_window_size, 1))
                        else:
                            total_dropped += 1
                            continue
            else:
                # 提取窗口内的特征和标签
                window_features_raw = features[window_sample_indices]
                window_labels_raw = labels[window_sample_indices]
                window_timestamps_raw = timestamps[window_sample_indices]
                
                # 在窗口内进行下采样
                if downsample_interval:
                    # 创建下采样时间网格
                    downsample_times = np.arange(window_start_time, current_time, downsample_interval)
                    
                    if len(downsample_times) == 0:
                        total_dropped += 1
                        continue
                    
                    # 对每个下采样时间点，找到最近的原始样本
                    downsampled_features = []
                    downsampled_glucose = []  # 新增：保存下采样的glucose值
                    for target_time in downsample_times:
                        # 找到最接近target_time的样本
                        closest_idx = np.argmin(np.abs(window_timestamps_raw - target_time))
                        downsampled_features.append(window_features_raw[closest_idx])
                        downsampled_glucose.append(window_labels_raw[closest_idx])
                    
                    history_window = np.array(downsampled_features)
                    glucose_history_window = np.array(downsampled_glucose)
                else:
                    history_window = window_features_raw
                    glucose_history_window = window_labels_raw
            
            window_features_list.append(history_window)
            window_labels_list.append(current_label)
            window_timestamps_list.append(current_time)
            window_experiment_indices_list.append(exp_id)
            window_glucose_history_list.append(glucose_history_window)  # 新增：保存glucose历史
            if return_source_indices:
                window_source_indices_list.append(current_global_idx)
            total_windows += 1
    
    if not window_features_list:
        raise ValueError(f"无法创建时间窗口数据: 所有样本的历史时长都小于{window_duration}秒")
    
    # 计算目标窗口大小（基于window_duration和downsample_interval）
    if downsample_interval:
        target_window_size = int(window_duration / downsample_interval)
    else:
        # 如果没有下采样，使用最大窗口大小
        target_window_size = max(len(w) for w in window_features_list)
    
    # 确保所有窗口有相同的大小（padding到目标大小）
    window_sizes = [len(w) for w in window_features_list]
    min_window_size = min(window_sizes)
    max_window_size = max(window_sizes)
    
    if min_window_size != target_window_size or max_window_size != target_window_size:
        print(f"    ⚠️  窗口大小不一致: {min_window_size} - {max_window_size} 样本")
        print(f"    使用edge padding统一到 {target_window_size} 个样本 (目标窗口长度)")
        
        # 统一窗口大小到目标大小
        uniform_window_features = []
        uniform_glucose_history = []  # 新增：同步处理glucose历史
        for window, glucose_hist in zip(window_features_list, window_glucose_history_list):
            if len(window) > target_window_size:
                # 截断：保留最后target_window_size个样本（最接近当前时刻）
                uniform_window_features.append(window[-target_window_size:])
                uniform_glucose_history.append(glucose_hist[-target_window_size:])
            elif len(window) < target_window_size:
                # Edge padding：用第一个样本填充前面的部分
                padding_needed = target_window_size - len(window)
                edge_value = window[0] if len(window) > 0 else np.zeros(n_features)
                padding = np.tile(edge_value, (padding_needed, 1))
                uniform_window_features.append(np.vstack([padding, window]))
                
                # Glucose历史也进行edge padding
                glucose_edge_value = glucose_hist[0] if len(glucose_hist) > 0 else 0.0
                glucose_padding = np.full(padding_needed, glucose_edge_value)
                uniform_glucose_history.append(np.concatenate([glucose_padding, glucose_hist]))
            else:
                uniform_window_features.append(window)
                uniform_glucose_history.append(glucose_hist)
        window_features_list = uniform_window_features
        window_glucose_history_list = uniform_glucose_history
    
    # 使用 float32 减少内存占用（float64 -> float32 可节省50%内存）
    print(f"    转换为numpy数组（使用float32以节省内存）...")
    window_features = np.array(window_features_list, dtype=np.float32)
    window_labels = np.array(window_labels_list, dtype=np.float32)
    window_timestamps = np.array(window_timestamps_list, dtype=np.float64)  # 时间戳保持float64精度
    window_experiment_indices = np.array(window_experiment_indices_list, dtype=np.int32)
    window_glucose_history = np.array(window_glucose_history_list, dtype=np.float32)
    window_source_indices = np.array(window_source_indices_list, dtype=np.int64) if return_source_indices else None
    
    # 释放原始列表内存
    del window_features_list, window_labels_list, window_glucose_history_list
    
    # 打印统计信息
    if total_dropped > 0:
        print(f"    ✓ 丢弃了 {total_dropped} 个前期样本（历史时长不足{window_duration}秒）")
    print(f"    ✓ 最终窗口样本数: {len(window_features)}")
    print(f"    ✓ 每个窗口的时间步数: {window_features.shape[1]}")
    print(f"    ✓ 数据大小: {window_features.nbytes / 1e9:.2f} GB")
    
    if return_source_indices:
        return window_features, window_labels, window_timestamps, window_experiment_indices, window_glucose_history, window_source_indices
    return window_features, window_labels, window_timestamps, window_experiment_indices, window_glucose_history


def create_glucose_history(labels, experiment_indices, ar_history, ar_aligned=True, padding_mode='edge'):
    """
    为自回归模式创建历史血糖序列
    
    Args:
        labels: numpy array, shape (n_samples,) - 所有样本的血糖标签
        experiment_indices: numpy array, shape (n_samples,) - 每个样本所属的实验ID
        ar_history: int - 使用多少个历史血糖值
        ar_aligned: bool - True: 历史不包含当前血糖（当前血糖未知）
                         False: 历史包含当前血糖（训练时可用于增强趋势学习）
        padding_mode: str - 前期样本不足时的填充模式: 'edge'(边缘值), 'zero'(零值)
    
    Returns:
        glucose_history: numpy array, shape (n_samples, ar_history) - 历史血糖序列
        valid_mask: numpy array, shape (n_samples,) - 标记有效样本（历史足够的）
    
    注意：
    - 如果ar_aligned=True，第i个样本的history是 [label[i-ar_history], ..., label[i-1]]
    - 如果ar_aligned=False，第i个样本的history是 [label[i-ar_history+1], ..., label[i]]
    - 历史不会跨越不同实验
    """
    n_samples = len(labels)
    glucose_history = np.zeros((n_samples, ar_history), dtype=np.float32)
    valid_mask = np.ones(n_samples, dtype=bool)
    
    # 按实验处理
    unique_experiments = np.unique(experiment_indices)
    
    for exp_id in unique_experiments:
        exp_mask = experiment_indices == exp_id
        exp_indices = np.where(exp_mask)[0]
        exp_labels = labels[exp_indices]
        
        for local_i, global_i in enumerate(exp_indices):
            if ar_aligned:
                # 历史不包含当前：[i-ar_history, i-1]
                history_start = local_i - ar_history
                history_end = local_i
            else:
                # 历史包含当前：[i-ar_history+1, i]
                history_start = local_i - ar_history + 1
                history_end = local_i + 1
            
            if history_start >= 0:
                # 有足够的历史
                glucose_history[global_i] = exp_labels[history_start:history_end]
            elif history_end > 0:
                # 部分历史可用，需要填充
                available_start = max(0, history_start)
                available_history = exp_labels[available_start:history_end]
                padding_len = ar_history - len(available_history)
                
                if padding_mode == 'edge':
                    # 用第一个可用值填充
                    padding_value = available_history[0] if len(available_history) > 0 else 0
                    padding = np.full(padding_len, padding_value)
                else:
                    # 零填充
                    padding = np.zeros(padding_len)
                
                glucose_history[global_i] = np.concatenate([padding, available_history])
            else:
                # 完全没有历史（第一个样本）
                if padding_mode == 'edge':
                    glucose_history[global_i] = np.full(ar_history, labels[global_i])
                else:
                    glucose_history[global_i] = np.zeros(ar_history)
                # 标记为部分有效（可选：根据需要调整）
    
    return glucose_history, valid_mask


def load_and_preprocess_data(config, dataset_root="./Dataset", normalize=False, 
                             seed=42, batch_size=32, num_workers=0):
    """
    加载并预处理真实数据集的主函数
    
    改进：无需手动配置实验时间，自动从目录名提取
    支持 'bin' 和 'db' 两种数据源格式
    
    Args:
        config: 配置对象，需包含:
            - user_name: 用户名（如 "Tao"）
            - data_source: 数据源格式 ('bin' 或 'db')
            - train_split, val_split, test_split: 数据划分比例
        dataset_root: Dataset 目录路径
        normalize: 是否对特征进行归一化，默认 False（保留原始数据）
        seed: 随机种子，用于数据划分
        batch_size: 批次大小
        num_workers: DataLoader 工作进程数
    
    Returns:
        train_loader, val_loader, test_loader, metadata_list
        如果 normalize=True，则返回 train_loader, val_loader, test_loader, scaler, metadata_list
    """
    print("\n" + "="*70)
    print("加载真实数据集")
    print("="*70)
    
    # 获取数据源配置
    data_source = getattr(config, 'data_source', 'bin')
    split_strategy = getattr(config, 'split_strategy', 'random')
    cross_user_mode = (split_strategy == 'cross_user')
    multi_user_independent_mode = (split_strategy == 'multi_user_independent')
    db_file_temporal_mode = (split_strategy == 'db_file_temporal_80_20')
    tao_db_label_shuffle_debug_mode = (split_strategy == 'tao_db_label_shuffle_debug')
    
    # 根据数据源选择不同的加载器
    if data_source == 'db':
        # 使用 DB 格式加载器（支持多用户）
        db_user_names = getattr(config, 'db_user_names', None)
        if db_user_names:
            print(f"数据源: DB格式 (多用户, root={dataset_root})")
        else:
            db_user_name = getattr(config, 'db_user_name', 'Tao_db')
            print(f"数据源: DB格式 (root={dataset_root}, user={db_user_name})")
        loader = DBGlucoseDatasetLoader(dataset_root=dataset_root)
    else:
        # 使用原始 BIN 格式加载器（支持多用户）
        print(f"数据源: BIN格式 ({dataset_root})")
        loader = GlucoseDatasetLoader(dataset_root=dataset_root)
    
    # 获取用户名/用户列表
    user_names = None
    if data_source == 'db':
        user_names = getattr(config, 'db_user_names', None)
        if not user_names:
            user_names = [getattr(config, 'db_user_name', 'Tao_db')]
    else:
        user_names = getattr(config, 'user_names', None)
        if not user_names:
            user_names = [getattr(config, 'user_name', 'Tao')]

    if cross_user_mode:
        train_user_name = getattr(config, 'train_user_name', None)
        predict_user_name = getattr(config, 'predict_user_name', None)
        if not train_user_name or not predict_user_name:
            raise ValueError("split_strategy='cross_user' 需要同时提供 --train_user_name 和 --predict_user_name")
        if train_user_name == predict_user_name:
            raise ValueError("cross_user模式下训练用户和预测用户不能相同")
        user_names = [train_user_name, predict_user_name]
    elif tao_db_label_shuffle_debug_mode:
        user_names = ['Tao_db']
    
    # 加载实验数据（自动从目录名提取时间）
    experiment_filter = getattr(config, 'experiment_filter', None)
    data_fusion = getattr(config, 'data_fusion', False)

    if tao_db_label_shuffle_debug_mode:
        if data_source != 'db':
            raise ValueError("split_strategy='tao_db_label_shuffle_debug' 仅支持 data_source='db'")

        requested_db_users = getattr(config, 'db_user_names', None)
        requested_db_user = getattr(config, 'db_user_name', 'Tao_db')

        if requested_db_users is not None:
            if len(requested_db_users) != 1 or requested_db_users[0] != 'Tao_db':
                raise ValueError("tao_db_label_shuffle_debug 仅支持单用户 Tao_db，请使用 --db_user Tao_db")
        elif requested_db_user != 'Tao_db':
            raise ValueError("tao_db_label_shuffle_debug 仅支持单用户 Tao_db，请使用 --db_user Tao_db")

        if experiment_filter is not None or getattr(config, 'user_experiments', None) is not None:
            print("  ⚠ tao_db_label_shuffle_debug 将忽略 experiments/user_experiments 过滤，强制使用 Tao_db 全部数据")
            experiment_filter = None
            config.data.experiment_filter = None
            config.data.user_experiments = None

        if data_fusion:
            print("  ⚠ tao_db_label_shuffle_debug 强制关闭多模态融合，仅使用 spectrum 单模态")
            data_fusion = False
            config.data.data_fusion = False
    
    # 检查是否使用时间窗口模式（如果是，则不在加载时下采样）
    mode = getattr(config, 'mode', 'instant')
    window_size = getattr(config, 'window_size', 10)
    window_duration = getattr(config, 'window_duration', None)
    window_padding = getattr(config, 'window_padding', 'drop')
    use_time_window = (mode == 'window' and window_duration is not None)
    
    # 如果使用时间窗口模式，暂时禁用全局下采样（将在窗口内下采样）
    if use_time_window:
        downsample = False
        downsample_interval = 1.0
        print(f"  ℹ️  时间窗口模式：全局下采样已禁用，将在窗口内进行下采样")
    else:
        downsample = getattr(config, 'downsample', False)
        downsample_interval = getattr(config, 'downsample_interval', 1.0)
    
    interpolation_method = getattr(config, 'interpolation_method', 'pchip')
    
    # 构建平滑配置
    smooth_config = None
    if getattr(config, 'smooth_data', False):
        smooth_config = {
            'smooth_data': True,
            'time_smooth_method': getattr(config, 'time_smooth_method', 'none'),
            'time_smooth_window': getattr(config, 'time_smooth_window', 5),
            'time_smooth_sigma': getattr(config, 'time_smooth_sigma', 2.0),
            'time_smooth_alpha': getattr(config, 'time_smooth_alpha', 0.3),
            'spectrum_smooth_method': getattr(config, 'spectrum_smooth_method', 'none'),
            'spectrum_smooth_window': getattr(config, 'spectrum_smooth_window', 5),
            'spectrum_smooth_sigma': getattr(config, 'spectrum_smooth_sigma', 2.0),
            'detect_outliers': getattr(config, 'detect_outliers', False),
            'outlier_method': getattr(config, 'outlier_method', 'zscore'),
            'outlier_threshold': getattr(config, 'outlier_threshold', 3.0),
            'handle_outliers': getattr(config, 'handle_outliers', False),
            'outlier_handle_method': getattr(config, 'outlier_handle_method', 'interpolate')
        }
    
    # 根据融合阶段选择对应的fusion_method
    fusion_stage = getattr(config, 'fusion_stage', 'early')
    if fusion_stage == 'early':
        fusion_method = getattr(config, 'fusion_method', 'concat')
    else:
        fusion_method = getattr(config, 'fusion_method_late', 'concat')
    
    icm_mode = getattr(config, 'icm_mode', 'raw')
    use_autoregressive = getattr(config, 'autoregressive', False)
    use_late_fusion = data_fusion and fusion_stage == 'late'
    lazy_window_enabled = bool(getattr(config, 'lazy_window', True))
    lazy_window_active = False
    base_features_2d = None
    sample_source_indices = None
    
    # 构建fusion_config（包含aux_override等配置）
    fusion_config = None
    if data_fusion:
        fusion_config = {
            'aux_override': getattr(config, 'aux_override', False),
            'aux_override_value': getattr(config, 'aux_override_value', 0.0),
            'aux_feature_mode': getattr(config, 'aux_feature_mode', 'raw'),
            'aux_shuffle': getattr(config, 'aux_shuffle', False),
            'aux_shuffle_mode': getattr(config, 'aux_shuffle_mode', 'sensor_groups'),
            'aux_shuffle_seed': getattr(config, 'aux_shuffle_seed', 42)
        }
    if db_file_temporal_mode:
        if data_source != 'db':
            raise ValueError("split_strategy='db_file_temporal_80_20' 仅支持 data_source='db'")
        if fusion_config is None:
            fusion_config = {}
        fusion_config['track_db_file_source'] = True
    
    # 获取用户实验映射 ⭐ NEW
    user_experiments = getattr(config, 'user_experiments', None)
    
    # 加载单用户或多用户数据
    if len(user_names) == 1:
        # 为单用户获取实验过滤器
        user_experiment_filter = experiment_filter
        if user_experiments and user_names[0] in user_experiments:
            user_experiment_filter = user_experiments[user_names[0]]
            print(f"  ✓ 用户 {user_names[0]} 使用指定实验: {', '.join(user_experiment_filter)}")
        else:
            print(f"  ✓ 用户 {user_names[0]} 加载所有实验（默认）")
        
        features, labels, timestamps, metadata_list, experiment_indices = loader.load_user_all_experiments(
            user_name=user_names[0],
            interpolation_method=interpolation_method,
            experiment_filter=user_experiment_filter,
            data_fusion=data_fusion,
            fusion_method=fusion_method,
            fusion_stage=fusion_stage,  # 传递fusion_stage参数
            downsample=downsample,
            downsample_interval=downsample_interval,
            smooth_config=smooth_config,
            icm_mode=icm_mode,
            fusion_config=fusion_config  # 传递fusion_config参数
        )
        # 保证metadata包含user信息
        if metadata_list:
            for meta in metadata_list:
                meta.setdefault('user_name', user_names[0])
                meta.setdefault('user_index', 0)
    else:
        # 多用户DB模式下，先做文件存在性预检查，避免先加载大数组后才在后续用户失败。
        if data_source == 'db' and hasattr(loader, 'validate_user_db_files'):
            missing_report = []
            for user_name in user_names:
                user_experiment_filter = experiment_filter
                if user_experiments and user_name in user_experiments:
                    user_experiment_filter = user_experiments[user_name]

                missing_files = loader.validate_user_db_files(
                    user_name=user_name,
                    experiment_filter=user_experiment_filter
                )
                if missing_files:
                    missing_report.append((user_name, missing_files))

            if missing_report:
                lines = ["检测到缺失的DB文件，已在加载前中止（避免内存激增）:"]
                for user_name, files in missing_report:
                    lines.append(f"  用户 {user_name}:")
                    for p in files:
                        lines.append(f"    - {p}")
                raise FileNotFoundError("\n".join(lines))

        all_features_list = []
        all_labels_list = []
        all_timestamps_list = []
        all_metadata_list = []
        all_experiment_indices_list = []
        experiment_offset = 0
        for user_idx, user_name in enumerate(user_names):
            print(f"\n{'-'*60}")
            print(f"加载用户: {user_name} ({user_idx+1}/{len(user_names)})")
            
            # 为当前用户获取实验过滤器 ⭐ NEW
            user_experiment_filter = experiment_filter
            if user_experiments and user_name in user_experiments:
                user_experiment_filter = user_experiments[user_name]
                print(f"  ✓ 用户 {user_name} 使用指定实验: {', '.join(user_experiment_filter)}")
            else:
                print(f"  ✓ 用户 {user_name} 加载所有实验（默认）")
            
            try:
                user_features, user_labels, user_timestamps, user_metadata_list, user_exp_indices = loader.load_user_all_experiments(
                    user_name=user_name,
                    interpolation_method=interpolation_method,
                    experiment_filter=user_experiment_filter,
                    data_fusion=data_fusion,
                    fusion_method=fusion_method,
                    fusion_stage=fusion_stage,
                    downsample=downsample,
                    downsample_interval=downsample_interval,
                    smooth_config=smooth_config,
                    icm_mode=icm_mode,
                    fusion_config=fusion_config
                )
            except Exception as e:
                # 失败时主动释放已聚合的大数组，降低异常路径下的内存峰值
                all_features_list.clear()
                all_labels_list.clear()
                all_timestamps_list.clear()
                all_metadata_list.clear()
                all_experiment_indices_list.clear()
                gc.collect()
                raise RuntimeError(f"加载用户 {user_name} 失败，已中止多用户聚合: {e}") from e

            # metadata追加用户信息
            for meta in user_metadata_list:
                meta['user_name'] = user_name
                meta['user_index'] = user_idx
            # 实验索引全局偏移
            user_exp_indices = user_exp_indices + experiment_offset
            experiment_offset += len(user_metadata_list)
            # 聚合
            all_features_list.append(user_features)
            all_labels_list.append(user_labels)
            all_timestamps_list.append(user_timestamps)
            all_metadata_list.extend(user_metadata_list)
            all_experiment_indices_list.append(user_exp_indices)
        features = np.vstack(all_features_list)
        labels = np.concatenate(all_labels_list)
        timestamps = np.concatenate(all_timestamps_list)
        metadata_list = all_metadata_list
        experiment_indices = np.concatenate(all_experiment_indices_list)

        # 及时释放多用户拼接中间列表，降低后续窗口构造前内存峰值
        del all_features_list, all_labels_list, all_timestamps_list, all_metadata_list, all_experiment_indices_list
        gc.collect()

    if cross_user_mode:
        print(f"  ✓ cross_user模式: train_user={user_names[0]}, predict_user={user_names[1]}")
    elif multi_user_independent_mode:
        print(f"  ✓ multi_user_independent模式: {len(user_names)}个用户将独立切分并合并训练集")
    elif db_file_temporal_mode:
        print("  ✓ db_file_temporal_80_20模式: 每个.db文件内前80%训练、后20%测试")
    
    if experiment_filter:
        print(f"  ✓ 已筛选实验: {', '.join(experiment_filter)}")
    else:
        print(f"  ✓ 使用所有可用实验")
    
    print("数据预处理...")

    full_timestamps = timestamps
    full_glucose = labels
    full_experiment_indices = experiment_indices
    original_db_file_indices = None
    db_file_id_to_name = {}
    db_file_id_to_path = {}

    if db_file_temporal_mode:
        original_db_file_indices = np.empty(len(labels), dtype=np.int32)
        sample_start = 0
        next_db_file_id = 0
        for exp_id, meta in enumerate(metadata_list):
            n_samples = int(meta.get('n_samples', 0))
            sample_end = sample_start + n_samples
            sample_db_ids = meta.get('sample_db_file_ids')
            db_names = meta.get('db_file_names', [])
            db_paths = meta.get('db_file_paths', [])

            if sample_db_ids is None:
                raise ValueError(
                    "db_file_temporal_80_20 需要DB加载器记录每个样本的源.db文件，"
                    f"但实验 {meta.get('experiment_name', exp_id)} 缺少 sample_db_file_ids"
                )
            if len(sample_db_ids) != n_samples:
                raise ValueError(
                    f"实验 {meta.get('experiment_name', exp_id)} 的 sample_db_file_ids 长度不一致: "
                    f"{len(sample_db_ids)} vs n_samples={n_samples}"
                )

            local_to_global = {}
            for local_db_id in np.unique(sample_db_ids):
                local_db_id_int = int(local_db_id)
                db_name = db_names[local_db_id_int] if local_db_id_int < len(db_names) else f"db_{local_db_id_int}"
                db_path = db_paths[local_db_id_int] if local_db_id_int < len(db_paths) else db_name
                global_db_id = next_db_file_id
                next_db_file_id += 1
                local_to_global[local_db_id_int] = global_db_id
                exp_name = meta.get('experiment_name', f'exp_{exp_id}')
                db_file_id_to_name[global_db_id] = f"{exp_name}/{db_name}"
                db_file_id_to_path[global_db_id] = db_path

            for local_db_id, global_db_id in local_to_global.items():
                mask = (sample_db_ids == local_db_id)
                original_db_file_indices[sample_start:sample_end][mask] = global_db_id

            sample_start = sample_end

        if sample_start != len(labels):
            raise ValueError(
                f"DB文件来源索引构造失败: metadata样本数={sample_start}, labels样本数={len(labels)}"
            )

    window_glucose_history_full = None  # 初始化：用于存储窗口内glucose历史
    source_sample_indices = None
    
    if mode == 'window':
        if use_time_window:
            # 时间窗口模式：先创建窗口，再在窗口内下采样
            print(f"  - 使用时间窗口模式: window_duration={window_duration}秒, padding={window_padding}")
            
            # 获取下采样参数
            downsample_enabled = getattr(config, 'downsample', False)
            downsample_interval = getattr(config, 'downsample_interval', 1.0) if downsample_enabled else None
            
            # 创建时间窗口数据（包含窗口内下采样）
            time_window_result = create_time_window_data(
                features, labels, timestamps, experiment_indices,
                window_duration=window_duration,
                downsample_interval=downsample_interval,
                padding_mode=window_padding,
                return_source_indices=db_file_temporal_mode
            )
            if db_file_temporal_mode:
                features, labels, timestamps, experiment_indices, window_glucose_history_full, source_sample_indices = time_window_result
            else:
                features, labels, timestamps, experiment_indices, window_glucose_history_full = time_window_result
            print(f"    ✓ 时间窗口数据: {features.shape[0]} 个样本, 每个窗口包含 {features.shape[1]} 个时间步, 窗口时长 {window_duration}秒")
        else:
            # 传统窗口模式：基于样本数
            print(f"  - 使用窗口模式: window_size={window_size}, padding={window_padding}")
            lazy_window_compatible = (
                lazy_window_enabled
                and window_padding == 'drop'
                and not use_autoregressive
            )

            if lazy_window_compatible:
                base_features_2d = features
                sample_source_indices, experiment_indices = create_window_sample_indices(
                    experiment_indices,
                    window_size,
                    padding_mode=window_padding,
                    n_features=base_features_2d.shape[1]
                )
                labels = labels[sample_source_indices]
                timestamps = timestamps[sample_source_indices]
                source_sample_indices = sample_source_indices
                # 仅保留当前时刻特征用于划分与调试；训练时将基于source_indices按需取窗口
                features = base_features_2d[sample_source_indices]
                lazy_window_active = True
                print(f"    ✓ 懒窗口数据: {features.shape[0]} 个样本, 每个样本按需读取 {window_size} 个历史时间步")
            else:
                if lazy_window_enabled:
                    print("    ℹ️  当前配置不满足懒窗口条件，回退到预展开3D窗口模式")
                # 创建窗口数据（在每个实验内部分别创建）
                window_result = create_window_data(
                    features, labels, timestamps, experiment_indices, window_size,
                    padding_mode=window_padding,
                    return_source_indices=db_file_temporal_mode
                )
                if db_file_temporal_mode:
                    features, labels, timestamps, experiment_indices, source_sample_indices = window_result
                else:
                    features, labels, timestamps, experiment_indices = window_result
                print(f"    ✓ 窗口数据: {features.shape[0]} 个样本, 每个样本包含 {window_size} 个历史时间步")
    else:
        print(f"  - 使用即时模式: 当前spectrum预测当前血糖")

    db_file_indices = None
    if db_file_temporal_mode:
        if mode == 'window':
            if source_sample_indices is None:
                raise ValueError("db_file_temporal_80_20 无法获取窗口样本对应的原始DB来源索引")
            db_file_indices = original_db_file_indices[source_sample_indices]
        else:
            db_file_indices = original_db_file_indices
    
    # 检查是否使用验证集
    use_val_set = getattr(config, 'use_val_set', True)
    split_strategy = getattr(config, 'split_strategy', 'random')

    # 按日期划分仅支持单用户，避免跨用户日期混合造成泄漏
    if split_strategy in ['date', 'date_random', 'max_day_train_min_day_test'] and len(user_names) != 1:
        raise ValueError("split_strategy='date'/'date_random' 或 'max_day_train_min_day_test' 仅支持单用户训练，请使用 --user/--db_user 并确保只加载1个用户")

    if split_strategy == 'cross_user' and len(user_names) != 2:
        raise ValueError("split_strategy='cross_user' 需要且仅需要2个用户（train_user和predict_user）")

    if split_strategy == 'multi_user_independent' and len(user_names) < 2:
        raise ValueError("split_strategy='multi_user_independent' 需要至少2个用户")

    if split_strategy == 'db_file_temporal_80_20' and use_val_set:
        print("  - db_file_temporal_80_20策略按每个.db文件前80%/后20%划分，已自动关闭验证集")
        use_val_set = False

    def _make_derangement(n_samples, rng_seed):
        if n_samples < 2:
            raise ValueError("tao_db_label_shuffle_debug 至少需要2个样本才能打乱标签对应关系")

        rng = np.random.default_rng(rng_seed)
        base_indices = np.arange(n_samples)

        for _ in range(64):
            perm = rng.permutation(n_samples)
            if not np.any(perm == base_indices):
                return perm

        return np.roll(base_indices, 1)
    
    # 先划分数据集（使用原始特征）
    if split_strategy == 'cross_user':
        # 跨用户划分：训练用户全量训练，预测用户作为测试集（可按比例截取）
        if use_val_set:
            print("  - cross_user策略默认不使用验证集，已自动关闭验证集")
        use_val_set = False

        predict_user_ratio = float(getattr(config, 'predict_user_ratio', 1.0))
        if predict_user_ratio <= 0 or predict_user_ratio > 1.0:
            raise ValueError(f"predict_user_ratio 必须在 (0, 1] 范围内，当前为 {predict_user_ratio}")

        experiment_to_user = np.array([m.get('user_index', 0) for m in metadata_list], dtype=np.int32)
        all_user_indices = experiment_to_user[experiment_indices]

        train_indices = np.where(all_user_indices == 0)[0]
        test_indices_all = np.where(all_user_indices == 1)[0]

        if len(train_indices) == 0 or len(test_indices_all) == 0:
            raise ValueError("cross_user划分失败：训练用户或预测用户样本为空")

        if predict_user_ratio < 1.0:
            keep_n = max(1, int(len(test_indices_all) * predict_user_ratio))
            test_indices = test_indices_all[:keep_n]
            print(f"  - 预测用户测试集按比例截取: {keep_n}/{len(test_indices_all)} ({predict_user_ratio:.1%})")
        else:
            test_indices = test_indices_all

        val_indices = np.array([], dtype=int)

        train_data = features[train_indices]
        train_labels = labels[train_indices]
        val_data = features[:0]
        val_labels = labels[:0]
        test_data = features[test_indices]
        test_labels = labels[test_indices]

        train_timestamps = timestamps[train_indices]
        val_timestamps = timestamps[:0]
        test_timestamps = timestamps[test_indices]

        train_exp_indices = experiment_indices[train_indices]
        val_exp_indices = experiment_indices[:0]
        test_exp_indices = experiment_indices[test_indices]

    elif split_strategy == 'multi_user_independent':
        # 多用户独立划分：每个用户按相同比例独立切分，再合并所有用户训练集
        if use_val_set:
            print("  - multi_user_independent策略默认不使用验证集，已自动关闭验证集")
        use_val_set = False

        test_ratio = float(getattr(config, 'test_split', 0.15))
        if test_ratio <= 0 or test_ratio >= 1.0:
            raise ValueError(f"multi_user_independent要求 test_split 在 (0, 1) 范围内，当前为 {test_ratio}")

        experiment_to_user = np.array([m.get('user_index', 0) for m in metadata_list], dtype=np.int32)
        all_user_indices = experiment_to_user[experiment_indices]

        user_index_to_name = {}
        for meta in metadata_list:
            idx = int(meta.get('user_index', 0))
            user_index_to_name.setdefault(idx, meta.get('user_name', f'user_{idx}'))

        unique_user_indices = np.unique(all_user_indices)
        train_indices_parts = []
        test_indices_parts = []

        print(f"  - 按用户独立划分（test_ratio={test_ratio:.1%}）")
        for user_idx in unique_user_indices:
            user_sample_indices = np.where(all_user_indices == user_idx)[0]
            user_name = user_index_to_name.get(int(user_idx), f'user_{int(user_idx)}')

            if len(user_sample_indices) < 2:
                raise ValueError(f"用户 {user_name} 样本数不足2，无法进行训练/测试划分")

            user_train_indices, user_test_indices = train_test_split(
                user_sample_indices,
                test_size=test_ratio,
                random_state=seed,
                shuffle=True
            )

            train_indices_parts.append(user_train_indices)
            test_indices_parts.append(user_test_indices)
            print(f"    用户 {user_name}: train={len(user_train_indices)}, test={len(user_test_indices)}")

        train_indices = np.concatenate(train_indices_parts)
        test_indices = np.concatenate(test_indices_parts)
        val_indices = np.array([], dtype=int)

        train_data = features[train_indices]
        train_labels = labels[train_indices]
        val_data = features[:0]
        val_labels = labels[:0]
        test_data = features[test_indices]
        test_labels = labels[test_indices]

        train_timestamps = timestamps[train_indices]
        val_timestamps = timestamps[:0]
        test_timestamps = timestamps[test_indices]

        train_exp_indices = experiment_indices[train_indices]
        val_exp_indices = experiment_indices[:0]
        test_exp_indices = experiment_indices[test_indices]

    elif split_strategy == 'db_file_temporal_80_20':
        print("  - 按每个.db文件时间顺序划分（前80%训练，后20%测试）...")

        result = split_data(
            features, labels,
            metadata_list=metadata_list,
            strategy=split_strategy,
            train_ratio=0.8,
            val_ratio=0,
            test_ratio=0.2,
            random_state=seed,
            return_indices=True,
            experiment_indices=experiment_indices,
            timestamps=timestamps,
            db_file_indices=db_file_indices,
            db_file_names=db_file_id_to_name
        )

        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices = result

        train_timestamps = timestamps[train_indices]
        val_timestamps = timestamps[:0]
        test_timestamps = timestamps[test_indices]

        train_exp_indices = experiment_indices[train_indices]
        val_exp_indices = experiment_indices[:0]
        test_exp_indices = experiment_indices[test_indices]

    elif split_strategy == 'tao_db_label_shuffle_debug':
        print("  - Tao_db debug划分：先打乱 spectrum 与血糖标签的一一对应关系，再做随机划分")
        label_permutation = _make_derangement(len(labels), seed)
        labels = labels[label_permutation]
        full_glucose = labels
        print(f"    ✓ 已完成标签错配打乱，样本数: {len(labels)}")

        if use_val_set:
            print("  - 划分数据集（训练/验证/测试，随机）...")
            result = split_data(
                features, labels,
                metadata_list=metadata_list,
                strategy='random',
                train_ratio=config.train_split,
                val_ratio=config.val_split,
                test_ratio=config.test_split,
                random_state=seed,
                return_indices=True
            )
        else:
            print("  - 划分数据集（训练/测试，无验证集，随机）...")
            result = split_data(
                features, labels,
                metadata_list=metadata_list,
                strategy='random',
                train_ratio=1 - config.test_split,
                val_ratio=0,
                test_ratio=config.test_split,
                random_state=seed,
                return_indices=True
            )

        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices = result

        train_timestamps = timestamps[train_indices]
        test_timestamps = timestamps[test_indices]
        train_exp_indices = experiment_indices[train_indices]
        test_exp_indices = experiment_indices[test_indices]

        if use_val_set:
            val_timestamps = timestamps[val_indices] if len(val_indices) > 0 else timestamps[:0]
            val_exp_indices = experiment_indices[val_indices] if len(val_indices) > 0 else experiment_indices[:0]
        else:
            val_timestamps = timestamps[:0]
            val_exp_indices = experiment_indices[:0]

    elif use_val_set:
        print("  - 划分数据集（训练/验证/测试）...")
        n_splits = getattr(config, 'n_splits', 10)  # 获取n_splits参数
        date_train_days = getattr(config, 'date_train_days', None)
        
        # 获取划分结果和索引
        result = split_data(
            features, labels,
            metadata_list=metadata_list,
            strategy=split_strategy,
            train_ratio=config.train_split,
            val_ratio=config.val_split,
            test_ratio=config.test_split,
            random_state=seed,
            n_splits=n_splits,
            return_indices=True,  # 获取索引
            experiment_indices=experiment_indices,  # 传递实验索引
            date_train_days=date_train_days
        )
        
        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices = result
        
        # 使用相同的索引划分timestamps和experiment_indices
        train_timestamps = timestamps[train_indices]
        val_timestamps = timestamps[val_indices] if len(val_indices) > 0 else timestamps[:0]
        test_timestamps = timestamps[test_indices]
        
        train_exp_indices = experiment_indices[train_indices]
        val_exp_indices = experiment_indices[val_indices] if len(val_indices) > 0 else experiment_indices[:0]
        test_exp_indices = experiment_indices[test_indices]
    else:
        print("  - 划分数据集（训练/测试，无验证集）...")
        # 无验证集模式：直接二分
        n_splits = getattr(config, 'n_splits', 10)
        date_train_days = getattr(config, 'date_train_days', None)
        
        # 获取划分结果和索引
        result = split_data(
            features, labels,
            metadata_list=metadata_list,
            strategy=split_strategy,
            train_ratio=1 - config.test_split,
            val_ratio=0,  # 无验证集
            test_ratio=config.test_split,
            random_state=seed,
            n_splits=n_splits,
            return_indices=True,  # 获取索引
            experiment_indices=experiment_indices,  # 传递实验索引
            date_train_days=date_train_days
        )
        
        train_data, val_data, test_data, train_labels, val_labels, test_labels, train_indices, val_indices, test_indices = result
        
        # 使用相同的索引划分timestamps和experiment_indices
        train_timestamps = timestamps[train_indices]
        test_timestamps = timestamps[test_indices]
        
        train_exp_indices = experiment_indices[train_indices]
        test_exp_indices = experiment_indices[test_indices]
        
        # 创建空的验证集（用于保持接口一致）
        val_timestamps = timestamps[:0]
        val_exp_indices = experiment_indices[:0]
    
    # 在非AR模式下，打乱训练/测试样本顺序，但仍在可视化时按时间戳恢复
    if not use_autoregressive:
        rng = np.random.default_rng(seed)

        def _shuffle_partition(name, data, labels, timestamps_arr, exp_indices_arr, split_indices_arr):
            if len(labels) == 0:
                return data, labels, timestamps_arr, exp_indices_arr, split_indices_arr

            perm = rng.permutation(len(labels))
            print(f"  - {name}集打乱顺序（AR未启用，后续可视化将按时间戳恢复）")

            data = data[perm]
            labels = labels[perm]

            if timestamps_arr is not None and len(timestamps_arr) == len(perm):
                timestamps_arr = timestamps_arr[perm]
            if exp_indices_arr is not None and len(exp_indices_arr) == len(perm):
                exp_indices_arr = exp_indices_arr[perm]
            if split_indices_arr is not None and len(split_indices_arr) == len(perm):
                split_indices_arr = split_indices_arr[perm]

            return data, labels, timestamps_arr, exp_indices_arr, split_indices_arr

        # 仅按需求打乱训练与测试集；验证集保持顺序便于诊断
        train_data, train_labels, train_timestamps, train_exp_indices, train_indices = _shuffle_partition(
            '训练', train_data, train_labels, train_timestamps, train_exp_indices, train_indices
        )
        test_data, test_labels, test_timestamps, test_exp_indices, test_indices = _shuffle_partition(
            '测试', test_data, test_labels, test_timestamps, test_exp_indices, test_indices
        )

    # 输出各数据集的血糖浓度范围和时间范围
    print(f"\n数据集血糖浓度范围和时间范围:")
    
    # 训练集血糖分析
    train_start_min, train_end_min = train_timestamps.min()/60, train_timestamps.max()/60
    train_start_idx = np.argmin(train_timestamps)  # 最早时间点的索引
    train_end_idx = np.argmax(train_timestamps)    # 最晚时间点的索引
    train_start_glucose = train_labels[train_start_idx]  # 起始血糖
    train_end_glucose = train_labels[train_end_idx]      # 结束血糖
    
    print(f"  训练集: 最低 {train_labels.min():.2f} - 最高 {train_labels.max():.2f} mmol/L (均值: {train_labels.mean():.2f} ± {train_labels.std():.2f})")
    print(f"          时间段: {train_start_min:.1f} - {train_end_min:.1f} 分钟 (时长: {train_end_min-train_start_min:.1f} 分钟)")
    print(f"          起始值: {train_start_glucose:.2f} → 结束值: {train_end_glucose:.2f} mmol/L (变化: {train_end_glucose-train_start_glucose:+.2f})")
    
    if use_val_set and len(val_labels) > 0:
        val_start_min, val_end_min = val_timestamps.min()/60, val_timestamps.max()/60
        val_start_idx = np.argmin(val_timestamps)
        val_end_idx = np.argmax(val_timestamps)
        val_start_glucose = val_labels[val_start_idx]
        val_end_glucose = val_labels[val_end_idx]
        
        print(f"  验证集: 最低 {val_labels.min():.2f} - 最高 {val_labels.max():.2f} mmol/L (均值: {val_labels.mean():.2f} ± {val_labels.std():.2f})")
        print(f"          时间段: {val_start_min:.1f} - {val_end_min:.1f} 分钟 (时长: {val_end_min-val_start_min:.1f} 分钟)")
        print(f"          起始值: {val_start_glucose:.2f} → 结束值: {val_end_glucose:.2f} mmol/L (变化: {val_end_glucose-val_start_glucose:+.2f})")
    
    # 测试集血糖分析
    test_start_min, test_end_min = test_timestamps.min()/60, test_timestamps.max()/60
    test_start_idx = np.argmin(test_timestamps)
    test_end_idx = np.argmax(test_timestamps)
    test_start_glucose = test_labels[test_start_idx]
    test_end_glucose = test_labels[test_end_idx]
    
    print(f"  测试集: 最低 {test_labels.min():.2f} - 最高 {test_labels.max():.2f} mmol/L (均值: {test_labels.mean():.2f} ± {test_labels.std():.2f})")
    print(f"          时间段: {test_start_min:.1f} - {test_end_min:.1f} 分钟 (时长: {test_end_min-test_start_min:.1f} 分钟)")
    print(f"          起始值: {test_start_glucose:.2f} → 结束值: {test_end_glucose:.2f} mmol/L (变化: {test_end_glucose-test_start_glucose:+.2f})")
    
    # 根据参数决定是否归一化（在划分后进行，避免数据泄漏）
    scaler = None
    if normalize:
        if lazy_window_active and base_features_2d is not None:
            print("  - 归一化特征 (懒窗口模式：在原始2D特征上归一化)")
            train_source_indices_for_norm = sample_source_indices[train_indices]
            _, scaler = loader.normalize_features(base_features_2d[train_source_indices_for_norm], scaler=None, fit=True)
            base_features_2d, _ = loader.normalize_features(base_features_2d, scaler=scaler, fit=False)
        else:
            print("  - 归一化特征 (使用训练集全局均值和标准差)")
            # 用训练集 fit scaler（计算全局均值和标准差）
            train_data, scaler = loader.normalize_features(train_data, scaler=None, fit=True)
            # 用训练集的 scaler transform 测试集（使用相同的全局统计量）
            test_data, _ = loader.normalize_features(test_data, scaler=scaler, fit=False)
            # 如果有验证集，也归一化（使用相同的全局统计量）
            if use_val_set:
                val_data, _ = loader.normalize_features(val_data, scaler=scaler, fit=False)
    else:
        print("  - 保留原始特征数据")

    # 划分完成后，释放无需保留的中间数组引用，避免与分割数组并存导致峰值内存过高
    del features
    gc.collect()
    
    # 创建 DataLoader
    print("创建 DataLoader...")
    
    # 对于窗口模式，每个样本已经包含了window_size个时间步的历史信息
    # 因此可以打乱样本顺序来增加训练多样性，避免过拟合到特定的时间段
    # 窗口内部的时序关系仍然保持完整
    should_shuffle = True  # 窗口模式下恢复打乱，因为时序信息已经在窗口内部
    
    # 判断是否使用Late Fusion（数据融合 + fusion_stage='late'）
    n_spectrum_features = metadata_list[0].get('n_spectrum_features', 1001) if metadata_list else 1001
    
    if use_late_fusion:
        print(f"  - Late Fusion模式: spectrum={n_spectrum_features}, aux={metadata_list[0].get('n_aux_features', 0)}")
    
    # 检查是否使用Auto-Regressive模式
    use_autoregressive = getattr(config, 'autoregressive', False)
    ar_glucose_history = getattr(config, 'ar_glucose_history', 5)
    ar_glucose_aligned = getattr(config, 'ar_glucose_aligned', True)
    
    if lazy_window_active and not use_autoregressive:
        print("  - 使用按需窗口Dataset（不预展开3D窗口）")
        train_source_indices = sample_source_indices[train_indices]
        test_source_indices = sample_source_indices[test_indices]
        val_source_indices = sample_source_indices[val_indices] if use_val_set and len(val_indices) > 0 else np.array([], dtype=np.int64)

        if use_late_fusion:
            train_dataset = LazyWindowLateFusionDataset(
                base_features_2d, train_source_indices, train_labels, window_size, n_spectrum_features
            )
            test_dataset = LazyWindowLateFusionDataset(
                base_features_2d, test_source_indices, test_labels, window_size, n_spectrum_features
            )
            val_dataset = LazyWindowLateFusionDataset(
                base_features_2d, val_source_indices, val_labels, window_size, n_spectrum_features
            ) if use_val_set and len(val_labels) > 0 else None
        else:
            train_dataset = LazyWindowGlucoseDataset(base_features_2d, train_source_indices, train_labels, window_size)
            test_dataset = LazyWindowGlucoseDataset(base_features_2d, test_source_indices, test_labels, window_size)
            val_dataset = LazyWindowGlucoseDataset(base_features_2d, val_source_indices, val_labels, window_size) if use_val_set and len(val_labels) > 0 else None

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            val_loader = None

    elif use_autoregressive:
        print(f"  - Auto-Regressive模式: 历史血糖数量={ar_glucose_history}, 对齐模式={'aligned' if ar_glucose_aligned else 'non-aligned'}")
        
        # 检查是否有窗口内glucose历史（时间窗口模式 + 下采样）
        if use_time_window and window_glucose_history_full is not None:
            print(f"    使用窗口内下采样后的glucose历史（与spectrum下采样对齐）")
            
            # 从窗口内glucose历史中提取最近ar_glucose_history个值
            # window_glucose_history_full shape: (n_samples, window_size)
            # 每个样本的glucose历史是窗口内下采样后的glucose值序列
            full_window_size = window_glucose_history_full.shape[1]
            
            if ar_glucose_history > full_window_size:
                print(f"    ⚠️  警告: ar_glucose_history ({ar_glucose_history}) > 窗口大小 ({full_window_size})")
                print(f"    自动调整为窗口大小: {full_window_size}")
                ar_glucose_history_effective = full_window_size
            else:
                ar_glucose_history_effective = ar_glucose_history
            
            # 划分前先提取glucose历史
            if ar_glucose_aligned:
                # aligned模式：不包含当前时刻，使用窗口内最后ar_glucose_history个值（不含最后一个）
                # 但窗口内最后一个值可能接近当前label，所以取倒数第2到第ar_glucose_history+1个
                full_glucose_history = window_glucose_history_full[:, -(ar_glucose_history_effective+1):-1]
            else:
                # non-aligned模式：包含当前时刻，使用窗口内最后ar_glucose_history个值
                full_glucose_history = window_glucose_history_full[:, -ar_glucose_history_effective:]
            
            print(f"    ✓ 从窗口内提取glucose历史: shape={full_glucose_history.shape}")
            
            # 按数据划分索引分配glucose历史
            train_glucose_history = full_glucose_history[train_indices]
            if use_val_set and len(val_indices) > 0:
                val_glucose_history = full_glucose_history[val_indices]
            else:
                val_glucose_history = np.array([])
            test_glucose_history = full_glucose_history[test_indices]
            
        else:
            # 传统方式：从labels创建历史（可能时间尺度不匹配）
            print(f"    使用传统方式创建glucose历史（基于labels）")
            train_glucose_history, _ = create_glucose_history(
                train_labels, train_exp_indices, ar_glucose_history, 
                ar_aligned=ar_glucose_aligned, padding_mode='edge'
            )
            
            if use_val_set and len(val_labels) > 0:
                val_glucose_history, _ = create_glucose_history(
                    val_labels, val_exp_indices, ar_glucose_history,
                    ar_aligned=ar_glucose_aligned, padding_mode='edge'
                )
            else:
                val_glucose_history = np.array([])  # 空数组
            
            test_glucose_history, _ = create_glucose_history(
                test_labels, test_exp_indices, ar_glucose_history,
                ar_aligned=ar_glucose_aligned, padding_mode='edge'
            )
        
        print(f"    ✓ 历史血糖序列创建完成: train={train_glucose_history.shape}, test={test_glucose_history.shape}")
        
        # 使用AR DataLoader
        if use_val_set:
            train_loader, val_loader, test_loader = create_ar_dataloaders(
                train_data, val_data, test_data,
                train_labels, val_labels, test_labels,
                train_glucose_history, val_glucose_history, test_glucose_history,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                mode=mode,
                shuffle_train=should_shuffle,
                has_aux=use_late_fusion,
                n_spectrum_features=n_spectrum_features
            )
        else:
            # 无验证集模式
            train_dataset = AutoRegressiveDataset(
                train_data, train_labels, train_glucose_history,
                n_spectrum_features=n_spectrum_features if use_late_fusion else None,
                mode=mode, has_aux=use_late_fusion
            )
            test_dataset = AutoRegressiveDataset(
                test_data, test_labels, test_glucose_history,
                n_spectrum_features=n_spectrum_features if use_late_fusion else None,
                mode=mode, has_aux=use_late_fusion
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=should_shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_loader = None
    elif use_val_set:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data,
            train_labels, val_labels, test_labels,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            mode=mode,
            shuffle_train=should_shuffle,
            late_fusion=use_late_fusion,  # 传递late fusion参数
            n_spectrum_features=n_spectrum_features
        )
    else:
        # 无验证集模式：只创建训练和测试loader
        if use_late_fusion:
            train_dataset = LateFusionDataset(train_data, train_labels, n_spectrum_features, mode=mode)
            test_dataset = LateFusionDataset(test_data, test_labels, n_spectrum_features, mode=mode)
        else:
            train_dataset = GlucoseDataset(train_data, train_labels, mode=mode)
            test_dataset = GlucoseDataset(test_data, test_labels, mode=mode)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=should_shuffle,  # 打乱以增加训练多样性
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = None  # 无验证集
    
    print(f"\n✓ 数据加载完成:")
    print(f"  训练集: {len(train_data)} 样本")
    if use_val_set:
        print(f"  验证集: {len(val_data)} 样本")
    else:
        print(f"  验证集: 无（使用 --no_val 模式）")
    print(f"  测试集: {len(test_data)} 样本")
    if mode == 'window' and not use_time_window and lazy_window_active and base_features_2d is not None:
        feature_dim = base_features_2d.shape[1]
    elif len(train_data.shape) >= 2:
        feature_dim = train_data.shape[-1]
    else:
        feature_dim = 0
    print(f"  特征维度: {feature_dim}")
    if use_late_fusion:
        print(f"  Late Fusion: spectrum={n_spectrum_features}, aux={feature_dim-n_spectrum_features}")
    print(f"  特征归一化: {'是' if normalize else '否（原始数据）'}")
    print(f"  血糖范围: {full_glucose.min():.2f} - {full_glucose.max():.2f} mmol/L")
    print("="*70 + "\n")
    
    # 生成 user_indices（根据experiment_indices映射）
    if metadata_list:
        experiment_to_user = np.array([m.get('user_index', 0) for m in metadata_list], dtype=np.int32)
        user_indices = experiment_to_user[experiment_indices] if experiment_indices is not None else None
        user_indices_full = experiment_to_user[full_experiment_indices] if full_experiment_indices is not None else None
        user_names_list = [m.get('user_name', 'Unknown') for m in metadata_list]
    else:
        user_indices = None
        user_indices_full = None
        user_names_list = []

    if lazy_window_active and sample_source_indices is not None:
        train_split_indices_meta = sample_source_indices[train_indices]
        test_split_indices_meta = sample_source_indices[test_indices]
    else:
        train_split_indices_meta = train_indices
        test_split_indices_meta = test_indices

    def _collect_dates(exp_indices_arr):
        if exp_indices_arr is None or metadata_list is None or len(exp_indices_arr) == 0:
            return []
        date_set = {}
        for exp_id in np.unique(exp_indices_arr):
            exp_id_int = int(exp_id)
            if exp_id_int < 0 or exp_id_int >= len(metadata_list):
                continue
            date_label, sort_key = _extract_date_info_from_metadata(metadata_list[exp_id_int])
            date_set[date_label] = sort_key
        return [d for d, _ in sorted(date_set.items(), key=lambda kv: kv[1])]

    train_dates_selected = _collect_dates(train_exp_indices if 'train_exp_indices' in locals() else None)
    val_dates_selected = _collect_dates(val_exp_indices if 'val_exp_indices' in locals() else None)
    test_dates_selected = _collect_dates(test_exp_indices if 'test_exp_indices' in locals() else None)

    train_db_file_indices = db_file_indices[train_indices] if db_file_temporal_mode and db_file_indices is not None else None
    test_db_file_indices = db_file_indices[test_indices] if db_file_temporal_mode and db_file_indices is not None else None

    if split_strategy in ['date', 'date_random', 'max_day_train_min_day_test']:
        print("\n日期划分记录:")
        print(f"  训练日期: {train_dates_selected}")
        if len(val_dates_selected) > 0:
            print(f"  验证日期: {val_dates_selected}")
        print(f"  测试日期: {test_dates_selected}")

    # 构建训练集和测试集的元数据（用于可视化）
    train_metadata = {
        'timestamps': train_timestamps,
        'experiment_names': [meta['experiment_name'] for meta in metadata_list] if metadata_list else [],
        'start_times': [meta.get('start_time', 'Unknown') for meta in metadata_list] if metadata_list else [],
        'experiment_indices': train_exp_indices if 'train_exp_indices' in locals() else None,
        'user_names': user_names_list,
        'user_indices': user_indices[train_indices] if user_indices is not None else None,
        'db_file_indices': train_db_file_indices,
        'db_file_names': db_file_id_to_name if db_file_temporal_mode else {},
        'db_file_paths': db_file_id_to_path if db_file_temporal_mode else {},
        'split_indices': train_split_indices_meta,  # 添加原始索引用于gap检测
        'selected_dates': train_dates_selected,
    }
    
    test_metadata = {
        'timestamps': test_timestamps,
        'experiment_names': [meta['experiment_name'] for meta in metadata_list] if metadata_list else [],
        'start_times': [meta.get('start_time', 'Unknown') for meta in metadata_list] if metadata_list else [],
        'experiment_indices': test_exp_indices if 'test_exp_indices' in locals() else None,
        'user_names': user_names_list,
        'user_indices': user_indices[test_indices] if user_indices is not None else None,
        'db_file_indices': test_db_file_indices,
        'db_file_names': db_file_id_to_name if db_file_temporal_mode else {},
        'db_file_paths': db_file_id_to_path if db_file_temporal_mode else {},
        'split_indices': test_split_indices_meta,  # 添加原始索引用于gap检测
        'selected_dates': test_dates_selected,
        'train_dates': train_dates_selected,
        'val_dates': val_dates_selected,
    }

    if split_strategy == 'db_file_temporal_80_20' and test_db_file_indices is not None:
        per_db_file_test_indices = {}
        for db_id in np.unique(test_db_file_indices):
            db_id_int = int(db_id)
            db_name = db_file_id_to_name.get(db_id_int, f"db_file_{db_id_int}")
            per_db_file_test_indices[db_name] = np.where(test_db_file_indices == db_id)[0]
        test_metadata['per_db_file_test_indices'] = per_db_file_test_indices

    if split_strategy == 'multi_user_independent' and test_metadata.get('user_indices') is not None:
        user_index_to_name = {}
        for meta in metadata_list:
            idx = int(meta.get('user_index', 0))
            user_index_to_name.setdefault(idx, meta.get('user_name', f'user_{idx}'))

        per_user_test_indices = {}
        unique_test_users = np.unique(test_metadata['user_indices'])
        for user_idx in unique_test_users:
            user_name = user_index_to_name.get(int(user_idx), f'user_{int(user_idx)}')
            per_user_test_indices[user_name] = np.where(test_metadata['user_indices'] == user_idx)[0]

        test_metadata['per_user_test_indices'] = per_user_test_indices
    
    # 构建完整数据集的元数据（用于在可视化中显示完整的ground truth）
    full_metadata = {
        'timestamps': full_timestamps,
        'glucose': full_glucose,
        'experiment_names': [meta['experiment_name'] for meta in metadata_list] if metadata_list else [],
        'start_times': [meta.get('start_time', 'Unknown') for meta in metadata_list] if metadata_list else [],
        'experiment_indices': full_experiment_indices,
        'user_names': user_names_list,
        'user_indices': user_indices_full,
        'db_file_names': db_file_id_to_name if db_file_temporal_mode else {},
        'db_file_paths': db_file_id_to_path if db_file_temporal_mode else {},
    }
    
    if normalize:
        return train_loader, val_loader, test_loader, scaler, train_metadata, test_metadata, full_metadata
    else:
        return train_loader, val_loader, test_loader, train_metadata, test_metadata, full_metadata

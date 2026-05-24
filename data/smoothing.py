"""
数据平滑处理模块
提供多种平滑方法来处理频谱数据的时间序列抖动和异常值

⚠️ 重要：时间轴平滑使用因果滤波器（只使用过去的数据）避免数据泄露
频谱轴平滑可以使用双向滤波（频谱维度不涉及时序关系）

支持的平滑方法：
1. 移动平均 (Moving Average) - 因果版本
2. 指数移动平均 (Exponential Moving Average, EMA) - 天然因果
3. 高斯平滑 (Gaussian Smoothing) - 因果版本
4. Savitzky-Golay滤波 (保留峰值特征) - 因果版本
5. 中值滤波 (Median Filter, 去除脉冲噪声) - 因果版本
6. 小波去噪 (Wavelet Denoising) - 需要额外处理保证因果性
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple


class DataSmoother:
    """数据平滑器 - 处理频谱时间序列的抖动和异常值"""
    
    def __init__(self, method: str = 'none', **kwargs):
        """
        Args:
            method: 平滑方法
                - 'none': 不进行平滑
                - 'moving_average': 移动平均
                - 'ema': 指数移动平均
                - 'gaussian': 高斯平滑
                - 'savgol': Savitzky-Golay滤波
                - 'median': 中值滤波
                - 'wavelet': 小波去噪
            **kwargs: 各方法的参数
                - window_size: 窗口大小 (moving_average, median)
                - alpha: EMA平滑系数 (ema), 0-1之间，越小越平滑
                - sigma: 高斯标准差 (gaussian)
                - poly_order: 多项式阶数 (savgol)
                - wavelet: 小波类型 (wavelet)
                - threshold_scale: 阈值缩放因子 (wavelet)
        """
        self.method = method.lower()
        self.params = kwargs
        
        # 默认参数
        self.default_params = {
            'moving_average': {'window_size': 5},
            'ema': {'alpha': 0.3},
            'gaussian': {'sigma': 2.0},
            'savgol': {'window_size': 11, 'poly_order': 3},
            'median': {'window_size': 5},
            'wavelet': {'wavelet': 'db4', 'threshold_scale': 1.0}
        }
        
    def smooth(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        对数据进行平滑处理
        
        Args:
            data: 输入数据，shape (n_samples, n_features)
            axis: 平滑的轴，0表示时间轴，1表示频谱轴
            
        Returns:
            平滑后的数据，shape与输入相同
        """
        if self.method == 'none':
            return data
        
        # 获取方法参数
        params = self.default_params.get(self.method, {}).copy()
        params.update(self.params)
        
        # 根据方法调用对应的平滑函数
        if self.method == 'moving_average':
            return self._moving_average(data, axis, **params)
        elif self.method == 'ema':
            return self._exponential_moving_average(data, axis, **params)
        elif self.method == 'gaussian':
            return self._gaussian_smooth(data, axis, **params)
        elif self.method == 'savgol':
            return self._savitzky_golay(data, axis, **params)
        elif self.method == 'median':
            return self._median_filter(data, axis, **params)
        elif self.method == 'wavelet':
            return self._wavelet_denoise(data, axis, **params)
        else:
            raise ValueError(f"未知的平滑方法: {self.method}")
    
    def _moving_average(self, data: np.ndarray, axis: int, window_size: int = 5) -> np.ndarray:
        """移动平均平滑 - 简单有效，适合去除高频噪声
        
        ⚠️ 时间轴平滑使用因果滤波器（只使用过去的数据），避免数据泄露
        频谱轴平滑可以使用双向滤波（频谱维度不涉及时序关系）
        """
        if window_size < 2:
            return data
        
        smoothed = data.copy()
        
        if axis == 0:  # 时间轴平滑 - 使用因果滤波器（只看过去）
            # 对每个特征（频率bin）独立平滑
            for i in range(data.shape[1]):
                # 因果移动平均：只使用当前和过去的数据
                for t in range(data.shape[0]):
                    start_idx = max(0, t - window_size + 1)
                    smoothed[t, i] = np.mean(data[start_idx:t+1, i])
        else:  # 频谱轴平滑 - 使用双向滤波（频谱维度无时序关系）
            # 确保窗口大小为奇数
            if window_size % 2 == 0:
                window_size += 1
            # 对每个时间点的频谱独立平滑
            for i in range(data.shape[0]):
                smoothed[i, :] = np.convolve(data[i, :], 
                                            np.ones(window_size) / window_size, 
                                            mode='same')
        
        return smoothed
    
    def _exponential_moving_average(self, data: np.ndarray, axis: int, alpha: float = 0.3) -> np.ndarray:
        """指数移动平均 - 对最近数据赋予更高权重，响应更快"""
        if not 0 < alpha <= 1:
            raise ValueError("alpha必须在(0, 1]之间")
        
        smoothed = data.copy()
        
        if axis == 0:  # 时间轴平滑
            for i in range(data.shape[1]):
                for t in range(1, data.shape[0]):
                    smoothed[t, i] = alpha * data[t, i] + (1 - alpha) * smoothed[t-1, i]
        else:  # 频谱轴平滑
            for t in range(data.shape[0]):
                for i in range(1, data.shape[1]):
                    smoothed[t, i] = alpha * data[t, i] + (1 - alpha) * smoothed[t, i-1]
        
        return smoothed
    
    def _gaussian_smooth(self, data: np.ndarray, axis: int, sigma: float = 2.0) -> np.ndarray:
        """高斯平滑 - 保持边缘特征，平滑效果好
        
        ⚠️ 时间轴平滑使用因果高斯滤波器（只使用过去的数据）
        频谱轴平滑可以使用双向滤波
        """
        if sigma <= 0:
            return data
        
        smoothed = data.copy()
        
        if axis == 0:  # 时间轴平滑 - 使用因果高斯滤波
            # 对每个特征独立平滑
            for i in range(data.shape[1]):
                # 因果高斯滤波：使用truncate只保留左侧（过去）
                # truncate=4表示截断点在4*sigma处
                smoothed[:, i] = gaussian_filter1d(data[:, i], sigma=sigma, 
                                                   mode='constant', cval=0.0, 
                                                   truncate=4.0)
                # 手动实现因果版本：对每个时间点，只使用过去的数据
                for t in range(data.shape[0]):
                    # 计算因果高斯窗口的权重
                    window_size = min(t + 1, int(4 * sigma))
                    if window_size > 0:
                        indices = np.arange(max(0, t - window_size + 1), t + 1)
                        weights = np.exp(-0.5 * ((t - indices) / sigma) ** 2)
                        weights /= weights.sum()
                        smoothed[t, i] = np.sum(data[indices, i] * weights)
        else:  # 频谱轴平滑 - 使用双向高斯滤波
            for t in range(data.shape[0]):
                smoothed[t, :] = gaussian_filter1d(data[t, :], sigma=sigma)
        
        return smoothed
    
    def _savitzky_golay(self, data: np.ndarray, axis: int, 
                       window_size: int = 11, poly_order: int = 3) -> np.ndarray:
        """Savitzky-Golay滤波 - 保留峰值和边缘特征，适合频谱数据
        
        ⚠️ 时间轴平滑使用因果SG滤波器（只使用过去的数据）
        频谱轴平滑可以使用双向滤波
        """
        # 确保窗口大小大于多项式阶数
        if window_size <= poly_order:
            window_size = poly_order + 2
        
        # 确保窗口不超过数据长度
        if axis == 0:
            window_size = min(window_size, data.shape[0])
        else:
            window_size = min(window_size, data.shape[1])
        
        if window_size < poly_order + 2:
            return data  # 数据太短，无法应用滤波
        
        smoothed = data.copy()
        
        try:
            if axis == 0:  # 时间轴平滑 - 使用因果SG滤波
                for i in range(data.shape[1]):
                    # 因果SG滤波：使用mode='interp'和pos参数
                    # pos=window_size-1 表示窗口在当前点左侧（只使用过去数据）
                    for t in range(data.shape[0]):
                        start_idx = max(0, t - window_size + 1)
                        end_idx = t + 1
                        actual_window = end_idx - start_idx
                        
                        if actual_window >= poly_order + 2:
                            # 对窗口内的数据进行SG滤波
                            window_data = data[start_idx:end_idx, i]
                            # 使用mode='nearest'避免边界效应，取最后一个点（当前点）
                            filtered = savgol_filter(window_data, 
                                                    min(actual_window if actual_window % 2 == 1 else actual_window - 1, window_size),
                                                    poly_order, 
                                                    mode='nearest')
                            smoothed[t, i] = filtered[-1]  # 取滤波后的最后一个点
                        else:
                            # 窗口太小，使用简单平均
                            smoothed[t, i] = np.mean(data[start_idx:end_idx, i])
            else:  # 频谱轴平滑 - 使用双向SG滤波
                # 确保窗口大小为奇数
                if window_size % 2 == 0:
                    window_size += 1
                for t in range(data.shape[0]):
                    smoothed[t, :] = savgol_filter(data[t, :], window_size, poly_order)
        except Exception as e:
            print(f"  ⚠ Savitzky-Golay滤波失败: {e}，返回原始数据")
            return data
        
        return smoothed
    
    def _median_filter(self, data: np.ndarray, axis: int, window_size: int = 5) -> np.ndarray:
        """中值滤波 - 有效去除脉冲噪声和异常值
        
        ⚠️ 时间轴平滑使用因果中值滤波器（只使用过去的数据）
        频谱轴平滑可以使用双向滤波
        """
        if window_size < 2:
            return data
        
        smoothed = data.copy()
        
        if axis == 0:  # 时间轴平滑 - 使用因果中值滤波
            # 对每个特征（频率bin）独立平滑
            for i in range(data.shape[1]):
                # 因果中值滤波：只使用当前和过去的数据
                for t in range(data.shape[0]):
                    start_idx = max(0, t - window_size + 1)
                    smoothed[t, i] = np.median(data[start_idx:t+1, i])
        else:  # 频谱轴平滑 - 使用双向中值滤波
            # 确保窗口大小为奇数
            if window_size % 2 == 0:
                window_size += 1
            for t in range(data.shape[0]):
                smoothed[t, :] = medfilt(data[t, :], kernel_size=window_size)
        
        return smoothed
    
    def _wavelet_denoise(self, data: np.ndarray, axis: int, 
                        wavelet: str = 'db4', threshold_scale: float = 1.0) -> np.ndarray:
        """小波去噪 - 多尺度分析，适合复杂噪声"""
        try:
            import pywt
        except ImportError:
            print("  ⚠ PyWavelets未安装，跳过小波去噪。请运行: pip install PyWavelets")
            return data
        
        smoothed = data.copy()
        
        def denoise_signal(signal, wavelet, threshold_scale):
            """对单个信号进行小波去噪"""
            # 小波分解
            coeffs = pywt.wavedec(signal, wavelet, level=None)
            
            # 计算阈值（使用MAD估计噪声标准差）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * threshold_scale * np.sqrt(2 * np.log(len(signal)))
            
            # 软阈值处理
            coeffs_thresh = [coeffs[0]]  # 保留低频部分
            for c in coeffs[1:]:
                coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
            
            # 重构信号
            return pywt.waverec(coeffs_thresh, wavelet)
        
        try:
            if axis == 0:  # 时间轴平滑
                for i in range(data.shape[1]):
                    denoised = denoise_signal(data[:, i], wavelet, threshold_scale)
                    # 处理长度不匹配的情况
                    smoothed[:, i] = denoised[:data.shape[0]]
            else:  # 频谱轴平滑
                for t in range(data.shape[0]):
                    denoised = denoise_signal(data[t, :], wavelet, threshold_scale)
                    smoothed[t, :] = denoised[:data.shape[1]]
        except Exception as e:
            print(f"  ⚠ 小波去噪失败: {e}，返回原始数据")
            return data
        
        return smoothed


class OutlierDetector:
    """异常值检测和处理"""
    
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0, 
                               axis: int = 0) -> np.ndarray:
        """
        使用Z-score方法检测异常值
        
        Args:
            data: 输入数据 (n_samples, n_features)
            threshold: Z-score阈值，通常取3.0
            axis: 检测的轴
            
        Returns:
            布尔数组，True表示异常值
        """
        if axis == 0:  # 时间轴检测
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-8))
            outliers = z_scores > threshold
        else:  # 频谱轴检测
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            z_scores = np.abs((data - mean) / (std + 1e-8))
            outliers = z_scores > threshold
        
        return outliers
    
    @staticmethod
    def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5, 
                           axis: int = 0) -> np.ndarray:
        """
        使用IQR (Interquartile Range) 方法检测异常值
        
        Args:
            data: 输入数据 (n_samples, n_features)
            factor: IQR倍数，通常取1.5
            axis: 检测的轴
            
        Returns:
            布尔数组，True表示异常值
        """
        if axis == 0:  # 时间轴检测
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outliers = (data < lower) | (data > upper)
        else:  # 频谱轴检测
            q1 = np.percentile(data, 25, axis=1, keepdims=True)
            q3 = np.percentile(data, 75, axis=1, keepdims=True)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outliers = (data < lower) | (data > upper)
        
        return outliers
    
    @staticmethod
    def handle_outliers(data: np.ndarray, outliers: np.ndarray, 
                       method: str = 'interpolate') -> np.ndarray:
        """
        处理检测到的异常值
        
        Args:
            data: 输入数据
            outliers: 异常值掩码
            method: 处理方法
                - 'interpolate': 线性插值
                - 'median': 用中位数替换
                - 'mean': 用均值替换
                - 'clip': 截断到合理范围
                
        Returns:
            处理后的数据
        """
        data_clean = data.copy()
        
        if method == 'interpolate':
            # 对每个特征独立处理
            for i in range(data.shape[1]):
                col_outliers = outliers[:, i]
                if np.any(col_outliers):
                    # 获取正常值的索引
                    normal_idx = np.where(~col_outliers)[0]
                    outlier_idx = np.where(col_outliers)[0]
                    
                    if len(normal_idx) > 1:
                        # 线性插值
                        data_clean[outlier_idx, i] = np.interp(
                            outlier_idx, normal_idx, data[normal_idx, i]
                        )
        
        elif method == 'median':
            for i in range(data.shape[1]):
                col_outliers = outliers[:, i]
                if np.any(col_outliers):
                    median_val = np.median(data[~col_outliers, i])
                    data_clean[col_outliers, i] = median_val
        
        elif method == 'mean':
            for i in range(data.shape[1]):
                col_outliers = outliers[:, i]
                if np.any(col_outliers):
                    mean_val = np.mean(data[~col_outliers, i])
                    data_clean[col_outliers, i] = mean_val
        
        elif method == 'clip':
            for i in range(data.shape[1]):
                col_outliers = outliers[:, i]
                if np.any(col_outliers):
                    normal_data = data[~col_outliers, i]
                    q1 = np.percentile(normal_data, 25)
                    q3 = np.percentile(normal_data, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    data_clean[:, i] = np.clip(data[:, i], lower, upper)
        
        return data_clean


def apply_smoothing_pipeline(data: np.ndarray, 
                             time_smooth_method: str = 'none',
                             time_smooth_params: Optional[dict] = None,
                             spectrum_smooth_method: str = 'none',
                             spectrum_smooth_params: Optional[dict] = None,
                             detect_outliers: bool = False,
                             outlier_method: str = 'zscore',
                             outlier_threshold: float = 3.0,
                             handle_outliers: bool = False,
                             outlier_handle_method: str = 'interpolate') -> Tuple[np.ndarray, dict]:
    """
    完整的数据平滑处理流程
    
    Args:
        data: 输入数据 (n_samples, n_features)
        time_smooth_method: 时间轴平滑方法
        time_smooth_params: 时间轴平滑参数
        spectrum_smooth_method: 频谱轴平滑方法
        spectrum_smooth_params: 频谱轴平滑参数
        detect_outliers: 是否检测异常值
        outlier_method: 异常值检测方法 ('zscore' 或 'iqr')
        outlier_threshold: 异常值阈值
        handle_outliers: 是否处理异常值
        outlier_handle_method: 异常值处理方法
        
    Returns:
        (smoothed_data, info_dict)
    """
    info = {}
    smoothed_data = data.copy()
    
    # 1. 异常值检测和处理
    if detect_outliers:
        detector = OutlierDetector()
        
        if outlier_method == 'zscore':
            outliers = detector.detect_outliers_zscore(smoothed_data, outlier_threshold, axis=0)
        elif outlier_method == 'iqr':
            outliers = detector.detect_outliers_iqr(smoothed_data, 1.5, axis=0)
        else:
            outliers = np.zeros_like(smoothed_data, dtype=bool)
        
        n_outliers = np.sum(outliers)
        outlier_ratio = n_outliers / smoothed_data.size
        info['outliers_detected'] = n_outliers
        info['outlier_ratio'] = outlier_ratio
        
        if handle_outliers and n_outliers > 0:
            smoothed_data = detector.handle_outliers(smoothed_data, outliers, outlier_handle_method)
            info['outliers_handled'] = True
    
    # 2. 时间轴平滑
    if time_smooth_method != 'none':
        smoother = DataSmoother(time_smooth_method, **(time_smooth_params or {}))
        smoothed_data = smoother.smooth(smoothed_data, axis=0)
        info['time_smoothing'] = time_smooth_method
    
    # 3. 频谱轴平滑
    if spectrum_smooth_method != 'none':
        smoother = DataSmoother(spectrum_smooth_method, **(spectrum_smooth_params or {}))
        smoothed_data = smoother.smooth(smoothed_data, axis=1)
        info['spectrum_smoothing'] = spectrum_smooth_method
    
    return smoothed_data, info

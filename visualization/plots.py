"""
可视化绘图模块
所有图表保存为文件，适合服务器环境
"""

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .metrics import (
    calculate_all_metrics,
    print_metrics_summary,
    calculate_clarke_zone_points,
)

# 设置 matplotlib 参数，防止大数据量绘图时的内存溢出
matplotlib.rcParams['agg.path.chunksize'] = 10000

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def _labels_to_numpy(labels):
    """将Dataset中的labels转为一维numpy数组，避免为了取标签而构造完整窗口batch。"""
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def _extract_labels_from_dataset(dataset):
    """优先从Dataset/Subset/ConcatDataset直接取labels，失败时返回None走DataLoader兜底。"""
    try:
        from torch.utils.data import ConcatDataset, Subset
    except Exception:
        ConcatDataset = ()
        Subset = ()

    if ConcatDataset and isinstance(dataset, ConcatDataset):
        label_parts = []
        for child in dataset.datasets:
            child_labels = _extract_labels_from_dataset(child)
            if child_labels is None:
                return None
            label_parts.append(child_labels)
        return np.concatenate(label_parts).reshape(-1) if label_parts else np.array([])

    if Subset and isinstance(dataset, Subset):
        base_labels = _extract_labels_from_dataset(dataset.dataset)
        if base_labels is None:
            return None
        return base_labels[np.asarray(dataset.indices)].reshape(-1)

    if hasattr(dataset, 'labels'):
        return _labels_to_numpy(dataset.labels).copy()

    return None


def plot_prediction_vs_actual(y_true, y_pred, save_path, title="Prediction vs Actual"):
    """
    绘制预测值 vs 实际值的对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none')
    
    # 完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # ±20% 范围线
    margin = 0.2
    ax.plot([min_val, max_val], [min_val*(1-margin), max_val*(1-margin)], 
            'g--', linewidth=1, alpha=0.5, label='±20%')
    ax.plot([min_val, max_val], [min_val*(1+margin), max_val*(1+margin)], 
            'g--', linewidth=1, alpha=0.5)
    
    # 计算指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    ax.set_xlabel('Actual Glucose (mmol/L)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mmol/L)', fontsize=12)
    ax.set_title(f'{title}\nMAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_error_distribution(y_true, y_pred, save_path, title="Error Distribution"):
    """
    绘制误差分布图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 标题
    """
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 误差直方图
    ax = axes[0, 0]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(errors.mean(), color='g', linestyle='--', linewidth=2, 
               label=f'Mean={errors.mean():.3f}')
    ax.set_xlabel('Prediction Error (mmol/L)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Histogram', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 绝对误差直方图
    ax = axes[0, 1]
    abs_errors = np.abs(errors)
    ax.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(abs_errors.mean(), color='r', linestyle='--', linewidth=2,
               label=f'MAE={abs_errors.mean():.3f}')
    ax.set_xlabel('Absolute Error (mmol/L)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Absolute Error Histogram', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q 图
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. 误差箱线图
    ax = axes[1, 1]
    ax.boxplot([errors], labels=['Errors'], vert=True)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_ylabel('Prediction Error (mmol/L)', fontsize=11)
    ax.set_title('Error Box Plot', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Mean: {errors.mean():.3f}\n'
    stats_text += f'Std: {errors.std():.3f}\n'
    stats_text += f'Median: {np.median(errors):.3f}\n'
    stats_text += f'MAE: {abs_errors.mean():.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_scatter_with_metrics(y_true, y_pred, save_path, title="Scatter Plot with Metrics", training_info=None):
    """
    带详细指标的散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 标题
        training_info: 训练配置信息（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 计算误差并上色
    errors = np.abs(y_pred - y_true)
    scatter = ax.scatter(y_true, y_pred, c=errors, cmap='YlOrRd', 
                        s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, 
            label='Perfect Prediction', zorder=5)
    
    # 添加色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error (mmol/L)', fontsize=11)
    
    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    # 添加指标文本（左上角）
    metrics_text = f'MAE: {mae:.4f} mmol/L\n'
    metrics_text += f'RMSE: {rmse:.4f} mmol/L\n'
    metrics_text += f'R²: {r2:.4f}\n'
    metrics_text += f'Correlation: {corr:.4f}\n'
    metrics_text += f'Samples: {len(y_true)}'
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 添加训练配置信息（右上角）
    if training_info is not None:
        info_lines = []
        # 模型信息
        if 'model_name' in training_info:
            model_str = f"Model: {training_info['model_name']}"
            if 'hidden_size' in training_info:
                model_str += f" (H:{training_info['hidden_size']}"
                if 'num_layers' in training_info:
                    model_str += f",L:{training_info['num_layers']}"
                model_str += ")"
            info_lines.append(model_str)
        
        # 训练超参数
        if 'learning_rate' in training_info:
            lr_str = f"LR: {training_info['learning_rate']}"
            if 'optimizer' in training_info:
                lr_str += f" ({training_info['optimizer']})"
            info_lines.append(lr_str)
        
        if 'batch_size' in training_info:
            batch_str = f"Batch: {training_info['batch_size']}"
            if 'epochs' in training_info:
                if 'best_epoch' in training_info:
                    batch_str += f" | Epochs: {training_info['best_epoch']}/{training_info['epochs']}"
                else:
                    batch_str += f" | Epochs: {training_info['epochs']}"
            info_lines.append(batch_str)
        
        if 'loss_function' in training_info:
            loss_str = f"Loss: {training_info['loss_function']}"
            if 'scheduler' in training_info:
                loss_str += f" | Sched: {training_info['scheduler']}"
            info_lines.append(loss_str)
        
        # 数据配置
        if 'mode' in training_info:
            mode_str = f"Mode: {training_info['mode']}"
            if 'window_duration' in training_info:
                mode_str += f" ({training_info['window_duration']})"
            elif 'window_size' in training_info:
                mode_str += f" (W:{training_info['window_size']})"
            info_lines.append(mode_str)
        
        data_parts = []
        if 'downsample_interval' in training_info:
            data_parts.append(f"DS:{training_info['downsample_interval']}")
        if 'normalize' in training_info:
            data_parts.append(f"Norm:{training_info['normalize']}")
        if data_parts:
            info_lines.append(' | '.join(data_parts))
        
        # Split策略和细节
        if 'split_strategy' in training_info:
            split_str = f"Split: {training_info['split_strategy']}"
            if 'n_splits' in training_info and training_info['split_strategy'] == 'alternating':
                split_str += f" (n={training_info['n_splits']})"
            if 'data_split' in training_info:
                split_str += f" [{training_info['data_split']}]"
            info_lines.append(split_str)
        
        if info_lines:
            info_text = '\n'.join(info_lines)
            ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=8, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5))
    
    ax.set_xlabel('Actual Glucose (mmol/L)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mmol/L)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_residual_analysis(y_true, y_pred, save_path, title="Residual Analysis"):
    """
    残差分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 标题
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 残差 vs 预测值
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values (mmol/L)', fontsize=11)
    ax.set_ylabel('Residuals (mmol/L)', fontsize=11)
    ax.set_title('Residuals vs Predicted', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. 残差 vs 真实值
    ax = axes[0, 1]
    ax.scatter(y_true, residuals, alpha=0.5, s=20, color='orange')
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual Values (mmol/L)', fontsize=11)
    ax.set_ylabel('Residuals (mmol/L)', fontsize=11)
    ax.set_title('Residuals vs Actual', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. 标准化残差
    ax = axes[1, 0]
    std_residuals = residuals / residuals.std()
    ax.scatter(y_pred, std_residuals, alpha=0.5, s=20, color='green')
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.axhline(2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Predicted Values (mmol/L)', fontsize=11)
    ax.set_ylabel('Standardized Residuals', fontsize=11)
    ax.set_title('Standardized Residuals', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Scale-Location plot
    ax = axes[1, 1]
    sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))
    ax.scatter(y_pred, sqrt_abs_std_resid, alpha=0.5, s=20, color='purple')
    # 添加平滑曲线
    from scipy.ndimage import gaussian_filter1d
    sort_idx = np.argsort(y_pred)
    smoothed = gaussian_filter1d(sqrt_abs_std_resid[sort_idx], sigma=10)
    ax.plot(y_pred[sort_idx], smoothed, 'r-', linewidth=2)
    ax.set_xlabel('Predicted Values (mmol/L)', fontsize=11)
    ax.set_ylabel('√|Standardized Residuals|', fontsize=11)
    ax.set_title('Scale-Location Plot', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def plot_error_over_time(y_true, y_pred, save_path, title="Error Over Time"):
    """
    绘制误差随样本序列的变化（时间序列分析）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 标题
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 预测值 vs 真实值随时间变化
    ax = axes[0]
    indices = np.arange(len(y_true))
    ax.plot(indices, y_true, 'b-', alpha=0.7, linewidth=1, label='Actual')
    ax.plot(indices, y_pred, 'r-', alpha=0.7, linewidth=1, label='Predicted')
    ax.fill_between(indices, y_true, y_pred, alpha=0.3, color='gray')
    ax.set_ylabel('Glucose (mmol/L)', fontsize=11)
    ax.set_title('Predictions vs Actual Over Sample Index', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 误差随时间变化
    ax = axes[1]
    ax.plot(indices, errors, 'g-', alpha=0.7, linewidth=1)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.fill_between(indices, 0, errors, alpha=0.3, color='green')
    ax.set_ylabel('Error (mmol/L)', fontsize=11)
    ax.set_title('Prediction Error Over Sample Index', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. 绝对误差和滚动均值
    ax = axes[2]
    ax.plot(indices, abs_errors, 'orange', alpha=0.5, linewidth=1, label='Absolute Error')
    # 计算滚动均值
    window = min(100, len(abs_errors) // 10)
    rolling_mean = np.convolve(abs_errors, np.ones(window)/window, mode='valid')
    ax.plot(indices[:len(rolling_mean)], rolling_mean, 'r-', linewidth=2, 
            label=f'Rolling Mean (window={window})')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Absolute Error (mmol/L)', fontsize=11)
    ax.set_title('Absolute Error with Rolling Average', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def _draw_clarke_error_grid_boundaries(ax, max_mgdl=400):
    """绘制与现有分区条件对应的 Clarke Error Grid 边界线。"""
    # 参考线
    ax.plot([0, max_mgdl], [0, max_mgdl], 'k-', linewidth=1.2, alpha=0.8, label='y=x')

    # Zone A: ±20%
    x20 = np.linspace(70, max_mgdl, 500)
    ax.plot(x20, 1.2 * x20, 'k--', linewidth=1.0, alpha=0.8)
    ax.plot(x20, 0.8 * x20, 'k--', linewidth=1.0, alpha=0.8)

    # 低血糖、高血糖关键分界
    ax.axvline(70, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axhline(70, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axvline(180, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axhline(180, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axvline(240, color='gray', linestyle=':', linewidth=0.9, alpha=0.7)

    # Zone C 边界（与分区判定完全一致）
    x_c_upper = np.linspace(70, 290, 300)
    ax.plot(x_c_upper, x_c_upper + 110, 'k-.', linewidth=1.0, alpha=0.8)

    x_c_lower = np.linspace(130, 180, 200)
    ax.plot(x_c_lower, (7 / 5) * x_c_lower - 182, 'k-.', linewidth=1.0, alpha=0.8)


def plot_clarke_error_grid(y_true, y_pred, save_path, title="Clarke Error Grid"):
    """
    绘制 Clarke Error Grid：显示每个区域中的点位分布。

    坐标单位采用 mg/dL，与 Clarke 标准一致。
    """
    zone_points = calculate_clarke_zone_points(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    max_mgdl = 400

    _draw_clarke_error_grid_boundaries(ax=ax, max_mgdl=max_mgdl)

    zone_colors = {
        'A': '#2ca02c',
        'B': '#1f77b4',
        'C': '#ff7f0e',
        'D': '#d62728',
        'E': '#9467bd',
    }

    total_samples = len(np.asarray(y_true).reshape(-1))
    for zone in ['A', 'B', 'C', 'D', 'E']:
        points = zone_points[zone]
        if points['count'] == 0:
            continue

        ax.scatter(
            points['true_mgdl'],
            points['pred_mgdl'],
            s=16,
            alpha=0.65,
            color=zone_colors[zone],
            edgecolors='none',
            label=f"Zone {zone}: {points['count']} ({points['percent']:.2f}%)"
        )

    ax.set_xlim(0, max_mgdl)
    ax.set_ylim(0, max_mgdl)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'{title}\nTotal Samples: {total_samples}', fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # 区域文字提示
    ax.text(45, 40, 'A', fontsize=12, weight='bold', alpha=0.65)
    ax.text(320, 250, 'B', fontsize=12, weight='bold', alpha=0.65)
    ax.text(105, 295, 'C', fontsize=12, weight='bold', alpha=0.65)
    ax.text(265, 120, 'D', fontsize=12, weight='bold', alpha=0.65)
    ax.text(40, 300, 'E', fontsize=12, weight='bold', alpha=0.65)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def save_clarke_zone_points(y_true, y_pred, save_path):
    """保存 Clarke 各区域点位明细到 CSV。"""
    zone_points = calculate_clarke_zone_points(y_true, y_pred)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'zone',
            'actual_mmol_per_l',
            'predicted_mmol_per_l',
            'actual_mg_per_dl',
            'predicted_mg_per_dl',
        ])

        for zone in ['A', 'B', 'C', 'D', 'E']:
            true_mmol = zone_points[zone]['true_mmol']
            pred_mmol = zone_points[zone]['pred_mmol']
            true_mgdl = zone_points[zone]['true_mgdl']
            pred_mgdl = zone_points[zone]['pred_mgdl']

            for i in range(zone_points[zone]['count']):
                writer.writerow([
                    zone,
                    float(true_mmol[i]),
                    float(pred_mmol[i]),
                    float(true_mgdl[i]),
                    float(pred_mgdl[i]),
                ])

    print(f"  ✓ 保存: {save_path}")


def plot_ground_truth_glucose(y_true, metadata, save_path, title_prefix="Ground Truth"):
    """
    绘制插值后的ground truth血糖浓度随时间变化
    
    Args:
        y_true: 真实血糖值（插值后）
        metadata: 包含时间戳和实验信息的字典
            - timestamps: 时间戳数组（秒）
            - experiment_names: 实验名称列表
            - start_times: 起始时间列表
        save_path: 保存路径
        title_prefix: 标题前缀
    """
    from datetime import datetime
    
    # 从metadata提取信息
    timestamps = metadata.get('timestamps', None)
    experiment_names = metadata.get('experiment_names', [])
    start_times = metadata.get('start_times', [])
    
    # ⚠️ 防止数据点过多导致 matplotlib OverflowError
    MAX_POINTS = 30000  # matplotlib 绘图的安全上限
    original_length = len(y_true)
    
    if original_length > MAX_POINTS:
        print(f"  ⚠️ 数据点过多 ({original_length}), 下采样到 {MAX_POINTS} 个点以避免内存溢出")
        # 均匀采样索引
        indices = np.linspace(0, original_length - 1, MAX_POINTS, dtype=int)
        y_true = y_true[indices]
        if timestamps is not None:
            timestamps = timestamps[indices]
    
    # 如果没有时间戳，创建序号作为x轴
    if timestamps is None:
        timestamps = np.arange(len(y_true))
        time_label = 'Sample Index'
        time_unit = ''
    else:
        # 转换为分钟
        timestamps = timestamps / 60.0  # 秒 -> 分钟
        time_label = 'Time (minutes)'
        time_unit = 'min'
    
    # 计算时间范围
    time_start = timestamps[0] if len(timestamps) > 0 else 0
    time_end = timestamps[-1] if len(timestamps) > 0 else len(y_true)
    time_duration = time_end - time_start
    
    # 提取数据集信息
    if len(experiment_names) > 0:
        if len(experiment_names) == 1:
            dataset_info = experiment_names[0]
        elif len(experiment_names) <= 3:
            dataset_info = ', '.join(experiment_names)
        else:
            dataset_info = f"{len(experiment_names)} experiments"
    else:
        dataset_info = "Unknown dataset"
    
    # 提取起始时间信息
    if len(start_times) > 0:
        time_info = f"Start: {start_times[0]}"
        if len(start_times) > 1:
            time_info += f" | {len(start_times)} sessions"
    else:
        time_info = ""
    
    # 计算血糖统计
    glucose_mean = np.mean(y_true)
    glucose_std = np.std(y_true)
    glucose_min = np.min(y_true)
    glucose_max = np.max(y_true)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 6))
    
    # 绘制血糖值
    plt.plot(timestamps, y_true, 'b-', linewidth=1.5, label='Interpolated Glucose', alpha=0.8)
    
    # 添加均值线
    plt.axhline(glucose_mean, color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {glucose_mean:.2f} mmol/L', alpha=0.7)
    
    # 添加±1std区域
    plt.fill_between(timestamps, glucose_mean - glucose_std, glucose_mean + glucose_std,
                     alpha=0.2, color='green', label=f'±1 STD: {glucose_std:.2f}')
    
    # 设置标签和标题
    plt.xlabel(time_label, fontsize=12)
    plt.ylabel('Glucose Concentration (mmol/L)', fontsize=12)
    
    # 构建详细的标题
    title_lines = [
        f'{title_prefix}: Interpolated Blood Glucose Ground Truth',
        f'Dataset: {dataset_info} | {time_info}',
        f'Duration: {time_duration:.1f} {time_unit} ({time_start:.1f} - {time_end:.1f} {time_unit}) | '
        f'Samples: {original_length}' + 
        (f' (displayed: {len(y_true)})' if original_length > len(y_true) else '') +
        f' | Range: {glucose_min:.2f} - {glucose_max:.2f} mmol/L | '
        f'Mean: {glucose_mean:.2f} ± {glucose_std:.2f} mmol/L'
    ]
    plt.title('\n'.join(title_lines), fontsize=11, pad=15)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 设置y轴范围（留出一些边距）
    y_margin = (glucose_max - glucose_min) * 0.1
    plt.ylim(glucose_min - y_margin, glucose_max + y_margin)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {save_path}")


def generate_ground_truth_visualization(data_loader, metadata, save_dir='./results', title_suffix=''):
    """
    生成 ground truth 血糖浓度可视化（用于检查插值质量）
    
    Args:
        data_loader: 数据加载器
        metadata: 元数据字典
            - timestamps: 时间戳数组（秒）
            - experiment_names: 实验名称列表
            - start_times: 起始时间列表
        save_dir: 保存目录（默认 './results'）
        title_suffix: 标题后缀（例如 'Test Set' 或 'Train Set'）
    """
    import os
    from datetime import datetime
    
    # 从dataset直接提取标签，避免完整遍历lazy-window特征导致测试后内存峰值暴涨。
    y_true = _extract_labels_from_dataset(data_loader.dataset)
    if y_true is None:
        all_labels = []
        for batch in data_loader:
            # 支持多种数据集格式：
            # - Normal: (data, labels) - 2个值
            # - Late Fusion: (spectrum, aux, labels) - 3个值
            # - AutoRegressive: (spectrum, glucose_history, labels) - 3个值
            # - AutoRegressive + Aux: (spectrum, aux, glucose_history, labels) - 4个值
            if len(batch) == 4:
                labels = batch[3]
            elif len(batch) == 3:
                labels = batch[2]
            elif len(batch) == 2:
                labels = batch[1]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            all_labels.append(_labels_to_numpy(labels).copy())
        y_true = np.concatenate(all_labels).flatten()
        del all_labels
    
    # 确定保存路径
    experiment_names = metadata.get('experiment_names', [])
    if len(experiment_names) == 1:
        subfolder = experiment_names[0]
    elif len(experiment_names) > 1:
        subfolder = f"{experiment_names[0]}_and_{len(experiment_names)-1}_more"
    else:
        subfolder = 'default'
    
    viz_dir = os.path.join(save_dir, subfolder)
    os.makedirs(viz_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 根据title_suffix生成不同的文件名
    if title_suffix:
        suffix_tag = title_suffix.replace(' ', '_').lower()
        filename = f'ground_truth_{suffix_tag}_{timestamp_str}.png'
        title = f"{title_suffix}: Ground Truth"
    else:
        filename = f'ground_truth_interpolated_{timestamp_str}.png'
        title = "Full Dataset"
    filepath = os.path.join(viz_dir, filename)
    
    # 生成可视化
    plot_ground_truth_glucose(
        y_true=y_true,
        metadata=metadata,
        save_path=filepath,
        title_prefix=title
    )
    
    return filepath


def plot_time_series_comparison(y_true, y_pred, metadata, save_path, title_prefix="Test", training_info=None, full_metadata=None):
    """
    绘制真实值与预测值随时间变化的对比图（带详细元数据）
    
    Args:
        y_true: 真实血糖值（当前数据集：train或test）
        y_pred: 预测血糖值（当前数据集：train或test）
        metadata: 包含时间戳和实验信息的字典（当前数据集）
            - timestamps: 时间戳数组（秒）
            - experiment_names: 实验名称列表
            - start_times: 起始时间列表
            - split_indices: 数据在原始序列中的索引（用于alternating模式的gap检测）
        save_path: 保存路径
        title_prefix: 标题前缀
        training_info: 训练信息字典（可选）
            - epochs: 训练轮数
            - learning_rate: 学习率
            - batch_size: 批次大小
            - model_name: 模型名称
            - window_size: 窗口大小
            - best_epoch: 最佳epoch
            - split_strategy: 数据划分策略（如'alternating'）
        full_metadata: 完整数据集的metadata（用于显示完整ground truth）
            - timestamps: 完整时间戳数组
            - glucose: 完整ground truth数组
            如果提供，则显示完整的ground truth，而预测值只显示当前数据集（train或test）
    """
    from datetime import datetime
    
    # 从metadata提取信息
    timestamps = metadata.get('timestamps', None)
    experiment_names = metadata.get('experiment_names', [])
    start_times = metadata.get('start_times', [])
    split_indices = metadata.get('split_indices', None)
    
    # 如果提供了full_metadata，准备完整的ground truth数据用于显示
    # 这样可以在train/test图中都显示完整的ground truth，但预测值只显示对应数据集
    full_y_true = None
    full_timestamps = None
    if full_metadata is not None:
        full_y_true = full_metadata.get('glucose', None)
        full_timestamps = full_metadata.get('timestamps', None)
        if full_y_true is not None and full_timestamps is not None:
            # 确保是numpy数组
            full_y_true = np.array(full_y_true)
            full_timestamps = np.array(full_timestamps)
    
    # ⚠️ 防止数据点过多导致matplotlib "Exceeded cell block limit" 错误
    MAX_POINTS = 30000  # matplotlib绘图的安全上限
    original_length = len(y_true)
    
    if original_length > MAX_POINTS:
        print(f"  ⚠️ 数据点过多 ({original_length}), 下采样到 {MAX_POINTS} 个点以避免内存溢出")
        # 均匀采样索引
        indices = np.linspace(0, original_length - 1, MAX_POINTS, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        if timestamps is not None:
            timestamps = timestamps[indices]
        
        # 重要：如果有split_indices，需要将其映射到下采样后的新索引
        if split_indices is not None:
            # 创建原始索引到新索引的映射字典
            old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
            # 将split_indices映射到新的索引空间
            new_split_indices = []
            for old_idx in split_indices:
                if old_idx in old_to_new_map:
                    new_split_indices.append(old_to_new_map[old_idx])
            split_indices = np.array(new_split_indices) if len(new_split_indices) > 0 else None
    
    # 同样对full数据进行下采样（如果有）
    if full_y_true is not None and full_timestamps is not None:
        full_original_length = len(full_y_true)
        if full_original_length > MAX_POINTS:
            full_indices = np.linspace(0, full_original_length - 1, MAX_POINTS, dtype=int)
            full_y_true = full_y_true[full_indices]
            full_timestamps = full_timestamps[full_indices]
    
    # 如果没有时间戳，创建序号作为x轴
    if timestamps is None:
        timestamps = np.arange(len(y_true))
        time_label = 'Sample Index'
        time_unit = ''
    else:
        # 转换为分钟
        timestamps = timestamps / 60.0  # 秒 -> 分钟
        time_label = 'Time (minutes)'
        time_unit = 'min'
    
    # 同样转换full_timestamps（如果有）
    if full_timestamps is not None:
        full_timestamps = full_timestamps / 60.0  # 秒 -> 分钟
    
    # 计算时间范围
    # 如果有full数据，使用full数据的时间范围；否则使用当前数据集的时间范围
    if full_timestamps is not None:
        time_start = full_timestamps[0] if len(full_timestamps) > 0 else 0
        time_end = full_timestamps[-1] if len(full_timestamps) > 0 else len(full_y_true)
    else:
        time_start = timestamps[0] if len(timestamps) > 0 else 0
        time_end = timestamps[-1] if len(timestamps) > 0 else len(y_true)
    time_duration = time_end - time_start
    
    # 提取数据集信息
    if len(experiment_names) > 0:
        if len(experiment_names) == 1:
            dataset_info = experiment_names[0]
        elif len(experiment_names) <= 3:
            dataset_info = ', '.join(experiment_names)
        else:
            dataset_info = f"{len(experiment_names)} experiments"
    else:
        dataset_info = "Unknown dataset"
    
    # 提取起始时间信息
    if len(start_times) > 0:
        time_info = f"Start: {start_times[0]}"
        if len(start_times) > 1:
            time_info += f" | {len(start_times)} sessions"
    else:
        time_info = ""
    
    # 计算误差统计
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 创建图形
    fig = plt.figure(figsize=(16, 7))  # 增加高度以容纳训练信息
    ax = plt.gca()
    
    # 检查是否为alternating模式，如果是则需要特殊处理gap
    is_alternating = (training_info is not None and 
                     training_info.get('split_strategy') == 'alternating')
    
    if is_alternating and split_indices is not None:
        # Alternating模式：检测segment之间的gap
        # 通过检查split_indices的连续性来识别gap
        gaps = []
        gap_ranges = []  # 记录gap的时间范围
        
        for i in range(len(split_indices) - 1):
            # 如果相邻两个数据点的原始索引差距大于1，说明中间有gap
            if split_indices[i+1] - split_indices[i] > 1:
                gaps.append(i)
                # 记录gap的时间范围（从当前点到下一个点之间）
                gap_ranges.append((timestamps[i], timestamps[i+1]))
        
        # 为当前数据集的segments添加背景色
        segment_start = 0
        first_data_bg = True
        for gap_idx in gaps + [len(timestamps)]:
            segment_end = gap_idx if gap_idx < len(timestamps) else len(timestamps)
            
            # 添加当前数据集的背景色（蓝色系）
            if segment_end > segment_start:
                data_label = f'{title_prefix} Set Data' if first_data_bg else None
                ax.axvspan(timestamps[segment_start], timestamps[segment_end-1], 
                          alpha=0.3, color='cornflowerblue', label=data_label, zorder=0)
                first_data_bg = False
            
            segment_start = segment_end
        
        # 为gap区域（对方数据集）添加不同颜色背景
        if len(gap_ranges) > 0:
            # 根据title_prefix判断当前是测试集还是训练集
            is_test_set = 'Test' in title_prefix or 'test' in title_prefix.lower()
            gap_color = 'lightgreen' if is_test_set else 'wheat'
            gap_label_text = 'Train Set Data (Gap)' if is_test_set else 'Test Set Data (Gap)'
            
            first_gap_bg = True
            for gap_start, gap_end in gap_ranges:
                gap_label = gap_label_text if first_gap_bg else None
                ax.axvspan(gap_start, gap_end, 
                          alpha=0.35, color=gap_color, label=gap_label, zorder=0)
                first_gap_bg = False
        
        # Ground truth绘制
        # 如果有full_metadata，显示完整的ground truth；否则只显示当前数据集的ground truth
        if full_y_true is not None and full_timestamps is not None:
            plt.plot(full_timestamps, full_y_true, 'b-', linewidth=1.5, 
                    label='True Glucose (Complete)', alpha=0.7)
        else:
            plt.plot(timestamps, y_true, 'b-', linewidth=1.5, 
                    label='True Glucose', alpha=0.7)
        
        # 预测值分段绘制（不跨越gap）- 只显示当前数据集的预测
        segment_start = 0
        first_pred_segment = True
        first_gap_line = True
        
        for gap_idx in gaps + [len(timestamps)]:  # 添加结尾位置
            segment_end = gap_idx if gap_idx < len(timestamps) else len(timestamps)
            
            # 绘制当前segment的预测值
            segment_times = timestamps[segment_start:segment_end]
            segment_pred = y_pred[segment_start:segment_end]
            
            # 只在第一个segment显示label
            pred_label = 'Predicted Glucose' if first_pred_segment else None
            
            plt.plot(segment_times, segment_pred, 'r-', linewidth=1.5, 
                    label=pred_label, alpha=0.7)
            
            first_pred_segment = False
            
            # 如果不是最后一个segment，绘制预测值跨越gap的虚线连接
            if gap_idx < len(timestamps):
                gap_label = 'Gap Connection (Predicted)' if first_gap_line else None
                # 使用灰色虚线连接gap两端的预测值
                plt.plot([timestamps[gap_idx], timestamps[gap_idx+1]], 
                        [y_pred[gap_idx], y_pred[gap_idx+1]], 
                        'gray', linestyle='--', linewidth=1.0, alpha=0.4, label=gap_label)
                first_gap_line = False
            
            segment_start = segment_end
        
        # 填充误差区域（需要分段填充，因为预测值是分段的）
        segment_start = 0
        first_error_area = True
        for gap_idx in gaps + [len(timestamps)]:
            segment_end = gap_idx if gap_idx < len(timestamps) else len(timestamps)
            segment_times = timestamps[segment_start:segment_end]
            segment_true = y_true[segment_start:segment_end]
            segment_pred = y_pred[segment_start:segment_end]
            
            error_label = 'Prediction Error' if first_error_area else None
            plt.fill_between(segment_times, segment_true, segment_pred, 
                           alpha=0.2, color='gray', label=error_label)
            first_error_area = False
            segment_start = segment_end
    else:
        # 非alternating模式：正常绘制连续曲线
        # Ground truth: 如果有full_metadata，显示完整的；否则只显示当前数据集的
        if full_y_true is not None and full_timestamps is not None:
            plt.plot(full_timestamps, full_y_true, 'b-', linewidth=1.5, 
                    label='True Glucose (Complete)', alpha=0.7)
        else:
            plt.plot(timestamps, y_true, 'b-', linewidth=1.5, label='True Glucose', alpha=0.7)
        
        # 预测值：只显示当前数据集的预测
        plt.plot(timestamps, y_pred, 'r-', linewidth=1.5, label='Predicted Glucose', alpha=0.7)
        
        # 填充误差区域：只在当前数据集的范围内
        plt.fill_between(timestamps, y_true, y_pred, alpha=0.2, color='gray', label='Prediction Error')
    
    # 设置标签和标题
    plt.xlabel(time_label, fontsize=12)
    plt.ylabel('Glucose Concentration (mmol/L)', fontsize=12)
    
    # 构建详细的标题
    title_lines = [
        f'{title_prefix} Set: Blood Glucose Prediction - True vs Predicted',
        f'Dataset: {dataset_info} | {time_info}',
        f'Duration: {time_duration:.1f} {time_unit} ({time_start:.1f} - {time_end:.1f} {time_unit}) | Samples: {original_length}' + 
        (f' (displayed: {len(y_true)})' if original_length > len(y_true) else '') +
        f' | MAE: {mae:.3f} | RMSE: {rmse:.3f} mmol/L'
    ]
    
    # 添加训练配置到标题（如果有）
    if training_info is not None:
        config_parts = []
        if 'model_name' in training_info:
            config_parts.append(f"Model:{training_info['model_name']}")
        if 'learning_rate' in training_info:
            config_parts.append(f"LR:{training_info['learning_rate']}")
        if 'batch_size' in training_info:
            config_parts.append(f"BS:{training_info['batch_size']}")
        if 'optimizer' in training_info:
            config_parts.append(f"Opt:{training_info['optimizer']}")
        if 'loss_function' in training_info:
            config_parts.append(f"Loss:{training_info['loss_function']}")
        if 'epochs' in training_info and 'best_epoch' in training_info:
            config_parts.append(f"Ep:{training_info['best_epoch']}/{training_info['epochs']}")
        
        # Split信息
        if 'split_strategy' in training_info:
            split_part = f"Split:{training_info['split_strategy']}"
            if 'n_splits' in training_info and training_info['split_strategy'] == 'alternating':
                split_part += f"(n={training_info['n_splits']})"
            config_parts.append(split_part)
        
        if config_parts:
            title_lines.append(' | '.join(config_parts))
    
    plt.title('\n'.join(title_lines), fontsize=10, pad=15)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 设置y轴范围（留出一些边距）
    # 如果有full数据，y轴范围应该包含完整数据的范围
    if full_y_true is not None:
        y_min = min(full_y_true.min(), y_pred.min())
        y_max = max(full_y_true.max(), y_pred.max())
    else:
        y_min = min(y_true.min(), y_pred.min())
        y_max = max(y_true.max(), y_pred.max())
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 添加训练信息文本框（右上角）
    if training_info is not None:
        info_lines = []
        
        # 模型配置
        if 'model_name' in training_info:
            model_str = f"Model: {training_info['model_name']}"
            if 'hidden_size' in training_info:
                model_str += f" (H:{training_info['hidden_size']}"
                if 'num_layers' in training_info:
                    model_str += f", L:{training_info['num_layers']}"
                model_str += ")"
            info_lines.append(model_str)
        
        # 训练配置
        train_config_parts = []
        if 'epochs' in training_info:
            if 'best_epoch' in training_info:
                train_config_parts.append(f"Ep:{training_info['epochs']}(Best:{training_info['best_epoch']})")
            else:
                train_config_parts.append(f"Ep:{training_info['epochs']}")
        if 'learning_rate' in training_info:
            train_config_parts.append(f"LR:{training_info['learning_rate']}")
        if 'batch_size' in training_info:
            train_config_parts.append(f"BS:{training_info['batch_size']}")
        if 'dropout' in training_info:
            train_config_parts.append(f"Drop:{training_info['dropout']}")
        if train_config_parts:
            info_lines.append(' | '.join(train_config_parts))
        
        # 损失函数和优化
        opt_parts = []
        if 'loss_function' in training_info:
            opt_parts.append(f"Loss:{training_info['loss_function']}")
        if 'attention' in training_info:
            opt_parts.append(f"Attn:{training_info['attention']}")
        if 'aggregation' in training_info:
            opt_parts.append(f"Agg:{training_info['aggregation']}")
        if opt_parts:
            info_lines.append(' | '.join(opt_parts))
        
        # 数据配置
        data_config_parts = []
        if 'mode' in training_info:
            mode_str = f"Mode:{training_info['mode']}"
            if 'window_duration' in training_info:
                mode_str += f"({training_info['window_duration']})"
            elif 'window_size' in training_info:
                mode_str += f"({training_info['window_size']})"
            data_config_parts.append(mode_str)
        if 'downsample_interval' in training_info:
            data_config_parts.append(f"DS:{training_info['downsample_interval']}")
        if 'normalize' in training_info:
            data_config_parts.append(f"Norm:{training_info['normalize']}")
        if data_config_parts:
            info_lines.append(' | '.join(data_config_parts))
        
        # 数据划分策略和细节
        if 'split_strategy' in training_info:
            split_str = f"Split:{training_info['split_strategy']}"
            if 'n_splits' in training_info and training_info['split_strategy'] == 'alternating':
                split_str += f" (n_splits={training_info['n_splits']})"
            if 'data_split' in training_info:
                split_str += f" ({training_info['data_split']})"
            info_lines.append(split_str)
        
        if info_lines:
            info_text = '\n'.join(info_lines)
            # 添加文本框到图表外部右侧，完全避免遮挡数据
            plt.text(1.01, 0.98, info_text,
                    transform=plt.gca().transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5),
                    family='monospace')
    
    # 紧凑布局，bbox_inches='tight'会自动包含图表外的文本
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {save_path}")


def create_evaluation_report(y_true, y_pred, save_dir, prefix="test", training_info=None):
    """
    创建完整的评估报告（包含所有可视化图表）
    
    Args:
        y_true: 真实值
        y_pred: 预测值  
        save_dir: 保存目录
        prefix: 文件名前缀
        training_info: 训练信息字典（可选）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"生成评估报告 - {prefix}")
    print(f"{'='*70}\n")
    
    # 计算所有指标
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics_summary(metrics, title=f"{prefix.upper()} Set Metrics")
    
    # 生成所有可视化图表
    print(f"\n生成可视化图表...")
    
    plot_prediction_vs_actual(
        y_true, y_pred, 
        save_dir / f"{prefix}_prediction_vs_actual.png",
        title=f"{prefix.upper()} Set: Prediction vs Actual"
    )
    
    plot_error_distribution(
        y_true, y_pred,
        save_dir / f"{prefix}_error_distribution.png",
        title=f"{prefix.upper()} Set: Error Distribution"
    )
    
    plot_scatter_with_metrics(
        y_true, y_pred,
        save_dir / f"{prefix}_scatter_metrics.png",
        title=f"{prefix.upper()} Set: Scatter Plot with Metrics",
        training_info=training_info
    )
    
    plot_residual_analysis(
        y_true, y_pred,
        save_dir / f"{prefix}_residual_analysis.png",
        title=f"{prefix.upper()} Set: Residual Analysis"
    )
    
    plot_error_over_time(
        y_true, y_pred,
        save_dir / f"{prefix}_error_over_time.png",
        title=f"{prefix.upper()} Set: Error Over Time"
    )

    plot_clarke_error_grid(
        y_true,
        y_pred,
        save_dir / f"{prefix}_clarke_error_grid.png",
        title=f"{prefix.upper()} Set: Clarke Error Grid"
    )

    clarke_points_csv = save_dir / f"{prefix}_clarke_zone_points.csv"
    save_clarke_zone_points(y_true, y_pred, clarke_points_csv)
    
    # 保存指标到文件
    metrics_file = save_dir / f"{prefix}_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"{prefix.upper()} Set Evaluation Metrics\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"基础指标:\n")
        f.write(f"  MAE:  {metrics['mae']:.4f} mmol/L\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f} mmol/L\n")
        f.write(f"  R²:   {metrics['r2']:.4f}\n")
        f.write(f"  相关系数: {metrics['pearson_corr']:.4f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n\n")
        
        f.write(f"误差统计:\n")
        f.write(f"  平均误差: {metrics['mean_error']:.4f} ± {metrics['std_error']:.4f}\n")
        f.write(f"  中位数误差: {metrics['median_error']:.4f}\n")
        f.write(f"  最大/最小误差: {metrics['max_error']:.4f} / {metrics['min_error']:.4f}\n\n")
        
        f.write(f"Clarke Error Grid:\n")
        f.write(f"  Zone A: {metrics['clarke_zone_a']:.2f}%\n")
        f.write(f"  Zone B: {metrics['clarke_zone_b']:.2f}%\n")
        f.write(f"  Zone C: {metrics['clarke_zone_c']:.2f}%\n")
        f.write(f"  Zone D: {metrics['clarke_zone_d']:.2f}%\n")
        f.write(f"  Zone E: {metrics['clarke_zone_e']:.2f}%\n")
        f.write(f"  点位明细CSV: {clarke_points_csv.name}\n")
        f.write(f"  可视化图片: {prefix}_clarke_error_grid.png\n\n")
        
        f.write(f"样本数: {metrics['n_samples']}\n")
    
    print(f"  ✓ 保存: {metrics_file}")
    
    print(f"\n{'='*70}")
    print(f"✓ 评估报告生成完成！")
    print(f"  保存位置: {save_dir}")
    print(f"{'='*70}\n")
    
    return metrics


def plot_time_series_by_experiment(y_true, y_pred, metadata, save_dir, title_prefix="Test", training_info=None, full_metadata=None):
    """
    为每个实验分别生成时间序列对比图
    当有多个实验时，这个函数会生成：
    1. 每个实验的单独图（清晰展示各实验的时序）
    2. 所有实验的合并图（展示整体效果）
    
    Args:
        y_true: 真实血糖值（当前数据集：train或test）
        y_pred: 预测血糖值（当前数据集：train或test）
        metadata: 包含时间戳和实验信息的字典（当前数据集）
            - timestamps: 时间戳数组（秒）
            - experiment_names: 实验名称列表
            - experiment_indices: 每个样本对应的实验索引
            - start_times: 起始时间列表
        save_dir: 保存目录
        title_prefix: 标题前缀
        training_info: 训练信息字典
        full_metadata: 完整数据集的metadata（包含所有ground truth）
            - timestamps: 完整时间戳数组
            - glucose: 完整ground truth数组
            - experiment_indices: 完整实验索引数组
            其他字段同metadata
    
    Returns:
        list: 生成的所有图片路径
    """
    from datetime import datetime
    import os
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamps = metadata.get('timestamps', None)
    experiment_names = metadata.get('experiment_names', [])
    experiment_indices = metadata.get('experiment_indices', None)
    
    saved_paths = []
    
    # 如果没有实验索引或只有一个实验，直接生成一张总图
    if experiment_indices is None or len(np.unique(experiment_indices)) <= 1:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = save_dir / f'{title_prefix.lower()}_time_series_comparison_{timestamp_str}.png'
        plot_time_series_comparison(y_true, y_pred, metadata, filepath, title_prefix, training_info, full_metadata)
        return [filepath]
    
    # 有多个实验：为每个实验生成单独的图
    unique_exp_indices = np.unique(experiment_indices)
    
    print(f"\n{'='*70}")
    print(f"为每个实验生成时间序列对比图 ({len(unique_exp_indices)} 个实验)")
    print(f"{'='*70}\n")
    
    # 获取split_indices（如果有）
    split_indices = metadata.get('split_indices', None)
    
    for exp_idx in unique_exp_indices:
        # 获取当前实验的数据
        exp_mask = experiment_indices == exp_idx
        exp_y_true = y_true[exp_mask]
        exp_y_pred = y_pred[exp_mask]
        
        # 获取实验名称
        if exp_idx < len(experiment_names):
            exp_name = experiment_names[int(exp_idx)]
        else:
            exp_name = f"Experiment_{int(exp_idx)}"
        
        # 提取当前实验的时间戳和split_indices
        exp_timestamps = timestamps[exp_mask] if timestamps is not None else None
        exp_split_indices = split_indices[exp_mask] if split_indices is not None else None
        
        # 🔑 关键修复：对当前实验的数据按时间戳重新排序
        # 这对于random/stratified split特别重要，因为提取后的数据顺序不是时间顺序
        if exp_timestamps is not None and len(exp_timestamps) > 0:
            exp_time_sort_idx = np.argsort(exp_timestamps)
            exp_y_true = exp_y_true[exp_time_sort_idx]
            exp_y_pred = exp_y_pred[exp_time_sort_idx]
            exp_timestamps = exp_timestamps[exp_time_sort_idx]
            if exp_split_indices is not None:
                exp_split_indices = exp_split_indices[exp_time_sort_idx]
        
        # 构建单个实验的metadata
        exp_metadata = {
            'timestamps': exp_timestamps,
            'experiment_names': [exp_name],
            'start_times': [metadata.get('start_times', [''])[int(exp_idx)]] if metadata.get('start_times') else [],
            'split_indices': exp_split_indices,
        }
        
        # 生成文件名
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_exp_name = exp_name.replace('/', '_').replace('\\', '_')
        filepath = save_dir / f'{title_prefix.lower()}_time_series_{safe_exp_name}_{timestamp_str}.png'
        
        # 绘制单个实验的图
        # 如果有full_metadata，提取该实验的完整ground truth
        exp_full_metadata = None
        if full_metadata is not None:
            full_exp_indices = full_metadata.get('experiment_indices', None)
            if full_exp_indices is not None:
                full_exp_mask = full_exp_indices == exp_idx
                exp_full_metadata = {
                    'timestamps': full_metadata.get('timestamps', None)[full_exp_mask] if full_metadata.get('timestamps') is not None else None,
                    'glucose': full_metadata.get('glucose', None)[full_exp_mask] if full_metadata.get('glucose') is not None else None,
                    'experiment_names': [exp_name],
                    'start_times': [metadata.get('start_times', [''])[int(exp_idx)]] if metadata.get('start_times') else [],
                }
        
        plot_time_series_comparison(
            exp_y_true, 
            exp_y_pred, 
            exp_metadata, 
            filepath, 
            f"{title_prefix} ({exp_name})", 
            training_info,
            exp_full_metadata
        )
        
        saved_paths.append(filepath)
    
    # 同时生成一张合并所有实验的图
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_filepath = save_dir / f'{title_prefix.lower()}_time_series_all_combined_{timestamp_str}.png'
    
    print(f"\n生成合并所有实验的时间序列图...")
    plot_time_series_comparison(
        y_true, 
        y_pred, 
        metadata, 
        combined_filepath, 
        f"{title_prefix} (All Experiments)", 
        training_info,
        full_metadata
    )
    saved_paths.append(combined_filepath)
    
    print(f"\n{'='*70}")
    print(f"✓ 已为 {len(unique_exp_indices)} 个实验分别生成时间序列图")
    print(f"  + 1 张合并图")
    print(f"{'='*70}\n")
    
    return saved_paths

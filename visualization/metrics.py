"""
评估指标计算模块
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


MMOL_TO_MGDL = 18.0182


def _classify_clarke_zone_mg(true_mg, pred_mg):
    """使用现有规则将单个样本划分到 Clarke Error Grid 区域。"""
    if (true_mg < 70 and pred_mg < 70) or abs(true_mg - pred_mg) < 0.2 * true_mg:
        return 'A'

    if true_mg <= 70 and pred_mg >= 180:
        return 'E'

    if true_mg >= 180 and pred_mg <= 70:
        return 'E'

    if true_mg >= 240 and 70 <= pred_mg <= 180:
        return 'D'

    if true_mg <= 70 and 70 <= pred_mg <= 180:
        return 'D'

    if 70 <= true_mg <= 290 and pred_mg >= true_mg + 110:
        return 'C'

    if 130 <= true_mg <= 180 and pred_mg <= (7 / 5) * true_mg - 182:
        return 'C'

    return 'B'


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值 array
        y_pred: 预测值 array
        
    Returns:
        dict: 包含所有指标的字典
    """
    # 基础指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 相关系数
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    
    # 误差统计
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 百分比误差
    mape = np.mean(np.abs(errors / (y_true + 1e-8))) * 100  # 避免除零
    
    # Clarke Error Grid 分区（血糖监测标准）
    zone_a, zone_b, zone_c, zone_d, zone_e = calculate_clarke_zones(y_true, y_pred)
    
    metrics = {
        # 基础指标
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'mape': mape,
        
        # 误差统计
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        
        # 误差分位数
        'q25_error': np.percentile(abs_errors, 25),
        'q50_error': np.percentile(abs_errors, 50),
        'q75_error': np.percentile(abs_errors, 75),
        'q95_error': np.percentile(abs_errors, 95),
        
        # Clarke Error Grid
        'clarke_zone_a': zone_a,
        'clarke_zone_b': zone_b,
        'clarke_zone_c': zone_c,
        'clarke_zone_d': zone_d,
        'clarke_zone_e': zone_e,
        
        # 样本数
        'n_samples': len(y_true),
    }
    
    return metrics


def calculate_clarke_zones(y_true, y_pred):
    """
    计算 Clarke Error Grid 各区域的样本百分比
    Clarke Error Grid 是血糖监测的临床准确性评估标准
    
    Zone A: 临床准确 (20%以内误差)
    Zone B: 临床可接受 (不会导致错误治疗)
    Zone C: 可能导致不必要的治疗
    Zone D: 可能无法检测到危险状况
    Zone E: 可能导致相反的治疗
    """
    n = len(y_true)
    zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    
    for true_val, pred_val in zip(y_true, y_pred):
        # 转换为 mg/dL (Clarke Grid 标准单位)
        true_mg = true_val * MMOL_TO_MGDL
        pred_mg = pred_val * MMOL_TO_MGDL
        zone = _classify_clarke_zone_mg(true_mg, pred_mg)
        zone_counts[zone] += 1
    
    # 转换为百分比
    return (
        zone_counts['A'] / n * 100,
        zone_counts['B'] / n * 100,
        zone_counts['C'] / n * 100,
        zone_counts['D'] / n * 100,
        zone_counts['E'] / n * 100
    )


def calculate_clarke_zone_points(y_true, y_pred):
    """
    统计每个 Clarke 区域中的点位坐标（真实值、预测值）。

    Returns:
        dict: {
            'A'|'B'|'C'|'D'|'E': {
                'true_mmol': np.ndarray,
                'pred_mmol': np.ndarray,
                'true_mgdl': np.ndarray,
                'pred_mgdl': np.ndarray,
                'count': int,
                'percent': float,
            }
        }
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = len(y_true)
    zone_buckets = {
        z: {
            'true_mmol': [],
            'pred_mmol': [],
            'true_mgdl': [],
            'pred_mgdl': [],
        }
        for z in ['A', 'B', 'C', 'D', 'E']
    }

    for true_val, pred_val in zip(y_true, y_pred):
        true_mg = float(true_val) * MMOL_TO_MGDL
        pred_mg = float(pred_val) * MMOL_TO_MGDL
        zone = _classify_clarke_zone_mg(true_mg, pred_mg)

        zone_buckets[zone]['true_mmol'].append(float(true_val))
        zone_buckets[zone]['pred_mmol'].append(float(pred_val))
        zone_buckets[zone]['true_mgdl'].append(true_mg)
        zone_buckets[zone]['pred_mgdl'].append(pred_mg)

    for zone in zone_buckets:
        for key in ['true_mmol', 'pred_mmol', 'true_mgdl', 'pred_mgdl']:
            zone_buckets[zone][key] = np.asarray(zone_buckets[zone][key], dtype=np.float64)

        count = len(zone_buckets[zone]['true_mmol'])
        zone_buckets[zone]['count'] = count
        zone_buckets[zone]['percent'] = (count / n * 100.0) if n > 0 else 0.0

    return zone_buckets


def print_metrics_summary(metrics, title="Evaluation Metrics"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    print(f"\n📊 基础指标:")
    print(f"  MAE:  {metrics['mae']:.4f} mmol/L")
    print(f"  RMSE: {metrics['rmse']:.4f} mmol/L")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  相关系数: {metrics['pearson_corr']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n📈 误差统计:")
    print(f"  平均误差: {metrics['mean_error']:.4f} ± {metrics['std_error']:.4f}")
    print(f"  中位数误差: {metrics['median_error']:.4f}")
    print(f"  最大/最小误差: {metrics['max_error']:.4f} / {metrics['min_error']:.4f}")
    
    print(f"\n📉 误差分位数:")
    print(f"  25%: {metrics['q25_error']:.4f}")
    print(f"  50%: {metrics['q50_error']:.4f}")
    print(f"  75%: {metrics['q75_error']:.4f}")
    print(f"  95%: {metrics['q95_error']:.4f}")
    
    print(f"\n🎯 Clarke Error Grid:")
    print(f"  Zone A (临床准确):     {metrics['clarke_zone_a']:.2f}%")
    print(f"  Zone B (临床可接受):   {metrics['clarke_zone_b']:.2f}%")
    print(f"  Zone C (可能过度治疗): {metrics['clarke_zone_c']:.2f}%")
    print(f"  Zone D (可能遗漏):     {metrics['clarke_zone_d']:.2f}%")
    print(f"  Zone E (危险误判):     {metrics['clarke_zone_e']:.2f}%")
    
    print(f"\n样本数: {metrics['n_samples']}")
    print("="*70 + "\n")

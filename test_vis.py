import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 区域划分代码 (保留原始逻辑)
# ==========================================
def clarke_error_zone_detailed(act, pred):
    # Zone A
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1

# ==========================================
# 2. 数据处理与实时计算函数
# ==========================================
def calculate_metrics(act_mmol, pred_mmol):
    act_mg = act_mmol * 18.0
    pred_mg = pred_mmol * 18.0
    
    zones = [clarke_error_zone_detailed(a, p) for a, p in zip(act_mg, pred_mg)]
    
    zone_A_count = zones.count(0)
    zone_B_count = zones.count(1) + zones.count(2)
    
    ab_percent = (zone_A_count + zone_B_count) / len(zones) * 100
    mard = np.mean(np.abs(pred_mmol - act_mmol) / act_mmol) * 100
    
    return ab_percent, mard

# ==========================================
# 3. 准备模拟数据
# ==========================================
np.random.seed(42)
n_points = 600

ref_glucose = np.concatenate([
    np.random.normal(loc=8.5, scale=2.5, size=400),
    np.random.uniform(3, 19, size=200)
])
ref_glucose = np.clip(ref_glucose, 3.5, 19)

pred_left = ref_glucose + np.random.normal(0, 0.18 * ref_glucose, size=n_points)
outlier_idx = np.random.choice(n_points, size=20, replace=False)
pred_left[outlier_idx] = pred_left[outlier_idx] * np.random.uniform(0.5, 2.0, size=20)

pred_right = ref_glucose + np.random.normal(0, 0.05 * ref_glucose, size=n_points)

ab_left, mard_left = calculate_metrics(ref_glucose, pred_left)
ab_right, mard_right = calculate_metrics(ref_glucose, pred_right)

# ==========================================
# 4. 图形渲染
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

def plot_clarke_error_grid(ax, title):
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Reference Glucose Value (mmol/L)', fontsize=14)
    ax.set_ylabel('Predicted Glucose Value (mmol/L)', fontsize=14)
    
    ax.tick_params(axis='both', direction='in', labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    c = 1 / 18.0 
    
    # 对角线
    ax.plot([2, 20], [2, 20], 'k--', linewidth=1, alpha=0.6)
    
    # 边界线
    ax.plot([2, 58.33*c], [70*c, 70*c], 'k-', lw=1)
    ax.plot([58.33*c, 400*c], [70*c, 400*c*1.2], 'k-', lw=1)
    ax.plot([70*c, 70*c], [2, 56*c], 'k-', lw=1)
    ax.plot([70*c, 400*c], [56*c, 400*c*0.8], 'k-', lw=1)
    ax.plot([70*c, 70*c], [70*c, 400*c], 'k-', lw=1)
    ax.plot([180*c, 400*c], [70*c, 70*c], 'k-', lw=1)
    ax.plot([2, 70*c], [180*c, 180*c], 'k-', lw=1)
    ax.plot([180*c, 180*c], [2, 70*c], 'k-', lw=1)
    ax.plot([70*c, 400*c], [180*c, 400*c + 110*c], 'k-', lw=1)
    ax.plot([130*c, 180*c], [0, 70*c], 'k-', lw=1)
    ax.plot([240*c, 240*c], [70*c, 180*c], 'k-', lw=1)
    ax.plot([240*c, 400*c], [180*c, 180*c], 'k-', lw=1)

    ax.set_xlim(2, 20)
    ax.set_ylim(2, 20)
    ax.set_xticks([2, 5, 10, 15, 20])
    ax.set_yticks([2, 4, 8, 12, 16, 20])

    fs = 14; fcolor = 'black'; falpha = 0.9; fw = 'bold'
    
    # A
    ax.text(14, 14, 'A', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    ax.text(3.4, 3.4, 'A', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    
    # B
    ax.text(8.5, 11.5, 'B', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    ax.text(16, 11.5, 'B', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    
    # C
    ax.text(8.5, 17, 'C', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    ax.text(9.4, 2.6, 'C', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    
    # D
    ax.text(3.2, 7.5, 'D', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    ax.text(17, 7.5, 'D', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    
    # E
    ax.text(3.2, 16, 'E', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center')
    # 修改处：将右下角的 x 坐标从 17 移到了 13，完美避开统计文本框
    ax.text(13, 3.2, 'E', fontsize=fs, color=fcolor, alpha=falpha, fontweight=fw, ha='center', va='center') 

plot_clarke_error_grid(ax1, 'Single RF Modal (R)')
plot_clarke_error_grid(ax2, 'Invention Multimodal Fusion Method')

scatter_kwargs = dict(facecolors='none', edgecolors='black', s=20, alpha=0.8, linewidths=0.8)
ax1.scatter(ref_glucose, pred_left, **scatter_kwargs)
ax2.scatter(ref_glucose, pred_right, **scatter_kwargs)

text_box_style = dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='none', alpha=0.95)

text_left = f"{ab_left:.1f}% in A+B Zones\n{mard_left:.1f}% MARD"
ax1.text(19.5, 2.5, text_left, fontsize=13, ha='right', va='bottom', bbox=text_box_style)

text_right = f"{ab_right:.1f}% in A+B Zones\n{mard_right:.1f}% MARD"
ax2.text(19.5, 2.5, text_right, fontsize=13, ha='right', va='bottom', bbox=text_box_style)

# ==========================================
# 5. 保存与展示
# ==========================================
plt.tight_layout()

output_filename = 'clarke_error_grid_adjusted.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"图片已成功保存至当前目录：{os.path.abspath(output_filename)}")
plt.show()
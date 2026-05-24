"""Create bar-chart summaries for recent glucose training results."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LATEST_DIR = ROOT / "results" / "01211555_Tao_and_4_more"
OUTPUT_DIR = ROOT / "visualization" / "training_result_bars"

CHINESE_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
]


def configure_chinese_font():
    """Configure Matplotlib to render Chinese labels on Linux servers."""
    for font_path in CHINESE_FONT_PATHS:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False


def chinese_font(size=None, weight="normal"):
    """Return a FontProperties object backed by an installed CJK font file."""
    for font_path in CHINESE_FONT_PATHS:
        if font_path.exists():
            return fm.FontProperties(fname=str(font_path), size=size, weight=weight)
    return fm.FontProperties(size=size, weight=weight)


LATEST_METRICS = {
    "MAE": 1.1487,
    "RMSE": 1.5530,
    "R2": 0.2288,
    "Correlation": 0.5138,
    "MAPE": 15.85,
    "Zone A": 68.66,
    "Zone B": 31.34,
    "Zone C": 0.0,
    "Zone D": 0.0,
    "Zone E": 0.0,
    "Samples": 34202,
}

RECENT_RUNS = [
    {
        "name": "用户1",
        "mae": 0.6679,
        "rmse": 0.9460,
        "mape": 9.53,
        "r2": 0.7139,
        "zone_a": 88.75,
        "samples": 34202,
    },
    {
        "name": "用户2",
        "mae": 0.5695,
        "rmse": 0.8407,
        "mape": 7.94,
        "r2": 0.7740,
        "zone_a": 91.83,
        "samples": 34202,
    },
    {
        "name": "用户3",
        "mae": 0.3381,
        "rmse": 0.5207,
        "mape": 5.74,
        "r2": 0.8640,
        "zone_a": 95.26,
        "samples": 34202,
    },
    {
        "name": "用户4",
        "mae": 0.4881,
        "rmse": 0.7207,
        "mape": 6.84,
        "r2": 0.8140,
        "zone_a": 93.83,
        "samples": 34202,
    },
    {
        "name": "用户5",
        "mae": 0.2289,
        "rmse": 0.3349,
        "mape": 4.38,
        "r2": 0.9069,
        "zone_a": 98.71,
        "samples": 18754,
    },
]

TRAINING_TIME = {
    "Data": 215.67,
    "Training": 46977.86,
    "Validation": 4390.56,
    "Testing": 189.96,
    "Other/plots": 18667.81,
}


def _read_latest_points():
    csv_path = LATEST_DIR / "test_clarke_zone_points_1.csv"
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual = float(row["actual_mmol_per_l"])
            pred = float(row["predicted_mmol_per_l"])
            rows.append((row["zone"], actual, pred, abs(pred - actual), pred - actual))
    return rows


def _annotate_bars(ax, values, fmt="{:.2f}", y_pad=0.02):
    ymax = max(values) if len(values) else 0
    offset = (ymax or 1) * y_pad
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_latest_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Latest Tao Test Result Summary", fontsize=15, fontweight="bold")

    metrics = ["MAE", "RMSE"]
    values = [LATEST_METRICS[m] for m in metrics]
    axes[0].bar(metrics, values, color=["#4C78A8", "#F58518"])
    axes[0].set_ylabel("mmol/L")
    axes[0].set_title("Error Magnitude")
    axes[0].set_ylim(0, max(values) * 1.25)
    _annotate_bars(axes[0], values)

    metrics = ["R2", "Correlation"]
    values = [LATEST_METRICS[m] for m in metrics]
    axes[1].bar(metrics, values, color=["#54A24B", "#72B7B2"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Fit and Correlation")
    _annotate_bars(axes[1], values)

    values = [LATEST_METRICS["MAPE"]]
    axes[2].bar(["MAPE"], values, color="#E45756")
    axes[2].set_ylabel("%")
    axes[2].set_title("Relative Error")
    axes[2].set_ylim(0, values[0] * 1.25)
    _annotate_bars(axes[2], values, fmt="{:.2f}%")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_latest_overall_metrics.png", dpi=220)
    plt.close(fig)


def plot_clarke_zones():
    zones = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]
    values = [LATEST_METRICS[z] for z in zones]
    colors = ["#2CA02C", "#86BC86", "#F2CF5B", "#F28E2B", "#D62728"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(zones, values, color=colors)
    ax.set_ylabel("Percent of samples (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Latest Tao Clarke Error Grid Distribution")
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, values, fmt="{:.2f}%")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_latest_clarke_zones.png", dpi=220)
    plt.close(fig)


def plot_error_bands(rows):
    abs_errors = np.array([r[3] for r in rows])
    bands = [
        ("<=0.5", abs_errors <= 0.5),
        ("0.5-1.0", (abs_errors > 0.5) & (abs_errors <= 1.0)),
        ("1.0-2.0", (abs_errors > 1.0) & (abs_errors <= 2.0)),
        ("2.0-3.0", (abs_errors > 2.0) & (abs_errors <= 3.0)),
        (">3.0", abs_errors > 3.0),
    ]
    labels = [b[0] for b in bands]
    counts = np.array([int(mask.sum()) for _, mask in bands])
    percents = counts / len(abs_errors) * 100

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(labels, percents, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"])
    ax.set_ylabel("Percent of samples (%)")
    ax.set_xlabel("Absolute error band (mmol/L)")
    ax.set_title("Latest Tao Absolute Error Bands")
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, percents, fmt="{:.1f}%")
    for i, count in enumerate(counts):
        ax.text(i, 1.0, f"n={count}", ha="center", va="bottom", fontsize=8, color="#333333")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_latest_error_bands.png", dpi=220)
    plt.close(fig)


def plot_glucose_bin_performance(rows):
    actual = np.array([r[1] for r in rows])
    abs_errors = np.array([r[3] for r in rows])
    bins = [
        ("<5.6", actual < 5.6),
        ("5.6-7.8", (actual >= 5.6) & (actual < 7.8)),
        ("7.8-10.0", (actual >= 7.8) & (actual < 10.0)),
        (">=10.0", actual >= 10.0),
    ]
    labels = [b[0] for b in bins]
    mae = np.array([abs_errors[mask].mean() if mask.any() else 0 for _, mask in bins])
    counts = np.array([int(mask.sum()) for _, mask in bins])

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax2 = ax1.twinx()
    x = np.arange(len(labels))

    ax1.bar(x - 0.18, mae, width=0.36, color="#4C78A8", label="MAE")
    ax2.bar(x + 0.18, counts, width=0.36, color="#B279A2", alpha=0.75, label="Samples")
    ax1.set_ylabel("MAE (mmol/L)")
    ax2.set_ylabel("Sample count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("实际血糖范围 (mmol/L)")
    ax1.set_title("多模态最优训练结果——按血糖范围误差表现")
    ax1.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(mae):
        ax1.text(idx - 0.18, value + max(mae) * 0.03, f"{value:.2f}", ha="center", fontsize=9)
    for idx, value in enumerate(counts):
        ax2.text(idx + 0.18, value + max(counts) * 0.03, f"{value}", ha="center", fontsize=8)
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.12, 0.5),
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_latest_glucose_range_performance.png", dpi=220)
    plt.close(fig)


def plot_recent_run_comparison():
    labels = [r["name"] for r in RECENT_RUNS]
    x = np.arange(len(labels))
    main_title_font = chinese_font(size=22, weight="bold")
    axis_title_font = chinese_font(size=21, weight="bold")
    tick_font = chinese_font(size=17)
    value_font = chinese_font(size=16, weight="bold")
    caption_font = chinese_font(size=18, weight="bold")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.suptitle(
        "多模态多用户训练结果对比",
        fontproperties=main_title_font,
    )

    panels = [
        ("MAE (mmol/L)", "mae", "#5BB7DC", False, "(a) MAE"),
        ("RMSE (mmol/L)", "rmse", "#F5A623", False, "(b) RMSE"),
        ("MAPE (%)", "mape", "#E86E6E", False, "(c) MAPE"),
        ("Clarke Zone A (%)", "zone_a", "#65B96E", True, "(d) Clarke Zone A"),
    ]
    for ax, (title, key, color, pct, caption) in zip(axes.ravel(), panels):
        values = [r[key] for r in RECENT_RUNS]
        ax.bar(
            x,
            values,
            width=0.42,
            color=color,
            edgecolor="black",
            linewidth=2.2,
            hatch="//",
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("训练结果类型", fontproperties=axis_title_font)
        ax.set_ylabel(title, fontproperties=axis_title_font)
        ax.set_title(
            caption,
            y=-0.36,
            fontproperties=caption_font,
        )
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            width=2.2,
            length=7,
            labelsize=17,
        )
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontproperties(tick_font)
        ax.grid(axis="y", linestyle="--", linewidth=1.8, alpha=0.32, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color("black")
        if pct:
            ax.set_ylim(0, 105)
            value_labels = [f"{value:.2f}%" for value in values]
        else:
            ax.set_ylim(0, max(values) * 1.25)
            value_labels = [f"{value:.2f}" for value in values]
        y_offset = ax.get_ylim()[1] * 0.025
        for xpos, value, label in zip(x, values, value_labels):
            ax.text(
                xpos,
                value + y_offset,
                label,
                ha="center",
                va="bottom",
                fontproperties=value_font,
            )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.13, wspace=0.34, hspace=0.62)
    fig.savefig(OUTPUT_DIR / "05_recent_run_comparison.png", dpi=220)
    plt.close(fig)


def plot_training_time_breakdown():
    labels = list(TRAINING_TIME.keys())
    minutes = np.array(list(TRAINING_TIME.values())) / 60.0
    colors = ["#72B7B2", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, minutes, color=colors)
    ax.set_ylabel("Minutes")
    ax.set_title("Latest Tao Runtime Breakdown")
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, minutes, fmt="{:.1f}")
    ax.text(
        0.02,
        0.95,
        "Best test epoch during training: 4\nBest MAE during training: 1.0524 mmol/L\nFinal test MAE: 1.1487 mmol/L",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB"),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_latest_training_time_breakdown.png", dpi=220)
    plt.close(fig)


def main():
    configure_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_latest_points()
    plot_latest_metrics()
    plot_clarke_zones()
    plot_error_bands(rows)
    plot_glucose_bin_performance(rows)
    plot_recent_run_comparison()
    plot_training_time_breakdown()
    print(f"Saved charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

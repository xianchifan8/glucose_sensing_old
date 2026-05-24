"""
可视化模块
用于训练后的模型评估和结果分析
所有图表保存到文件，不在服务器上显示
"""

from .plots import (
    plot_prediction_vs_actual,
    plot_error_distribution,
    plot_scatter_with_metrics,
    plot_residual_analysis,
    plot_error_over_time,
    create_evaluation_report
)

from .metrics import (
    calculate_all_metrics,
    print_metrics_summary
)

__all__ = [
    'plot_prediction_vs_actual',
    'plot_error_distribution',
    'plot_scatter_with_metrics',
    'plot_residual_analysis',
    'plot_error_over_time',
    'create_evaluation_report',
    'calculate_all_metrics',
    'print_metrics_summary',
]

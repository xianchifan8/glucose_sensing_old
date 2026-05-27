"""
Glucose Sensing Training - 主训练入口
简洁、模块化的训练框架，集成 W&B 实验管理
"""

import argparse
import wandb
import time
import numpy as np
import os
import copy
import csv
from datetime import datetime

from config import Config, get_config_from_args
from utils import setup_device, set_seed, print_model_info
from Model import create_model
from data import load_and_preprocess_data
from trainer import GlucoseTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Glucose Sensing Training')
    
    # 训练参数
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--weight_decay', type=float, help='权重衰减（L2正则化）')
    parser.add_argument('--loss', type=str, choices=['MSE', 'MAE', 'Huber'],
                       help='损失函数: MSE, MAE, Huber')
    parser.add_argument('--huber_delta', type=float, help='Huber Loss的delta参数')
    parser.add_argument('--mae_window_loss_weight', type=float,
                       help='跨多个batch聚合后的MAE辅助损失权重，0表示关闭')
    parser.add_argument('--mae_window_batches', type=int,
                       help='累积多少个训练batch后，计算一次聚合MAE并执行更新')
    parser.add_argument('--fusion_diagnostics', action='store_true',
                       help='记录gated residual的gate/correction诊断统计，不改变训练目标')
    parser.add_argument('--fusion_correction_loss_weight', type=float,
                       help='gated residual修正量正则权重: loss += weight * mean(correction^2)，0表示关闭')
    parser.add_argument('--gated_residual_scale_mode', type=str,
                       choices=['sigmoid', 'clamp'],
                       help='gated residual的residual_scale约束方式: sigmoid为默认修复版，clamp为旧实现')
    parser.add_argument('--gated_residual_min_scale', type=float,
                       help='sigmoid模式下gated residual的residual_scale下限，防止修正分支塌缩为0')
    parser.add_argument('--gated_residual_correction_scale', type=str,
                       choices=['none', 'spectrum_rms', 'multiplicative'],
                       help='gated residual修正量尺度: none为旧实现, spectrum_rms按spectrum feature RMS放大residual, multiplicative使用(1+scale*gate*residual)*spectrum_feature')
    parser.add_argument('--gated_residual_differential_lr', action='store_true',
                       help='启用base_model与late gated residual/aux/fusion分支的差异学习率')
    parser.add_argument('--gated_residual_fusion_lr', type=float,
                       help='差异学习率开启时，late fusion/aux/gate/residual分支学习率；不设则使用主lr')
    parser.add_argument('--gated_residual_scale_lr', type=float,
                       help='差异学习率开启时，residual_scale单独学习率；不设则使用fusion_lr')
    
    # 模型参数
    parser.add_argument('--model', type=str, choices=['MLP', 'CNN', 'Transformer', 'TCN', 'RF_CNN'],
                       help='模型架构')
    parser.add_argument('--hidden_size', type=int, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, help='模型层数（TCN的TemporalBlock层数）')
    parser.add_argument('--dropout', type=float, help='Dropout 比例')
    parser.add_argument('--use_attention', action='store_true', help='启用注意力机制（频谱注意力+时间注意力）')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false', help='禁用注意力机制')
    parser.set_defaults(use_attention=True)  # 默认启用注意力
    parser.add_argument('--aggregation', type=str, 
                       choices=['attention', 'weighted', 'concat', 'lstm', 'mean', 'last'],
                       default='attention',
                       help='时间步聚合策略: attention(注意力加权), weighted(可学习权重), concat(拼接), lstm(LSTM), mean(平均池化), last(仅最后一步)')
    parser.add_argument('--use_dropblock', action='store_true', help='启用DropBlock正则化（比Dropout更强）')
    parser.add_argument('--no_dropblock', dest='use_dropblock', action='store_false', help='禁用DropBlock')
    parser.set_defaults(use_dropblock=True)  # 默认启用DropBlock
    parser.add_argument('--use_skip', action='store_true', help='启用跳跃连接（频谱特征直连输出层）')
    parser.add_argument('--no_skip', dest='use_skip', action='store_false', help='禁用跳跃连接')
    parser.set_defaults(use_skip=True)  # 默认启用跳跃连接
    
    # 数据参数
    parser.add_argument('--user', type=str, help='用户名（单用户）')
    parser.add_argument('--users', type=str, nargs='+',
                       help='多用户列表（空格分隔），例如: --users Tao Lili Ruichun')
    parser.add_argument('--normalize', action='store_true', help='是否归一化特征')
    
    # 数据源参数
    parser.add_argument('--data_source', type=str, choices=['bin', 'db'],
                       default='bin',
                       help='数据源格式: bin(原始BIN格式,默认) 或 db(SQLite数据库格式)')
    parser.add_argument('--db_user', type=str, default='Tao_db',
                       help='DB格式数据的用户目录名（单用户, 默认: Tao_db）')
    parser.add_argument('--db_users', type=str, nargs='+',
                       help='DB格式多用户目录名（空格分隔），例如: --db_users Tao_db Lili_db')
    
    parser.add_argument('--data_fusion', action='store_true', help='是否使用辅助传感器数据融合（BME680, PPG, T117, ICM）')
    
    parser.add_argument('--fusion_stage', type=str,
                       choices=['early', 'late'],
                       default='early',
                       help='数据融合阶段: early(数据层/频谱层融合,1001+14→SpectralCNN), late(特征层融合,频谱→128维+aux→32维→拼接)')
    
    # Early fusion方法 (数据层融合)
    parser.add_argument('--fusion_method', type=str, 
                       choices=['concat', 'film', 'film_sensor_aware', 'attention_pool', 'residual_film', 'conditioned_rf', 'attention_fusion'],
                       default='concat',
                       help='Early fusion方法(数据层): '
                            'concat(拼接,基线), '
                            'film(FiLM全局调制), '
                            'film_sensor_aware(传感器感知FiLM,针对不同生理信号), '
                            'attention_pool(注意力池化), '
                            'residual_film(残差FiLM), '
                            'conditioned_rf(条件化RF建模,aux仅作调制), '
                            'attention_fusion(交叉注意力融合,动态学习信任度)')
    
    # Late fusion方法 (特征层融合)
    parser.add_argument('--fusion_method_late', type=str,
                       choices=['concat', 'cross_attention', 'film', 'cross_attention_film', 'aux_cross_only', 'film_aux_only', 'gated_residual', 'gated_residual_film', 'ppg_icm_gated_residual_film', 'ppg_icm_gated_residual_only', 'ppg_icm_engineered_only', 'ppg_icm_segment_raw_only', 'spectrum_only_late_head'],
                       default='concat',
                       help='Late fusion方法(特征层融合): '
                            'concat(基线-简单拼接), '
                            'cross_attention(推荐-双向注意力学习模态关联), '
                           'film(条件调制-aux调制spectrum), '
                           'cross_attention_film(先交叉注意力融合非温湿度aux，再用温湿度aux进行FiLM调制), '
                           'aux_cross_only(仅使用去温湿度后的aux分支预测，主模态不参与输出), '
                           'film_aux_only(仅使用温湿度等aux进行FiLM调制，不使用其他aux传感器), '
                           'gated_residual(aux生成门控残差修正spectrum，适合质量/运动状态辅助), '
                           'gated_residual_film(先用PPG/ICM等动态aux门控残差修正spectrum，再用温湿度aux进行FiLM调制), '
                           'ppg_icm_gated_residual_film(明确PPG/ICM做门控残差，BME680/T117做FiLM调制), '
                           'ppg_icm_gated_residual_only(仅PPG/ICM engineered做门控残差修正spectrum，不做环境FiLM), '
                           'ppg_icm_engineered_only(仅使用engineered后的PPG/ICM特征预测), '
                           'ppg_icm_segment_raw_only(仅使用PPG/ICM同一秒内原始序列槽位预测), '
                           'spectrum_only_late_head(仅spectrum特征接late MLP预测头，用于干净验证late head收益)')
    parser.add_argument('--late_film_aux_indices', type=int, nargs='+', default=None,
                       help='在cross_attention_film/aux_cross_only/film_aux_only中生效：指定用于FiLM分支的aux索引。aux顺序为[BME,PPG,T117,ICM]，例如 --late_film_aux_indices 0 1 2 4')
    parser.add_argument('--aux_feature_mode', type=str,
                       choices=['raw', 'engineered', 'both', 'segment_raw'],
                       default='raw',
                       help='辅助传感器特征模式: raw(保持旧逻辑), engineered(PPG/ICM按秒聚合统计与质量特征), both(拼接raw与engineered), segment_raw(同一秒内PPG/ICM原始序列槽位)')
    
    parser.add_argument('--icm_mode', type=str,
                       choices=['raw', 'processed', 'both'],
                       default='raw',
                       help='ICM数据处理模式: '
                            'raw(原始加速度+陀螺仪,默认), '
                            'processed(积分处理后的位移+角度,更平滑), '
                            'both(两者都使用)')
    parser.add_argument('--aux_override', action='store_true',
                       help='覆盖aux sensor数值为固定值（用于消融实验，测试频谱单独效果）')
    parser.add_argument('--aux_override_value', type=float, default=0.0,
                       help='覆盖aux sensor时使用的数值（默认0.0，可设为其他固定值如1.0）')
    parser.add_argument('--interpolation', type=str, 
                       choices=['linear', 'cubic', 'pchip', 'akima', 'makima', 'nearest'],
                       help='血糖插值方法（推荐pchip或akima用于CGM低采样率数据）')
    parser.add_argument('--split_strategy', type=str, 
                       choices=['random', 'temporal', 'stratified', 'experiment', 'date', 'date_random', 'max_day_train_min_day_test', 'alternating', 'hybrid_alternating', 'cross_user', 'multi_user_independent', 'db_file_temporal_80_20', 'tao_db_label_shuffle_debug'],
                       help='数据集划分策略: random(随机), temporal(时序), stratified(分层), experiment(按实验), date(按日期时序), date_random(按日期随机抽取训练日, 支持指定单用户如--db_user Tao_db), max_day_train_min_day_test(最多日期训练最少日期测试), alternating(交叉时序), hybrid_alternating(混合交叉), cross_user(跨用户训练测试), multi_user_independent(多用户独立划分并分别测试), tao_db_label_shuffle_debug(Tao_db专用debug: 打乱频谱与血糖标签对应关系后随机划分，并强制仅用spectrum单模态)')
    parser.add_argument('--date_user', type=str, default=None,
                       help='date/date_random策略下指定单用户别名: DB模式如 Tao_db, BIN模式如 Tao')
    parser.add_argument('--train_user_name', type=str,
                       help='cross_user模式: 训练用户名称（该用户全部数据用于训练）')
    parser.add_argument('--predict_user_name', type=str,
                       help='cross_user模式: 预测用户名称（该用户数据用于测试）')
    parser.add_argument('--predict_user_ratio', type=float, default=1.0,
                       help='cross_user模式: 预测用户测试集使用比例(0,1]，默认1.0表示全部')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='交叉时序划分(alternating/hybrid_alternating)的分段数，默认10段')
    parser.add_argument('--date_random_repeats', type=int, default=1,
                       help='date_random策略重复实验次数，建议5-10')
    parser.add_argument('--date_train_days', type=int, default=None,
                       help='date/date_random策略下训练集使用的日期数（不指定则按train_split推断）')
    parser.add_argument('--mode', type=str, choices=['instant', 'window', 'rf_image'],
                       default='instant', help='预测模式: instant(当前spectrum预测当前血糖), window(历史窗口预测当前血糖), rf_image(射频时间-频谱图预测窗口平均血糖)')
    parser.add_argument('--window_size', type=int, default=10,
                       help='窗口模式下使用的历史时间步数（样本数）')
    parser.add_argument('--window_duration', type=float, default=None,
                       help='基于时间的窗口模式：窗口时长（秒），例如30表示使用过去30秒的数据。设置此参数将启用时间窗口模式，忽略window_size')
    parser.add_argument('--window_padding', type=str, 
                       choices=['drop', 'zero', 'repeat', 'edge'],
                       default='drop',
                       help='窗口模式下前期样本处理: drop(丢弃,默认), zero(零填充), repeat(重复第一个样本), edge(边缘填充)')
    parser.add_argument('--rf_image_minutes', type=float, default=5.0,
                       help='rf_image模式下每张时间-频谱图覆盖的历史时长（分钟，默认5分钟）')
    parser.add_argument('--rf_image_time_bins', type=int, default=300,
                       help='rf_image模式下二维图的时间采样点数（默认300）')
    
    # 数据下采样参数
    parser.add_argument('--downsample', action='store_true', 
                       help='是否对频谱数据进行下采样')
    parser.add_argument('--downsample_interval', type=float, default=1.0,
                       help='下采样时间间隔（秒），例如 1.0 表示每秒采样一次，2.0 表示每2秒采样一次')
    
    # 数据平滑参数
    parser.add_argument('--smooth', action='store_true',
                       help='启用数据平滑处理（减少时间轴抖动和频谱噪声）')
    parser.add_argument('--time_smooth', type=str, 
                       choices=['none', 'moving_average', 'ema', 'gaussian', 'savgol', 'median', 'wavelet'],
                       default='none',
                       help='时间轴平滑方法（因果滤波，只使用过去数据）：moving_average(移动平均), ema(指数移动平均,推荐), gaussian(高斯), savgol(SG滤波), median(中值,推荐), wavelet(小波)')
    parser.add_argument('--time_smooth_window', type=int, default=5,
                       help='时间轴平滑窗口大小（用于moving_average, median, savgol）')
    parser.add_argument('--time_smooth_sigma', type=float, default=2.0,
                       help='时间轴高斯平滑标准差（用于gaussian）')
    parser.add_argument('--time_smooth_alpha', type=float, default=0.3,
                       help='时间轴EMA平滑系数（用于ema），0-1之间，越小越平滑')
    parser.add_argument('--spectrum_smooth', type=str,
                       choices=['none', 'moving_average', 'gaussian', 'savgol', 'median'],
                       default='none',
                       help='频谱轴平滑方法')
    parser.add_argument('--spectrum_smooth_window', type=int, default=5,
                       help='频谱轴平滑窗口大小')
    parser.add_argument('--spectrum_smooth_sigma', type=float, default=2.0,
                       help='频谱轴高斯平滑标准差')
    
    # 异常值处理参数
    parser.add_argument('--detect_outliers', action='store_true',
                       help='启用异常值检测')
    parser.add_argument('--outlier_method', type=str, choices=['zscore', 'iqr'],
                       default='zscore',
                       help='异常值检测方法：zscore(Z-score), iqr(四分位数)')
    parser.add_argument('--outlier_threshold', type=float, default=3.0,
                       help='异常值检测阈值（Z-score方法）')
    parser.add_argument('--handle_outliers', action='store_true',
                       help='处理检测到的异常值')
    parser.add_argument('--outlier_handle_method', type=str,
                       choices=['interpolate', 'median', 'mean', 'clip'],
                       default='interpolate',
                       help='异常值处理方法：interpolate(插值), median(中位数), mean(均值), clip(截断)')
    
    # Auto-Regressive (自回归) 参数
    parser.add_argument('--autoregressive', action='store_true',
                       help='启用自回归模式：使用历史血糖值辅助预测')
    parser.add_argument('--ar_glucose_history', type=int, default=5,
                       help='自回归模式下使用多少个历史血糖值（默认5个）')
    parser.add_argument('--ar_aligned', action='store_true', dest='ar_glucose_aligned',
                       help='历史血糖与当前spectrum对齐（当前血糖未知，默认）')
    parser.add_argument('--ar_no_aligned', action='store_false', dest='ar_glucose_aligned',
                       help='历史血糖包含当前时刻（用于增强趋势学习）')
    parser.set_defaults(ar_glucose_aligned=True)
    parser.add_argument('--ar_smoothness_weight', type=float, default=0.1,
                       help='自回归平滑性正则化权重：限制预测跳变（默认0.1）')
    parser.add_argument('--ar_history_weight', type=float, default=0.05,
                       help='自回归历史一致性权重：限制偏离历史趋势（默认0.05）')
    parser.add_argument('--ar_test_interval', type=int, default=5,
                       help='AR模式下真实推理测试间隔：每N个epoch进行一次真实AR推理测试（默认5，0表示每个epoch都测试）')
    parser.add_argument('--ar_fusion_strategy', type=str, 
                       choices=['concat', 'film', 'gating'],
                       default='gating',
                       help='AR模式glucose_history融合策略：concat(拼接), film(调制), gating(门控，推荐)')
    parser.add_argument('--ar_glucose_encoder_dim', type=int, default=128,
                       help='AR glucose encoder输出维度（默认128，建议与spectrum特征维度接近）')
    
    # 数据集选择
    parser.add_argument('--experiments', type=str, nargs='+', 
                       help='指定要使用的实验，例如: --experiments 11201828_Tao 11211544_Tao，或 --experiments all 使用所有实验')
    
    # 多用户实验过滤 ⭐ NEW
    parser.add_argument('--user_experiments', type=str, nargs='+', action='append',
                       help='为每个用户指定实验列表，格式: --user_experiments Tao_db exp1 exp2 --user_experiments Weiyi_db exp3 exp4')
    parser.add_argument('--no_val', action='store_true', 
                       help='不使用验证集，只划分训练集和测试集')
    parser.add_argument('--train_split', type=float, help='训练集比例，例如: --train_split 0.8')
    parser.add_argument('--val_split', type=float, help='验证集比例，例如: --val_split 0.1')
    parser.add_argument('--test_split', type=float, help='测试集比例，例如: --test_split 0.1')
    
    # W&B 参数
    parser.add_argument('--no_wandb', action='store_true', help='禁用 W&B')
    parser.add_argument('--wandb_name', type=str, help='W&B run 名称')
    
    parser.add_argument('--aux_shuffle', action='store_true',
                        help='Shuffle auxiliary features while keeping spectrum-label alignment unchanged')
    parser.add_argument('--aux_shuffle_mode', type=str, default='sensor_groups',
                        choices=['rows', 'features', 'sensor_groups'],
                        help='Aux shuffle mode')
    parser.add_argument('--aux_shuffle_seed', type=int, default=42,
                        help='Random seed for auxiliary feature shuffling')

    return parser.parse_args()


def _run_single_experiment(args, config, device, run_idx=1, total_runs=1, results_dir_override=None):
    """执行一次完整训练+测试流程。"""
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 初始化 W&B
    use_wandb = config.training.wandb_enabled and not args.no_wandb
    if use_wandb:
        # 生成数据集标识（与run_batch.py保持一致）
        if config.data.experiment_filter:
            # 使用简短的实验名（去掉_Tao后缀）
            exp_names = [exp.replace('_Tao', '') for exp in config.data.experiment_filter]
            dataset_tag = '_'.join(exp_names) if len(exp_names) <= 3 else f"{len(exp_names)}exps"
        else:
            dataset_tag = "all"
        
        # 开始构建命名部分列表（与run_batch.py的generate_task_name()格式完全一致）
        name_parts = []
        
        # 0. 数据源标识（db 或 bin）
        data_source = getattr(config.data, 'data_source', 'bin')
        name_parts.append(f"[{data_source.upper()}]")
        
        # 1. 模型架构
        name_parts.append(config.model.architecture)
        
        # 2. 实验数据集
        name_parts.append(f"exp{dataset_tag}")
        
        # 3. 模式标识（window或instant）
        if config.data.mode == 'window':
            # 优先使用window_duration，其次使用window_size
            if hasattr(config.data, 'window_duration') and config.data.window_duration is not None:
                name_parts.append(f"win_wd{int(config.data.window_duration)}")
            else:
                name_parts.append(f"win_ws{config.data.window_size}")
        elif config.data.mode == 'rf_image':
            name_parts.append(f"rfimg_{config.data.rf_image_minutes:g}min_t{config.data.rf_image_time_bins}")
        else:
            name_parts.append("inst")
        
        # 4. 数据融合状态和方法（重要！与run_batch.py完全一致）
        if config.data.data_fusion:
            # 根据融合阶段选择对应的fusion_method
            if config.data.fusion_stage == 'early':
                fusion_method = config.data.fusion_method
            else:
                fusion_method = config.data.fusion_method_late
            
            fusion_abbr = {
                'concat': 'con', 
                'film': 'film',
                'film_sensor_aware': 'filmsa',
                'attention_pool': 'ap',
                'residual_film': 'rfilm',
                'conditioned_rf': 'crf',
                'attention_fusion': 'attnfus',
                'cross_attention': 'cattn',  # Late fusion: 交叉注意力
                'cross_attention_film': 'cafilm',  # Late fusion: 交叉注意力 + FiLM
                'aux_cross_only': 'auxonly',  # Late fusion: 仅aux-cross分支预测
                'film_aux_only': 'faux',  # Late fusion: 仅aux-film分支FiLM调制
                'gated_residual': 'gres',  # Late fusion: aux门控残差修正spectrum
                'gated_residual_film': 'gresfilm',  # Late fusion: 门控残差 + 温湿度FiLM
                'ppg_icm_gated_residual_film': 'pigrfilm',  # Late fusion: PPG/ICM门控残差 + 环境FiLM
                'ppg_icm_gated_residual_only': 'pigronly',  # Late fusion: PPG/ICM门控残差，无环境FiLM
                'ppg_icm_engineered_only': 'piengonly',  # Late fusion: 仅engineered PPG/ICM
                'ppg_icm_segment_raw_only': 'pisegraw',  # Late fusion: 仅PPG/ICM原始段
                'spectrum_only_late_head': 'solate',  # Late fusion: 仅spectrum + late MLP head
                'attention': 'attn'
            }.get(fusion_method, fusion_method[:4])
            
            # 添加融合阶段标识
            stage_abbr = 'e' if config.data.fusion_stage == 'early' else 'l'
            name_parts.append(f"fus{stage_abbr}{fusion_abbr}")
            
            # 4.5. ICM模式（当使用数据融合时）
            if hasattr(config.data, 'icm_mode') and config.data.icm_mode != 'raw':
                icm_abbr = {
                    'raw': 'raw',
                    'processed': 'proc',
                    'both': 'both'
                }.get(config.data.icm_mode, config.data.icm_mode)
                name_parts.append(f"icm{icm_abbr}")
            if hasattr(config.data, 'aux_feature_mode') and config.data.aux_feature_mode != 'raw':
                aux_feat_abbr = {
                    'engineered': 'eng',
                    'both': 'both',
                    'segment_raw': 'segraw'
                }.get(config.data.aux_feature_mode, config.data.aux_feature_mode)
                name_parts.append(f"aux{aux_feat_abbr}")
        
        # 4.6. Aux Override（消融实验标识）- 独立判断，与run_batch.py一致
        if hasattr(config.data, 'aux_override') and config.data.aux_override:
            aux_val = config.data.aux_override_value
            print(f"✓ 检测到 aux_override={config.data.aux_override}, value={aux_val}")
            # 格式化值：整数直接显示，小数保留1位
            if aux_val == int(aux_val):
                name_parts.append(f"auxov{int(aux_val)}")
            else:
                name_parts.append(f"auxov{aux_val:.1f}")
            print(f"✓ 已添加命名标识: auxov{int(aux_val) if aux_val == int(aux_val) else aux_val:.1f}")

        if hasattr(config.data, 'aux_shuffle') and config.data.aux_shuffle:
            aux_shuffle_mode = getattr(config.data, 'aux_shuffle_mode', 'sensor_groups')
            name_parts.append(f"auxshuf{aux_shuffle_mode}")
            print(f"✓ 检测到 aux_shuffle=True, mode={aux_shuffle_mode}")
        
        # 5. Attention状态
        if hasattr(config.model, 'use_attention'):
            if config.model.use_attention:
                name_parts.append("attn")
            else:
                name_parts.append("noattn")
        
        # 6. Normalize状态
        if config.data.normalize:
            name_parts.append("norm")
        
        # 7. 学习率
        name_parts.append(f"lr{config.training.learning_rate}")
        
        # 8. Epochs
        name_parts.append(f"ep{config.training.epochs}")
        
        # 9. Dropout（仅当dropout > 0时添加）
        if hasattr(config.model, 'dropout') and config.model.dropout > 0:
            name_parts.append(f"dr{config.model.dropout}")
        
        # 10. Auto-Regressive模式
        if config.data.autoregressive:
            # AR历史长度
            name_parts.append(f"ar{config.data.ar_glucose_history}")
            # AR融合策略
            fusion_abbr = {
                'concat': 'cat',
                'film': 'film',
                'gating': 'gate'
            }.get(config.data.ar_fusion_strategy, config.data.ar_fusion_strategy[:4])
            name_parts.append(f"fus{fusion_abbr}")
            # AR encoder维度（仅当不是默认值32时添加）
            if config.data.ar_glucose_encoder_dim != 32:
                name_parts.append(f"enc{config.data.ar_glucose_encoder_dim}")
            # AR正则化权重（仅当非零时添加）
            if config.data.ar_smoothness_weight > 0:
                name_parts.append(f"sm{config.data.ar_smoothness_weight}")
            if config.data.ar_history_weight > 0:
                name_parts.append(f"hs{config.data.ar_history_weight}")
        
        # 11. Split策略
        split_abbr = {
            'random': 'rnd', 
            'temporal': 'tmp', 
            'stratified': 'str', 
            'experiment': 'exp', 
            'date': 'date',
            'date_random': 'drnd',
            'max_day_train_min_day_test': 'mxmn',
            'alternating': 'alt',
            'hybrid_alternating': 'halt',
            'cross_user': 'xusr',
            'multi_user_independent': 'mui',
            'db_file_temporal_80_20': 'db8020',
            'tao_db_label_shuffle_debug': 'tdbg'
        }.get(config.data.split_strategy, config.data.split_strategy[:3])
        
        # 如果是alternating或hybrid_alternating策略，添加n_splits
        if config.data.split_strategy in ['alternating', 'hybrid_alternating'] and hasattr(config.data, 'n_splits'):
            name_parts.append(f"{split_abbr}{config.data.n_splits}")
        else:
            name_parts.append(split_abbr)
        
        # 自动生成描述性的 run 名称（与run_batch.py格式完全一致）
        if args.wandb_name:
            run_name = args.wandb_name if total_runs == 1 else f"{args.wandb_name}_r{run_idx:02d}"
        else:
            base_name = "_".join(name_parts)
            # 如果在batch模式下，添加实验编号前缀
            batch_exp_num = os.environ.get('BATCH_EXP_NUMBER')
            if batch_exp_num:
                run_name = f"exp{batch_exp_num}_{base_name}"
            else:
                run_name = base_name
            if total_runs > 1:
                run_name = f"{run_name}_r{run_idx:02d}"
        wandb.init(
            entity=config.training.wandb_entity,
            project=config.training.wandb_project,
            name=run_name,
            config=config.to_dict()
        )
        print(f"✓ W&B 初始化完成: {wandb.run.name}")
    
    # 加载数据
    print(f"\n{'='*70}")
    print("数据加载")
    print(f"{'='*70}")
    
    # 记录数据处理时间
    data_start_time = time.time()
    
    result = load_and_preprocess_data(
        config.data,
        dataset_root=config.data.dataset_root,
        normalize=config.data.normalize,
        seed=config.training.seed,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers
    )
    
    # 根据是否归一化解包返回值
    if config.data.normalize:
        train_loader, val_loader, test_loader, scaler, train_metadata, test_metadata, full_metadata = result
    else:
        train_loader, val_loader, test_loader, train_metadata, test_metadata, full_metadata = result
    
    # 记录数据处理完成时间
    data_processing_time = time.time() - data_start_time
    print(f"\n✓ 数据处理完成，用时: {data_processing_time:.2f}秒 ({data_processing_time/60:.2f}分钟)")
    
    # 从数据推断输入维度
    sample_batch = next(iter(train_loader))
    
    # 检测是否使用Late Fusion模式
    use_late_fusion = config.data.data_fusion and config.data.fusion_stage == 'late'
    
    # 检测是否使用Auto-Regressive模式
    use_autoregressive = config.data.autoregressive
    
    # 根据不同模式解析batch结构
    # AR + Late Fusion: (spectrum, aux, glucose_hist, label)
    # AR only: (spectrum, glucose_hist, label)
    # Late Fusion only: (spectrum, aux, label)
    # Normal: (data, label)
    
    if use_autoregressive and use_late_fusion:
        # AR + Late Fusion: (spectrum, aux, glucose_hist, label)
        spectrum_sample, aux_sample, glucose_hist_sample, _ = sample_batch
        ar_history_len = glucose_hist_sample.shape[1]
        if config.data.mode == 'instant':
            config.model.input_size = spectrum_sample.shape[1]  # spectrum dim
            aux_input_dim = aux_sample.shape[1]  # aux dim
            print(f"✓ AR + Late Fusion模式 - spectrum维度: {config.model.input_size}, aux维度: {aux_input_dim}, 历史长度: {ar_history_len}")
        else:  # window mode
            config.model.input_size = spectrum_sample.shape[2]  # (batch, window_size, spectrum_dim)
            aux_input_dim = aux_sample.shape[2]  # (batch, window_size, aux_dim)
            print(f"✓ AR + Late Fusion模式 - spectrum维度: {config.model.input_size}, aux维度: {aux_input_dim}, 窗口大小: {spectrum_sample.shape[1]}, 历史长度: {ar_history_len}")
    elif use_autoregressive:
        # AR only: (spectrum, glucose_hist, label)
        spectrum_sample, glucose_hist_sample, _ = sample_batch
        ar_history_len = glucose_hist_sample.shape[1]
        if config.data.mode == 'instant':
            config.model.input_size = spectrum_sample.shape[1]  # spectrum dim
            print(f"✓ Auto-Regressive模式 - spectrum维度: {config.model.input_size}, 历史长度: {ar_history_len}")
        else:  # window mode
            config.model.input_size = spectrum_sample.shape[2]  # (batch, window_size, spectrum_dim)
            print(f"✓ Auto-Regressive模式 - spectrum维度: {config.model.input_size}, 窗口大小: {spectrum_sample.shape[1]}, 历史长度: {ar_history_len}")
        aux_input_dim = None  # No aux in AR-only mode
    elif use_late_fusion:
        # Late Fusion: sample_batch = (spectrum, aux, label)
        spectrum_sample, aux_sample, _ = sample_batch
        if config.data.mode == 'instant':
            config.model.input_size = spectrum_sample.shape[1]  # spectrum dim
            aux_input_dim = aux_sample.shape[1]  # aux dim
            print(f"✓ Late Fusion模式 - spectrum维度: {config.model.input_size}, aux维度: {aux_input_dim}")
        else:  # window mode
            config.model.input_size = spectrum_sample.shape[2]  # (batch, window_size, spectrum_dim)
            aux_input_dim = aux_sample.shape[2]  # (batch, window_size, aux_dim)
            print(f"✓ Late Fusion模式 - spectrum维度: {config.model.input_size}, aux维度: {aux_input_dim}, 窗口大小: {spectrum_sample.shape[1]}")
    else:
        # Early Fusion或无融合: sample_batch = (data, label)
        aux_input_dim = None
        if config.data.mode == 'instant':
            config.model.input_size = sample_batch[0].shape[1]
            print(f"✓ 输入特征维度: {config.model.input_size}")
            if config.data.data_fusion:
                print(f"✓ 数据融合阶段: 数据层(Early - 频谱+aux→SpectralCNN)")
        elif config.data.mode == 'rf_image':
            config.model.input_size = sample_batch[0].shape[-1]
            print(
                f"✓ RF时间-频谱图输入: batch形状={tuple(sample_batch[0].shape)}, "
                f"时间bins={sample_batch[0].shape[-2]}, 频谱维度={config.model.input_size}"
            )
        else:  # window mode
            config.model.input_size = sample_batch[0].shape[2]  # (batch, window_size, features)
            print(f"✓ 输入特征维度: {config.model.input_size}, 窗口大小: {sample_batch[0].shape[1]}")
            if config.data.data_fusion:
                print(f"✓ 数据融合阶段: 数据层(Early - 频谱+aux→SpectralCNN)")
    
    # 创建模型
    print(f"\n{'='*70}")
    print("模型创建")
    print(f"{'='*70}")
    
    # 根据模式选择基础模型
    if config.data.mode == 'instant':
        base_model = create_model(config.model)
    elif config.data.mode == 'rf_image':
        base_model = create_model(config.model)
    else:  # window mode
        from Model.window import create_window_model
        base_model = create_window_model(config.model, config.data.window_size)
    
    # 如果使用Late Fusion，包装基础模型
    if use_late_fusion:
        from Model.late_fusion import create_late_fusion_model
        
        # 获取基础模型的特征维度
        if hasattr(base_model, 'get_feature_dim'):
            spectrum_feature_dim = base_model.get_feature_dim()
        else:
            # 兼容模式：使用hidden_size作为特征维度
            spectrum_feature_dim = config.model.hidden_size
            print(f"⚠️  基础模型没有get_feature_dim方法，使用hidden_size={spectrum_feature_dim}作为特征维度")
        
        model = create_late_fusion_model(
            base_model=base_model,
            spectrum_feature_dim=spectrum_feature_dim,
            aux_input_dim=aux_input_dim,
            config=config
        )
        print(f"✓ Late Fusion模型创建完成:")
        print(f"  - 基础模型: {config.model.architecture}")
        print(f"  - spectrum特征维度: {spectrum_feature_dim}")
        print(f"  - aux输入维度: {aux_input_dim}")
        print(f"  - 融合方法: {config.data.fusion_method_late}")
    else:
        model = base_model
    
    # 如果使用Auto-Regressive模式，包装模型
    use_autoregressive = config.data.autoregressive
    if use_autoregressive:
        from Model.window import create_ar_model
        
        # 获取AR历史长度
        ar_history_len = config.data.ar_glucose_history
        
        model = create_ar_model(model, config)
        print(f"✓ Auto-Regressive模型创建完成:")
        print(f"  - 历史血糖数量: {ar_history_len}")
        print(f"  - 融合策略: {config.data.ar_fusion_strategy}")
        print(f"  - Glucose Encoder维度: {config.data.ar_glucose_encoder_dim}")
        print(f"  - 对齐模式: {'aligned (当前血糖未知)' if config.data.ar_glucose_aligned else 'non-aligned (包含当前血糖)'}")
        print(f"  - 平滑性权重: {config.data.ar_smoothness_weight}")
        print(f"  - 历史一致性权重: {config.data.ar_history_weight}")
    
    model = model.to(device)
    print_model_info(model)

    # 释放用于推断输入维度的样例batch，避免它在整个训练周期中一直占用内存。
    del sample_batch
    try:
        del spectrum_sample
    except UnboundLocalError:
        pass
    try:
        del aux_sample
    except UnboundLocalError:
        pass
    try:
        del glucose_hist_sample
    except UnboundLocalError:
        pass
    
    # 创建训练器
    trainer = GlucoseTrainer(model, config, device, use_wandb=use_wandb, 
                             late_fusion=use_late_fusion, autoregressive=use_autoregressive)
    
    # 获取结果保存目录（批量运行时从环境变量读取）
    if results_dir_override is not None:
        results_dir = results_dir_override
    else:
        results_dir = os.environ.get('BATCH_RESULTS_DIR', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练模型（传入test_loader用于每个epoch评估，传入train_metadata用于训练集可视化，传入test_metadata用于AR真实推理）
    # 注意：为了在可视化时显示完整的ground truth，同时传入full_metadata
    trainer.train(train_loader, val_loader, test_loader, save_dir=config.training.checkpoint_dir, 
                  train_metadata=train_metadata, test_metadata=test_metadata, full_metadata=full_metadata)
    
    # 最终测试模型（传入test_metadata用于可视化，使用批量结果目录）
    # 同时传入full_metadata以在测试图中显示完整ground truth
    test_loss, test_mae, test_rmse, test_mape = trainer.test(
        test_loader,
        test_metadata=test_metadata,
        full_metadata=full_metadata,
        viz_dir=results_dir
    )

    # 记录本次划分日期（尤其用于 date/date_random 策略）
    selected_train_dates = train_metadata.get('selected_dates', []) if train_metadata else []
    selected_test_dates = test_metadata.get('selected_dates', []) if test_metadata else []
    split_info_path = os.path.join(results_dir, 'split_info.txt')
    with open(split_info_path, 'w', encoding='utf-8') as f:
        f.write(f"run_index: {run_idx}/{total_runs}\n")
        f.write(f"split_strategy: {config.data.split_strategy}\n")
        f.write(f"seed: {config.training.seed}\n")
        f.write(f"train_dates: {selected_train_dates}\n")
        f.write(f"test_dates: {selected_test_dates}\n")
        f.write(f"test_mae: {test_mae:.6f}\n")
        f.write(f"test_rmse: {test_rmse:.6f}\n")
        f.write(f"test_mape: {test_mape:.6f}\n")
        if getattr(trainer, 'best_test_epoch', 0) > 0:
            f.write(f"best_test_epoch_during_training: {trainer.best_test_epoch}\n")
            f.write(f"best_test_mae_during_training: {trainer.best_test_mae:.6f}\n")
            f.write(f"best_test_rmse_during_training: {trainer.best_test_rmse:.6f}\n")
            f.write(f"best_test_loss_during_training: {trainer.best_test_loss:.6f}\n")
    print(f"✓ 划分信息已保存: {split_info_path}")

    # multi_user_independent策略: 对每个用户测试集单独评估并输出结果
    if config.data.split_strategy == 'multi_user_independent':
        per_user_test_indices = test_metadata.get('per_user_test_indices', {}) if test_metadata else {}
        if per_user_test_indices:
            from torch.utils.data import DataLoader, Subset

            print("\n" + "="*70)
            print("多用户独立测试结果")
            print("="*70)

            def _slice_test_metadata_by_local_indices(metadata, local_indices):
                """按test_loader中的局部索引切分metadata，保持可视化与统计一致。"""
                sliced = {}
                sample_keys = {'timestamps', 'experiment_indices', 'user_indices', 'split_indices'}
                n_test_samples = len(metadata.get('timestamps', []))

                for key, value in metadata.items():
                    if key == 'per_user_test_indices':
                        continue

                    if key in sample_keys and isinstance(value, np.ndarray) and len(value) == n_test_samples:
                        sliced[key] = value[local_indices]
                    else:
                        sliced[key] = value

                return sliced

            per_user_results = []
            for user_name, local_indices in per_user_test_indices.items():
                if len(local_indices) == 0:
                    continue

                user_dataset = Subset(test_loader.dataset, local_indices.tolist())
                user_test_loader = DataLoader(
                    user_dataset,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    num_workers=config.data.num_workers,
                    pin_memory=config.data.pin_memory
                )

                user_test_metadata = _slice_test_metadata_by_local_indices(test_metadata, local_indices)
                safe_user_name = str(user_name).replace(' ', '_')
                user_viz_dir = os.path.join(results_dir, f"user_{safe_user_name}")

                user_test_loss, user_test_mae, user_test_rmse, user_test_mape = trainer.test(
                    user_test_loader,
                    test_metadata=user_test_metadata,
                    full_metadata=full_metadata,
                    viz_dir=user_viz_dir,
                    metric_prefix=f"final_test_{safe_user_name}",
                    log_to_summary=False
                )

                per_user_results.append((user_name, len(local_indices), user_test_loss, user_test_mae, user_test_rmse, user_test_mape))

            print("\n按用户测试指标汇总:")
            for user_name, n_samples, loss, mae, rmse, mape in per_user_results:
                print(f"  - {user_name}: samples={n_samples}, loss={loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

            if use_wandb and per_user_results:
                per_user_log = {}
                for user_name, n_samples, _, mae, rmse, mape in per_user_results:
                    safe_user_name = str(user_name).replace(' ', '_')
                    per_user_log[f"per_user/{safe_user_name}_mae"] = mae
                    per_user_log[f"per_user/{safe_user_name}_rmse"] = rmse
                    per_user_log[f"per_user/{safe_user_name}_mape"] = mape
                    per_user_log[f"per_user/{safe_user_name}_samples"] = n_samples
                wandb.log(per_user_log)

            print("="*70)

    # db_file_temporal_80_20策略: 对每个.db文件的后20%测试集单独评估
    if config.data.split_strategy == 'db_file_temporal_80_20':
        per_db_file_test_indices = test_metadata.get('per_db_file_test_indices', {}) if test_metadata else {}
        if per_db_file_test_indices:
            from torch.utils.data import DataLoader, Subset

            print("\n" + "="*70)
            print("按DB文件独立测试结果")
            print("="*70)

            def _slice_test_metadata_by_local_indices(metadata, local_indices):
                """按test_loader中的局部索引切分metadata，保持可视化与统计一致。"""
                sliced = {}
                sample_keys = {'timestamps', 'experiment_indices', 'user_indices', 'split_indices', 'db_file_indices'}
                n_test_samples = len(metadata.get('timestamps', []))

                for key, value in metadata.items():
                    if key == 'per_db_file_test_indices':
                        continue

                    if key in sample_keys and isinstance(value, np.ndarray) and len(value) == n_test_samples:
                        sliced[key] = value[local_indices]
                    else:
                        sliced[key] = value

                return sliced

            per_db_results = []
            for db_file_name, local_indices in per_db_file_test_indices.items():
                if len(local_indices) == 0:
                    continue

                db_dataset = Subset(test_loader.dataset, local_indices.tolist())
                db_test_loader = DataLoader(
                    db_dataset,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    num_workers=config.data.num_workers,
                    pin_memory=config.data.pin_memory
                )

                db_test_metadata = _slice_test_metadata_by_local_indices(test_metadata, local_indices)
                safe_db_name = str(db_file_name).replace(os.sep, '_').replace('/', '_').replace(' ', '_')
                db_viz_dir = os.path.join(results_dir, f"db_file_{safe_db_name}")

                db_test_loss, db_test_mae, db_test_rmse, db_test_mape = trainer.test(
                    db_test_loader,
                    test_metadata=db_test_metadata,
                    full_metadata=full_metadata,
                    viz_dir=db_viz_dir,
                    metric_prefix=f"final_test_db_{safe_db_name}",
                    log_to_summary=False
                )

                per_db_results.append((db_file_name, len(local_indices), db_test_loss, db_test_mae, db_test_rmse, db_test_mape))

            print("\n按DB文件测试指标汇总:")
            for db_file_name, n_samples, loss, mae, rmse, mape in per_db_results:
                print(f"  - {db_file_name}: samples={n_samples}, loss={loss:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

            if use_wandb and per_db_results:
                per_db_log = {}
                for db_file_name, n_samples, _, mae, rmse, mape in per_db_results:
                    safe_db_name = str(db_file_name).replace(os.sep, '_').replace('/', '_').replace(' ', '_')
                    per_db_log[f"per_db_file/{safe_db_name}_mae"] = mae
                    per_db_log[f"per_db_file/{safe_db_name}_rmse"] = rmse
                    per_db_log[f"per_db_file/{safe_db_name}_mape"] = mape
                    per_db_log[f"per_db_file/{safe_db_name}_samples"] = n_samples
                wandb.log(per_db_log)

            print("="*70)
    
    # 生成 ground truth 血糖浓度可视化（检查插值质量）
    print("\n" + "="*70)
    print("生成 Ground Truth 血糖浓度可视化")
    print("="*70)
    
    from visualization.plots import generate_ground_truth_visualization
    
    # 创建完整数据集的 loader（训练+测试，用于检查整体插值质量）
    # 使用测试集 loader 和 metadata（因为已经包含完整的时间序列信息）
    # 如果需要查看完整数据，可以合并 train 和 test
    from torch.utils.data import ConcatDataset, DataLoader
    
    full_dataset = ConcatDataset([train_loader.dataset, test_loader.dataset])
    full_loader = DataLoader(
        full_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 合并训练和测试的 metadata
    full_metadata = {
        'timestamps': np.concatenate([train_metadata['timestamps'], test_metadata['timestamps']]),
        'experiment_names': train_metadata.get('experiment_names', []),
        'start_times': train_metadata.get('start_times', [])
    }
    
    gt_filepath = generate_ground_truth_visualization(
        data_loader=full_loader,
        metadata=full_metadata,
        save_dir=results_dir
    )
    
    # 上传到 W&B
    if use_wandb:
        wandb.log({"ground_truth_interpolated": wandb.Image(gt_filepath)})
    
    print("="*70 + "\n")
    
    # 计算总用时
    total_time = time.time() - total_start_time
    
    # 上传时间统计到 W&B
    if use_wandb:
        time_stats = {
            "time/total_time": total_time,
            "time/data_processing_time": data_processing_time,
            "time/training_time": trainer.total_train_time if hasattr(trainer, 'total_train_time') else 0,
            "time/testing_time": trainer.total_test_time if hasattr(trainer, 'total_test_time') else 0,
        }
        
        wandb.log(time_stats)
        print(f"✓ 时间统计已上传到 W&B")
    
    # 完成 W&B
    if use_wandb:
        wandb.finish()
    
    print("\n" + "="*70)
    print("🎉 训练流程完成！")
    print("="*70)
    print(f"\n⏱️  总体用时统计:")
    print(f"  总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"  - 数据处理: {data_processing_time:.2f}秒 ({data_processing_time/total_time*100:.1f}%)")
    if hasattr(trainer, 'total_train_time'):
        print(f"  - 训练: {trainer.total_train_time:.2f}秒 ({trainer.total_train_time/total_time*100:.1f}%)")
        print(f"  - 验证: {trainer.total_val_time:.2f}秒 ({trainer.total_val_time/total_time*100:.1f}%)")
        print(f"  - 测试: {trainer.total_test_time:.2f}秒 ({trainer.total_test_time/total_time*100:.1f}%)")
        other_time = total_time - trainer.total_train_time - trainer.total_val_time - trainer.total_test_time - data_processing_time
        print(f"  - 其他: {other_time:.2f}秒 ({other_time/total_time*100:.1f}%) [模型创建、可视化等]")
    print("="*70 + "\n")

    return {
        'run_index': run_idx,
        'seed': config.training.seed,
        'train_dates': selected_train_dates,
        'test_dates': selected_test_dates,
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'results_dir': results_dir,
    }


def main():
    """主函数"""
    args = parse_args()
    config = get_config_from_args(args)
    device = setup_device()

    split_strategy = config.data.split_strategy
    repeats = int(getattr(config.data, 'date_random_repeats', 1))
    if split_strategy != 'date_random':
        repeats = 1

    if split_strategy == 'date_random' and repeats < 1:
        raise ValueError("date_random_repeats 必须 >= 1")

    if split_strategy == 'date_random' and repeats > 1 and not (5 <= repeats <= 10):
        print(f"⚠️  当前 date_random_repeats={repeats}，通常建议设置为5-10次")

    base_results_dir = os.environ.get('BATCH_RESULTS_DIR', './results')
    base_ckpt_dir = config.training.checkpoint_dir
    os.makedirs(base_results_dir, exist_ok=True)

    all_runs = []
    base_seed = config.training.seed

    for i in range(repeats):
        run_idx = i + 1
        run_config = copy.deepcopy(config)
        run_config.training.seed = base_seed + i

        if repeats > 1:
            run_suffix = f"date_random_run_{run_idx:02d}"
            run_results_dir = os.path.join(base_results_dir, run_suffix)
            run_config.training.checkpoint_dir = os.path.join(base_ckpt_dir, run_suffix)
        else:
            run_results_dir = base_results_dir

        os.makedirs(run_results_dir, exist_ok=True)
        os.makedirs(run_config.training.checkpoint_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"开始实验轮次 {run_idx}/{repeats} | seed={run_config.training.seed}")
        print("#" * 80)

        run_summary = _run_single_experiment(
            args=args,
            config=run_config,
            device=device,
            run_idx=run_idx,
            total_runs=repeats,
            results_dir_override=run_results_dir
        )
        all_runs.append(run_summary)

    if repeats > 1:
        summary_path = os.path.join(
            base_results_dir,
            f"date_random_repeat_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_index', 'seed', 'train_dates', 'test_dates',
                'test_loss', 'test_mae', 'test_rmse', 'test_mape', 'results_dir'
            ])
            for row in all_runs:
                writer.writerow([
                    row['run_index'],
                    row['seed'],
                    '|'.join(row['train_dates']) if row['train_dates'] else '',
                    '|'.join(row['test_dates']) if row['test_dates'] else '',
                    f"{row['test_loss']:.6f}",
                    f"{row['test_mae']:.6f}",
                    f"{row['test_rmse']:.6f}",
                    f"{row['test_mape']:.6f}",
                    row['results_dir'],
                ])

        print("\n" + "=" * 70)
        print("date_random 多次重复实验汇总")
        print("=" * 70)
        for row in all_runs:
            print(
                f"  Run {row['run_index']:02d}: seed={row['seed']}, "
                f"MAE={row['test_mae']:.4f}, RMSE={row['test_rmse']:.4f}, "
                f"MAPE={row['test_mape']:.2f}%, train_dates={row['train_dates']}"
            )
        print(f"✓ 汇总CSV已保存: {summary_path}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # for first down-sampling, then split
    # python main.py --use_attention --epochs 300 --lr 0.01 --dropout 0.1 --experiments 12101540_Tao --window_size 30 --downsample --downsample_interval 10 --mode window --model TCN --no_val --train_split 0.7 --test_split 0.3 --split_strategy random
    # for first split, then down-sampling
    # python main.py --smooth --time_smooth median --use_attention --epochs 50 --lr 0.001 --dropout 0.15 --weight_decay 0.01 --experiments 12101540_Tao --window_duration 600 --downsample --downsample_interval 10 --mode window --model TCN --no_val --train_split 0.7 --test_split 0.3 --split_strategy random
    # for all experiments
    # python main.py --use_attention --epochs 30 --lr 0.001 --dropout 0.1 --experiments all --window_size 50 --downsample --downsample_interval 5 --mode window --model TCN --normalize --no_val --train_split 0.5 --test_split 0.5 --split_strategy random
    # for training with nohup
    # nohup python main.py --use_attention --epochs 300 --lr 0.03 --dropout 0.1 --experiments all --window_size 20 --downsample --downsample_interval 5 --mode window --model TCN --no_val --train_split 0.7 --test_split 0.3 --split_strategy random &
    # for many batches
    # python run_batch.py experiments.txt
    main()

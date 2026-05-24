"""
配置管理模块
集中管理训练、模型、数据等所有配置参数
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础训练参数
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    weight_decay: float = 1e-4
    
    # 优化器和调度器
    optimizer: str = "Adam"  # Adam, SGD, AdamW
    scheduler: str = "CosineAnnealing"  # ReduceLROnPlateau, StepLR, CosineAnnealing
    patience: int = 5
    
    # 损失函数
    loss_function: str = "MSE"  # MSE, MAE, Huber
    huber_delta: float = 1.0  # Huber Loss的delta参数
    mae_window_loss_weight: float = 0.0  # 跨多个batch聚合后的MAE辅助损失权重（0表示关闭）
    mae_window_batches: int = 4  # 累积多少个batch后计算一次聚合MAE并更新
    
    # 随机种子
    seed: int = 42
    
    # 设备
    device: str = "cuda"  # cuda, cpu
    
    # W&B 配置
    wandb_entity: str = "glucose_msra"
    wandb_project: str = "sensing"
    wandb_enabled: bool = True
    
    # 保存和日志
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10  # 每多少个batch记录一次
    fusion_diagnostics: bool = False  # 是否记录gated residual的gate/correction诊断
    fusion_correction_loss_weight: float = 0.0  # gated residual修正量正则权重（0表示关闭）
    gated_residual_scale_mode: str = 'sigmoid'  # residual_scale约束方式: sigmoid(默认修复版), clamp(旧实现)
    gated_residual_min_scale: float = 0.02  # sigmoid模式下residual_scale下限，防止aux修正分支塌缩为0
    gated_residual_correction_scale: str = 'none'  # none, spectrum_rms, multiplicative
    gated_residual_differential_lr: bool = False  # 是否为base_model与late gated residual分支使用不同学习率
    gated_residual_fusion_lr: Optional[float] = None  # late fusion/aux/gate/residual分支学习率，None表示使用主lr
    gated_residual_scale_lr: Optional[float] = None  # residual_scale单独学习率，None表示使用fusion_lr


@dataclass
class ModelConfig:
    """模型相关配置"""
    # 模型架构
    architecture: str = "MLP"  # MLP, CNN, Transformer, TCN
    
    # 模型参数 (input_size 会从数据自动推断)
    input_size: Optional[int] = None
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.4
    use_attention: bool = True  # 是否使用注意力机制（频谱+时间注意力）
    aggregation: str = 'attention'  # 时间步聚合策略: attention, weighted, concat, lstm, mean, last
    use_dropblock: bool = True  # 是否使用DropBlock正则化（比Dropout更强）
    use_skip: bool = True  # 是否使用跳跃连接（频谱特征直连输出）
    
    # 任务类型
    task_type: str = "regression"  # regression, classification
    num_classes: Optional[int] = None  # 分类任务时使用


@dataclass
class DataConfig:
    """数据相关配置
    
    可用数据集概览 (所有时间均为北京时间):
    
    BIN格式数据 (data_source='bin'):
    - Tao: 15个实验 (11141404_Tao ~ 12161609_Tao)
    
    DB格式数据 (data_source='db') - 使用文件夹名作为实验名:
    - Tao_db: 8个实验 (北京时间)
        * DEV_002_qiangtao_260121 (1月21日 15:55-19:09 下午)
        * DEV_002_qiangtao_260123 (1月23日 12:29-18:03 下午)
        * DEV_002_qiangtao_260126 (1月26日 13:42-18:42 下午)
        * DEV_002_qiangtao_260127 (1月27日 13:34-19:46 下午)
        * DEV_002_qiangtao_260128 (1月28日 13:04-18:04 下午)
        * DEV_002_qiangtao_260129 (1月29日 14:17-16:59 下午)
        * DEV_002_qiangtao_260130 (1月30日 13:30-15:59 下午)
        * DEV_002_qiangtao_260202 (2月02日 14:15-16:30 下午)
    
    - Weiyi_db: 6个实验 (北京时间)
        * DEV_001_weiyi_260122 (1月22日 12:03-21:16 白天到晚上)
        * DEV_001_weiyi_260123 (1月23日 12:50-16:20 下午)
        * DEV_001_weiyi_260127 (1月27日 13:29-15:53 下午)
        * DEV_001_weiyi_260128 (1月28日 13:25-16:08 下午)
        * DEV_001_weiyi_260129 (1月29日 14:15-16:29 下午)
        * DEV_001_weiyi_260130 (1月30日 14:17-17:29 下午, 3个分段文件)
    
    - Yiheng_db: 3个实验 (北京时间)
        * DEV_002_yiheng_260122 (1月22日 12:02-18:00 白天)
        * DEV_002_yiheng_260126 (1月26日 18:51-21:34 晚上)
        * DEV_002_yiheng_260129 (1月29日 18:24-19:47 晚上, 2个分段文件)
    
    总计: 17个有效实验 (8+6+3)
    
    使用示例:
    --experiments DEV_002_qiangtao_260121 DEV_002_qiangtao_260123
    --db_users Tao_db Weiyi_db Yiheng_db
    
    注意: 所有时间为北京时间，内部会自动转换为UTC用于数据库查询
    """
    # 数据路径
    dataset_root: str = "./Dataset"
    user_name: str = "Tao"
    user_names: Optional[List[str]] = None  # 多用户（BIN格式）
    
    # 数据源选择
    data_source: str = "bin"  # 数据源格式: 'bin' (原始BIN格式) 或 'db' (SQLite数据库格式)
    db_user_name: str = "Tao_db"  # DB格式数据的用户目录名（默认: Tao_db）
    db_user_names: Optional[List[str]] = None  # 多用户（DB格式）- 可选: ["Tao_db", "Weiyi_db", "Yiheng_db"]
    
    # 数据选择
    experiment_filter: Optional[List[str]] = None  # 指定要加载的实验，None表示全部
    user_experiments: Optional[Dict[str, List[str]]] = None  # 用户实验映射（为每个用户指定不同的实验）
    
    # 数据划分
    use_val_set: bool = True  # 是否使用验证集
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    split_strategy: str = "random"  # random, temporal, stratified, experiment, date, date_random, max_day_train_min_day_test, alternating, hybrid_alternating, cross_user, multi_user_independent, db_file_temporal_80_20, tao_db_label_shuffle_debug
    n_splits: int = 10  # 交叉时序划分(alternating)的分段数
    date_random_repeats: int = 1  # date_random策略重复次数（建议5-10）
    date_train_days: Optional[int] = None  # date/date_random策略下训练日期数量（None表示按train_split推断）
    train_user_name: Optional[str] = None  # cross_user模式: 训练用户
    predict_user_name: Optional[str] = None  # cross_user模式: 预测用户
    predict_user_ratio: float = 1.0  # cross_user模式: 测试用户使用比例
    
    # 数据处理
    normalize: bool = False  # 是否归一化特征
    interpolation_method: str = "pchip"  # 插值方法: linear, cubic, pchip, akima, makima, nearest
                                          # 推荐pchip或akima用于CGM低采样率数据
    
    # 数据下采样
    downsample: bool = False  # 是否对数据进行下采样
    downsample_interval: float = 1.0  # 下采样间隔（秒），例如 1.0 表示每秒采样一次
    
    # 数据平滑处理（减少时间轴抖动和频谱噪声）
    smooth_data: bool = False  # 是否启用数据平滑
    time_smooth_method: str = 'none'  # 时间轴平滑方法: none, moving_average, ema, gaussian, savgol, median, wavelet
    time_smooth_window: int = 5  # 时间轴平滑窗口大小（用于moving_average, median, savgol）
    time_smooth_sigma: float = 2.0  # 高斯平滑标准差（用于gaussian）
    time_smooth_alpha: float = 0.3  # EMA平滑系数（用于ema），0-1之间，越小越平滑
    spectrum_smooth_method: str = 'none'  # 频谱轴平滑方法: none, moving_average, gaussian, savgol, median
    spectrum_smooth_window: int = 5  # 频谱轴平滑窗口大小
    spectrum_smooth_sigma: float = 2.0  # 频谱轴高斯平滑标准差
    
    # 异常值检测和处理
    detect_outliers: bool = False  # 是否检测异常值
    outlier_method: str = 'zscore'  # 异常值检测方法: zscore, iqr
    outlier_threshold: float = 3.0  # Z-score阈值
    handle_outliers: bool = False  # 是否处理异常值
    outlier_handle_method: str = 'interpolate'  # 异常值处理方法: interpolate, median, mean, clip
    
    # 数据融合
    data_fusion: bool = False  # 是否使用辅助传感器数据融合（aux sensor: BME680, PPG, T117, ICM）
    fusion_stage: str = 'early'  # 融合阶段: early(数据层/频谱层融合), late(特征层融合)
    
    # Early fusion方法 (数据层融合：1001频谱+14aux → 拼接/调制 → SpectralCNN)
    fusion_method: str = 'concat'  # early融合方法: concat, film, film_sensor_aware, attention_pool, residual_film, conditioned_rf, attention_fusion
    
    # Late fusion方法 (特征层融合：频谱→128维 + aux→32维 → 融合)
    # 支持的方法:
    #   - concat: 简单拼接+MLP (基线方法，快速但效果一般)
    #   - cross_attention: 交叉注意力融合 (推荐，自适应学习模态关联)
    #   - film: FiLM调制 (适合aux作为环境条件的场景)
    #   - cross_attention_film: 先Cross-Attention(非温湿度等动态aux)，再FiLM(温湿度等环境aux)
    #   - aux_cross_only: 仅使用去温湿度后的aux分支做预测（主模态不参与输出）
    #   - film_aux_only: 仅使用温湿度等aux进行FiLM调制（不使用其他aux传感器）
    #   - gated_residual_film: 先用PPG/ICM等动态aux门控残差修正spectrum，再用温湿度等环境aux做FiLM调制
    #   - ppg_icm_gated_residual_film: 明确使用PPG/ICM做gated residual，再用BME680/T117温湿度做FiLM调制
    #   - ppg_icm_gated_residual_only: 仅使用PPG/ICM engineered特征做gated residual，不做环境FiLM
    #   - ppg_icm_engineered_only: 仅使用engineered后的PPG/ICM特征进行预测，不使用spectrum/BME680/T117
    #   - ppg_icm_segment_raw_only: 仅使用PPG/ICM同一秒内原始序列槽位进行预测，不使用spectrum/BME680/T117
    #   - spectrum_only_late_head: 仅使用spectrum特征接late MLP预测头，用于验证late head本身收益
    fusion_method_late: str = 'concat'
    aux_encoded_dim: int = 32  # Late Fusion中aux特征编码后的维度
    late_film_aux_indices: Optional[List[int]] = None  # mixed late fusion中用于FiLM调制的aux索引
    aux_feature_mode: str = 'raw'  # raw(旧逻辑), engineered(PPG/ICM统计特征), both(两者拼接), segment_raw(PPG/ICM同秒原始序列槽位)
    
    icm_mode: str = 'raw'  # ICM数据模式: raw(原始加速度+陀螺仪), processed(位移+角度), both(两者都用)
    
    # Aux sensor覆盖（用于消融实验）
    aux_override: bool = False  # 是否覆盖aux sensor数值（用于测试/消融实验）
    aux_override_value: float = 0.0  # 覆盖aux sensor时使用的数值（默认0，可设为其他固定值）
    aux_shuffle: bool = False  # 是否打乱aux sensor与spectrum/label的对应关系（aux真实性验证）
    aux_shuffle_mode: str = 'sensor_groups'  # rows/features/sensor_groups
    aux_shuffle_seed: int = 42  # aux shuffle随机种子
    
    # Auto-Regressive (自回归) 模式配置
    autoregressive: bool = False  # 是否启用自回归模式（使用历史血糖数据作为输入）
    ar_glucose_history: int = 5  # 使用多少个历史血糖值作为输入
    ar_glucose_aligned: bool = True  # True: 历史血糖与当前spectrum时间对齐（当前血糖未知）; False: 包含当前时刻血糖
    ar_fusion_strategy: str = 'gating'  # AR融合策略: concat(拼接), film(调制), gating(门控，推荐)
    ar_glucose_encoder_dim: int = 32  # AR glucose encoder输出维度
    ar_smoothness_weight: float = 0.1  # 平滑性正则化权重，限制预测值跳变
    ar_history_weight: float = 0.05  # 历史一致性正则化权重，限制预测值与历史趋势偏离
    ar_test_interval: int = 5  # AR模式下真实推理测试间隔（每N个epoch测试一次，0表示每个epoch都测试）
    
    # 预测模式
    mode: str = "instant"  # instant: 当前spectrum预测当前血糖, window: 历史窗口预测当前血糖
    window_size: int = 10  # 窗口模式下使用的历史时间步数（样本数）
    window_duration: float = None  # 基于时间的窗口模式：窗口时长（秒），None表示使用window_size（样本数）
    window_padding: str = 'drop'  # 窗口前期样本处理: drop(丢弃), zero(零填充), repeat(重复填充), edge(边缘填充)
    
    # DataLoader 参数
    num_workers: int = 0  # Linux上建议设为0
    pin_memory: bool = True


@dataclass
class Config:
    """完整配置，包含所有子配置"""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """配置验证"""
        # 验证数据划分比例
        total = self.data.train_split + self.data.val_split + self.data.test_split
        assert abs(total - 1.0) < 1e-6, f"数据划分比例之和必须为1，当前为{total}"
        
        # 验证架构名称
        valid_archs = ["MLP", "CNN", "Transformer", "TCN"]
        assert self.model.architecture in valid_archs, \
            f"不支持的架构: {self.model.architecture}，可选: {valid_archs}"
    
    def to_dict(self):
        """转换为字典格式（用于 W&B 等）"""
        return {
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "weight_decay": self.training.weight_decay,
            "optimizer": self.training.optimizer,
            "scheduler": self.training.scheduler,
            "loss_function": self.training.loss_function,
            "huber_delta": self.training.huber_delta,
            "mae_window_loss_weight": self.training.mae_window_loss_weight,
            "mae_window_batches": self.training.mae_window_batches,
            "fusion_diagnostics": self.training.fusion_diagnostics,
            "fusion_correction_loss_weight": self.training.fusion_correction_loss_weight,
            "gated_residual_scale_mode": self.training.gated_residual_scale_mode,
            "gated_residual_min_scale": self.training.gated_residual_min_scale,
            "gated_residual_correction_scale": self.training.gated_residual_correction_scale,
            "gated_residual_differential_lr": self.training.gated_residual_differential_lr,
            "gated_residual_fusion_lr": self.training.gated_residual_fusion_lr,
            "gated_residual_scale_lr": self.training.gated_residual_scale_lr,
            "architecture": self.model.architecture,
            "hidden_size": self.model.hidden_size,
            "dropout": self.model.dropout,
            "input_size": self.model.input_size,
            "normalize": self.data.normalize,
            "user_name": self.data.user_name,
            "user_names": self.data.user_names,
            "data_source": self.data.data_source,
            "db_user_name": self.data.db_user_name,
            "db_user_names": self.data.db_user_names,
            "data_fusion": self.data.data_fusion,
            "fusion_stage": self.data.fusion_stage,
            "fusion_method": self.data.fusion_method,
            "fusion_method_late": self.data.fusion_method_late,
            "late_film_aux_indices": self.data.late_film_aux_indices,
            "aux_feature_mode": self.data.aux_feature_mode,
            "icm_mode": self.data.icm_mode,
            "aux_override": self.data.aux_override,
            "aux_override_value": self.data.aux_override_value,
            "aux_shuffle": self.data.aux_shuffle,
            "aux_shuffle_mode": self.data.aux_shuffle_mode,
            "aux_shuffle_seed": self.data.aux_shuffle_seed,
            # Auto-Regressive config
            "autoregressive": self.data.autoregressive,
            "ar_glucose_history": self.data.ar_glucose_history,
            "ar_glucose_aligned": self.data.ar_glucose_aligned,
            "ar_fusion_strategy": self.data.ar_fusion_strategy,
            "ar_glucose_encoder_dim": self.data.ar_glucose_encoder_dim,
            "ar_smoothness_weight": self.data.ar_smoothness_weight,
            "ar_history_weight": self.data.ar_history_weight,
            "date_random_repeats": self.data.date_random_repeats,
            "date_train_days": self.data.date_train_days,
            "split_strategy": self.data.split_strategy,
        }


def get_config() -> Config:
    """获取默认配置"""
    return Config()


def get_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config()
    
    # 更新训练配置（只在参数不为 None 时更新）
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.seed is not None:
        config.training.seed = args.seed
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if hasattr(args, 'loss') and args.loss is not None:
        config.training.loss_function = args.loss
    if hasattr(args, 'huber_delta') and args.huber_delta is not None:
        config.training.huber_delta = args.huber_delta
    if hasattr(args, 'mae_window_loss_weight') and args.mae_window_loss_weight is not None:
        config.training.mae_window_loss_weight = args.mae_window_loss_weight
    if hasattr(args, 'mae_window_batches') and args.mae_window_batches is not None:
        config.training.mae_window_batches = args.mae_window_batches
    if hasattr(args, 'fusion_diagnostics') and args.fusion_diagnostics:
        config.training.fusion_diagnostics = True
    if hasattr(args, 'fusion_correction_loss_weight') and args.fusion_correction_loss_weight is not None:
        config.training.fusion_correction_loss_weight = args.fusion_correction_loss_weight
    if hasattr(args, 'gated_residual_scale_mode') and args.gated_residual_scale_mode is not None:
        config.training.gated_residual_scale_mode = args.gated_residual_scale_mode
    if hasattr(args, 'gated_residual_min_scale') and args.gated_residual_min_scale is not None:
        config.training.gated_residual_min_scale = args.gated_residual_min_scale
    if hasattr(args, 'gated_residual_correction_scale') and args.gated_residual_correction_scale is not None:
        config.training.gated_residual_correction_scale = args.gated_residual_correction_scale
    if hasattr(args, 'gated_residual_differential_lr') and args.gated_residual_differential_lr:
        config.training.gated_residual_differential_lr = True
    if hasattr(args, 'gated_residual_fusion_lr') and args.gated_residual_fusion_lr is not None:
        config.training.gated_residual_fusion_lr = args.gated_residual_fusion_lr
    if hasattr(args, 'gated_residual_scale_lr') and args.gated_residual_scale_lr is not None:
        config.training.gated_residual_scale_lr = args.gated_residual_scale_lr
    
    # 更新模型配置
    if args.model is not None:
        config.model.architecture = args.model
    if args.hidden_size is not None:
        config.model.hidden_size = args.hidden_size
    if hasattr(args, 'num_layers') and args.num_layers is not None:
        config.model.num_layers = args.num_layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if hasattr(args, 'use_attention'):
        config.model.use_attention = args.use_attention
    if hasattr(args, 'aggregation') and args.aggregation is not None:
        config.model.aggregation = args.aggregation
    if hasattr(args, 'use_dropblock'):
        config.model.use_dropblock = args.use_dropblock
    if hasattr(args, 'use_skip'):
        config.model.use_skip = args.use_skip
    
    # 更新数据配置
    if hasattr(args, 'users') and args.users is not None:
        if isinstance(args.users, str):
            config.data.user_names = [u for u in args.users.split(',') if u]
        else:
            config.data.user_names = args.users
    if args.user is not None:
        config.data.user_name = args.user
    if args.normalize:
        config.data.normalize = args.normalize
    
    # 数据源配置
    if hasattr(args, 'data_source') and args.data_source is not None:
        config.data.data_source = args.data_source
    if hasattr(args, 'db_users') and args.db_users is not None:
        if isinstance(args.db_users, str):
            config.data.db_user_names = [u for u in args.db_users.split(',') if u]
        else:
            config.data.db_user_names = args.db_users
    
    # 处理用户实验映射 ⭐ NEW
    if hasattr(args, 'user_experiments') and args.user_experiments is not None:
        user_experiments = {}
        for user_exp_list in args.user_experiments:
            if len(user_exp_list) >= 2:
                user_name = user_exp_list[0]  # 第一个是用户名
                experiments = user_exp_list[1:]  # 剩余的是实验名
                user_experiments[user_name] = experiments
        config.data.user_experiments = user_experiments
    if hasattr(args, 'db_user') and args.db_user is not None:
        config.data.db_user_name = args.db_user
    
    if hasattr(args, 'data_fusion') and args.data_fusion:
        config.data.data_fusion = args.data_fusion
    if hasattr(args, 'fusion_stage') and args.fusion_stage is not None:
        config.data.fusion_stage = args.fusion_stage
    if hasattr(args, 'fusion_method') and args.fusion_method is not None:
        config.data.fusion_method = args.fusion_method
    if hasattr(args, 'fusion_method_late') and args.fusion_method_late is not None:
        config.data.fusion_method_late = args.fusion_method_late
    if hasattr(args, 'late_film_aux_indices') and args.late_film_aux_indices is not None:
        config.data.late_film_aux_indices = args.late_film_aux_indices
    if hasattr(args, 'aux_feature_mode') and args.aux_feature_mode is not None:
        config.data.aux_feature_mode = args.aux_feature_mode
    if hasattr(args, 'icm_mode') and args.icm_mode is not None:
        config.data.icm_mode = args.icm_mode
    if hasattr(args, 'aux_override') and args.aux_override:
        config.data.aux_override = args.aux_override
    if hasattr(args, 'aux_override_value') and args.aux_override_value is not None:
        config.data.aux_override_value = args.aux_override_value
    if hasattr(args, 'aux_shuffle') and args.aux_shuffle:
        config.data.aux_shuffle = True
    if hasattr(args, 'aux_shuffle_mode') and args.aux_shuffle_mode is not None:
        config.data.aux_shuffle_mode = args.aux_shuffle_mode
    if hasattr(args, 'aux_shuffle_seed') and args.aux_shuffle_seed is not None:
        config.data.aux_shuffle_seed = args.aux_shuffle_seed
    if hasattr(args, 'interpolation') and args.interpolation is not None:
        config.data.interpolation_method = args.interpolation
    if args.split_strategy is not None:
        config.data.split_strategy = args.split_strategy

    # date/date_random/max_day_train_min_day_test: 统一单用户入口
    if config.data.split_strategy in ['date', 'date_random', 'max_day_train_min_day_test']:
        date_user = getattr(args, 'date_user', None)
        data_source = getattr(config.data, 'data_source', 'bin')

        if data_source == 'db':
            # 优先级: --date_user > --db_user > --user
            if date_user:
                config.data.db_user_name = date_user
                config.data.db_user_names = None
            elif hasattr(args, 'db_user') and args.db_user is not None:
                config.data.db_user_name = args.db_user
                config.data.db_user_names = None
            elif hasattr(args, 'user') and args.user is not None:
                # 兼容写法：用户直接传 --user Tao_db
                config.data.db_user_name = args.user
                config.data.db_user_names = None

            if hasattr(args, 'db_users') and args.db_users is not None:
                if len(args.db_users) == 1:
                    config.data.db_user_name = args.db_users[0]
                    config.data.db_user_names = None
                else:
                    raise ValueError("date/date_random策略仅支持单用户，请使用 --db_user Tao_db 或 --date_user Tao_db")
        else:
            # BIN模式单用户
            if date_user:
                config.data.user_name = date_user
                config.data.user_names = None

            if hasattr(args, 'users') and args.users is not None:
                if len(args.users) == 1:
                    config.data.user_name = args.users[0]
                    config.data.user_names = None
                else:
                    raise ValueError("date/date_random策略仅支持单用户，请使用 --user Tao")
    if hasattr(args, 'train_user_name') and args.train_user_name is not None:
        config.data.train_user_name = args.train_user_name
    if hasattr(args, 'predict_user_name') and args.predict_user_name is not None:
        config.data.predict_user_name = args.predict_user_name
    if hasattr(args, 'predict_user_ratio') and args.predict_user_ratio is not None:
        config.data.predict_user_ratio = args.predict_user_ratio
    if hasattr(args, 'n_splits') and args.n_splits is not None:
        config.data.n_splits = args.n_splits
    if hasattr(args, 'date_random_repeats') and args.date_random_repeats is not None:
        config.data.date_random_repeats = args.date_random_repeats
    if hasattr(args, 'date_train_days') and args.date_train_days is not None:
        config.data.date_train_days = args.date_train_days
    
    # 应用下采样选项
    if hasattr(args, 'downsample') and args.downsample:
        config.data.downsample = True
    if hasattr(args, 'downsample_interval') and args.downsample_interval is not None:
        config.data.downsample_interval = args.downsample_interval
    
    # 应用平滑选项
    if hasattr(args, 'smooth') and args.smooth:
        config.data.smooth_data = True
    if hasattr(args, 'time_smooth') and args.time_smooth is not None:
        config.data.time_smooth_method = args.time_smooth
    if hasattr(args, 'time_smooth_window') and args.time_smooth_window is not None:
        config.data.time_smooth_window = args.time_smooth_window
    if hasattr(args, 'time_smooth_sigma') and args.time_smooth_sigma is not None:
        config.data.time_smooth_sigma = args.time_smooth_sigma
    if hasattr(args, 'time_smooth_alpha') and args.time_smooth_alpha is not None:
        config.data.time_smooth_alpha = args.time_smooth_alpha
    if hasattr(args, 'spectrum_smooth') and args.spectrum_smooth is not None:
        config.data.spectrum_smooth_method = args.spectrum_smooth
    if hasattr(args, 'spectrum_smooth_window') and args.spectrum_smooth_window is not None:
        config.data.spectrum_smooth_window = args.spectrum_smooth_window
    if hasattr(args, 'spectrum_smooth_sigma') and args.spectrum_smooth_sigma is not None:
        config.data.spectrum_smooth_sigma = args.spectrum_smooth_sigma
    
    # 应用异常值检测和处理选项
    if hasattr(args, 'detect_outliers') and args.detect_outliers:
        config.data.detect_outliers = True
    if hasattr(args, 'outlier_method') and args.outlier_method is not None:
        config.data.outlier_method = args.outlier_method
    if hasattr(args, 'outlier_threshold') and args.outlier_threshold is not None:
        config.data.outlier_threshold = args.outlier_threshold
    if hasattr(args, 'handle_outliers') and args.handle_outliers:
        config.data.handle_outliers = True
    if hasattr(args, 'outlier_handle_method') and args.outlier_handle_method is not None:
        config.data.outlier_handle_method = args.outlier_handle_method
    
    # 应用实验选择
    if hasattr(args, 'experiments') and args.experiments:
        # 如果指定 'all'，则设置为 None（表示使用所有实验）
        if len(args.experiments) == 1 and args.experiments[0].lower() == 'all':
            config.data.experiment_filter = None
        else:
            config.data.experiment_filter = args.experiments
    
    # 应用验证集选项
    if hasattr(args, 'no_val') and args.no_val:
        config.data.use_val_set = False
        # 无验证集时，强制val_split为0
        config.data.val_split = 0.0
        # 如果用户没有指定train_split和test_split，使用默认的80/20
        if args.train_split is None and args.test_split is None:
            config.data.train_split = 0.8
            config.data.test_split = 0.2
    
    # 应用预测模式
    if hasattr(args, 'mode') and args.mode is not None:
        config.data.mode = args.mode
    if hasattr(args, 'window_size') and args.window_size is not None:
        config.data.window_size = args.window_size
    if hasattr(args, 'window_duration') and args.window_duration is not None:
        config.data.window_duration = args.window_duration
    if hasattr(args, 'window_padding') and args.window_padding is not None:
        config.data.window_padding = args.window_padding
    
    # 应用数据划分比例
    if hasattr(args, 'train_split') and args.train_split is not None:
        config.data.train_split = args.train_split
    if hasattr(args, 'val_split') and args.val_split is not None:
        # 如果使用了--no_val，忽略val_split参数
        if not (hasattr(args, 'no_val') and args.no_val):
            config.data.val_split = args.val_split
    if hasattr(args, 'test_split') and args.test_split is not None:
        config.data.test_split = args.test_split
    
    # 应用Auto-Regressive选项
    if hasattr(args, 'autoregressive') and args.autoregressive:
        config.data.autoregressive = True
    if hasattr(args, 'ar_glucose_history') and args.ar_glucose_history is not None:
        config.data.ar_glucose_history = args.ar_glucose_history
    # 处理 --ar_aligned 和 --ar_no_aligned 的情况
    if hasattr(args, 'ar_glucose_aligned') and args.ar_glucose_aligned is not None:
        config.data.ar_glucose_aligned = args.ar_glucose_aligned
    if hasattr(args, 'ar_smoothness_weight') and args.ar_smoothness_weight is not None:
        config.data.ar_smoothness_weight = args.ar_smoothness_weight
    if hasattr(args, 'ar_history_weight') and args.ar_history_weight is not None:
        config.data.ar_history_weight = args.ar_history_weight
    if hasattr(args, 'ar_test_interval') and args.ar_test_interval is not None:
        config.data.ar_test_interval = args.ar_test_interval
    
    # 验证比例之和
    total = config.data.train_split + config.data.val_split + config.data.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"数据划分比例之和必须为1，当前为{total:.4f} (train={config.data.train_split}, val={config.data.val_split}, test={config.data.test_split})")
    if config.training.mae_window_loss_weight < 0:
        raise ValueError("mae_window_loss_weight 不能为负数")
    if config.training.mae_window_batches < 1:
        raise ValueError("mae_window_batches 必须 >= 1")
    
    return config

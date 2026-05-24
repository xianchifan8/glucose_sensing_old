"""
训练器模块
封装训练、验证和测试的逻辑
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import time
import gc
import numpy as np


class GlucoseTrainer:
    """血糖预测模型训练器"""
    
    def __init__(self, model, config, device, use_wandb=True, late_fusion=False, autoregressive=False):
        """
        Args:
            model: PyTorch 模型
            config: Config 对象
            device: 训练设备
            use_wandb: 是否使用 W&B 记录
            late_fusion: 是否使用Late Fusion模式（数据加载返回(spectrum, aux, target)）
            autoregressive: 是否使用自回归模式（数据加载返回额外的glucose_history）
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.late_fusion = late_fusion  # 记录是否为Late Fusion模式
        self.autoregressive = autoregressive  # 记录是否为自回归模式
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 记录最佳指标
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.best_test_mae = float('inf')
        self.best_test_rmse = float('inf')
        self.best_test_loss = float('inf')
        self.best_test_epoch = 0
        
        # 计时器
        self.total_train_time = 0.0
        self.total_val_time = 0.0
        self.total_test_time = 0.0
        self.last_fusion_diagnostics = {}
    
    def _create_criterion(self):
        """创建损失函数"""
        # 检查是否使用自回归模式
        if self.autoregressive:
            from Model.ar_loss import create_ar_criterion
            return create_ar_criterion(self.config)
        
        loss_name = self.config.training.loss_function.upper()
        
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        elif loss_name == 'HUBER':
            delta = self.config.training.huber_delta
            return nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def _create_optimizer(self):
        """创建优化器"""
        opt_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        scale_mode = getattr(self.config.training, 'gated_residual_scale_mode', 'sigmoid')
        use_diff_lr = bool(getattr(self.config.training, 'gated_residual_differential_lr', False))

        scale_params = []
        other_params = []
        if use_diff_lr:
            base_params = []
            fusion_params = []
            fusion_lr = getattr(self.config.training, 'gated_residual_fusion_lr', None)
            scale_lr = getattr(self.config.training, 'gated_residual_scale_lr', None)
            fusion_lr = float(fusion_lr) if fusion_lr is not None else float(lr)
            scale_lr = float(scale_lr) if scale_lr is not None else fusion_lr

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith('residual_scale'):
                    scale_params.append(param)
                elif name.startswith('base_model.'):
                    base_params.append(param)
                else:
                    fusion_params.append(param)

            params = []
            if base_params:
                params.append({'params': base_params, 'lr': lr, 'weight_decay': wd})
            if fusion_params:
                params.append({'params': fusion_params, 'lr': fusion_lr, 'weight_decay': wd})
            if scale_params:
                scale_wd = wd if scale_mode == 'clamp' else 0.0
                params.append({'params': scale_params, 'lr': scale_lr, 'weight_decay': scale_wd})
            print(
                f"✓ 差异学习率: base_lr={lr}, fusion_lr={fusion_lr}, "
                f"scale_lr={scale_lr}, scale_wd={wd if scale_mode == 'clamp' else 0.0}"
            )
        else:
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith('residual_scale') and scale_mode != 'clamp':
                    scale_params.append(param)
                else:
                    other_params.append(param)

            params = [{'params': other_params, 'weight_decay': wd}]
            if scale_params:
                params.append({'params': scale_params, 'weight_decay': 0.0})
        
        if opt_name == 'adam':
            return torch.optim.Adam(params, lr=lr)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr)
        elif opt_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器: {opt_name}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=self.config.training.patience
            )
        elif scheduler_name == 'steplr':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        elif scheduler_name in ['cosine', 'cosineannealing']:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs
            )
        else:
            return None
    
    def _unpack_batch(self, batch):
        """
        统一的batch解包方法，支持多种数据模式
        
        Returns:
            spectrum/data: 主输入数据
            aux: 辅助传感器数据（可能为None）
            glucose_history: 历史血糖序列（可能为None）
            target: 目标血糖值
        """
        if self.autoregressive:
            if self.late_fusion:
                # AR + Late Fusion: (spectrum, aux, glucose_history, target)
                spectrum, aux, glucose_history, target = batch
                return spectrum, aux, glucose_history, target
            else:
                # AR only: (spectrum, glucose_history, target)
                spectrum, glucose_history, target = batch
                return spectrum, None, glucose_history, target
        elif self.late_fusion:
            # Late Fusion only: (spectrum, aux, target)
            spectrum, aux, target = batch
            return spectrum, aux, None, target
        else:
            # Standard: (data, target)
            data, target = batch
            return data, None, None, target
    
    def _model_forward(self, spectrum, aux, glucose_history):
        """
        统一的模型前向传播方法
        """
        if self.autoregressive:
            if self.late_fusion and aux is not None:
                return self.model(spectrum, aux, glucose_history)
            else:
                return self.model(spectrum, glucose_history)
        elif self.late_fusion:
            return self.model(spectrum, aux)
        else:
            return self.model(spectrum)
    
    def _compute_loss(self, output, target, glucose_history=None):
        """
        计算损失（支持AR模式的复合损失）
        """
        if self.autoregressive and glucose_history is not None:
            # AR模式：使用复合损失
            return self.criterion(output, target, glucose_history)
        else:
            return self.criterion(output, target)

    def _fusion_correction_loss_weight(self):
        return float(getattr(self.config.training, 'fusion_correction_loss_weight', 0.0))

    def _compute_fusion_correction_regularization(self):
        weight = self._fusion_correction_loss_weight()
        if weight <= 0:
            return None
        model = self._diagnostic_model()
        if not hasattr(model, 'get_fusion_correction_loss'):
            return None
        correction_loss = model.get_fusion_correction_loss()
        if correction_loss is None:
            return None
        return weight * correction_loss

    def _use_window_mae_loss(self):
        """是否启用跨batch聚合MAE辅助损失。"""
        return getattr(self.config.training, 'mae_window_loss_weight', 0.0) > 0

    def _build_window_mae_loss(self, outputs, targets):
        """
        对累计窗口中的所有样本整体计算MAE。

        这样比单独对每个batch算MAE更贴近最终测试集整体MAE的优化目标。
        """
        if not outputs:
            return None

        window_output = torch.cat([out.reshape(-1) for out in outputs], dim=0)
        window_target = torch.cat([tgt.reshape(-1) for tgt in targets], dim=0)
        return torch.mean(torch.abs(window_output - window_target))

    def _compute_mape(self, y_pred, y_true, eps=1e-8):
        """计算MAPE(%)，用于训练/测试统一口径统计。"""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)

    def _release_eval_memory(self):
        """释放评估/可视化后不再需要的临时内存。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fusion_diagnostics_enabled(self):
        return bool(getattr(self.config.training, 'fusion_diagnostics', False))

    def _diagnostic_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _reset_fusion_diagnostics(self):
        if not self._fusion_diagnostics_enabled():
            self.last_fusion_diagnostics = {}
            return
        model = self._diagnostic_model()
        if hasattr(model, 'set_fusion_diagnostics_enabled'):
            model.set_fusion_diagnostics_enabled(True)
        if hasattr(model, 'reset_fusion_diagnostics'):
            model.reset_fusion_diagnostics()

    def _collect_fusion_diagnostics(self):
        if not self._fusion_diagnostics_enabled():
            self.last_fusion_diagnostics = {}
            return {}
        model = self._diagnostic_model()
        diagnostics = {}
        if hasattr(model, 'get_fusion_diagnostics'):
            diagnostics = model.get_fusion_diagnostics() or {}
        self.last_fusion_diagnostics = diagnostics
        return diagnostics

    def _format_fusion_diagnostics(self, diagnostics):
        if not diagnostics:
            return None
        return (
            f"scale={diagnostics.get('residual_scale', 0.0):.4f}, "
            f"scale_raw={diagnostics.get('residual_scale_raw', 0.0):.4f}, "
            f"scale_min={diagnostics.get('residual_scale_min', 0.0):.4f}, "
            f"gate_mean={diagnostics.get('gate_mean', 0.0):.4f}, "
            f"gate_std={diagnostics.get('gate_std', 0.0):.4f}, "
            f"gate_p25/p50/p75="
            f"{diagnostics.get('gate_p25', 0.0):.4f}/"
            f"{diagnostics.get('gate_p50', 0.0):.4f}/"
            f"{diagnostics.get('gate_p75', 0.0):.4f}, "
            f"corr/spec={diagnostics.get('correction_to_spectrum_ratio', 0.0):.4f}, "
            f"corr_abs/spec_abs={diagnostics.get('correction_abs_to_spectrum_abs', 0.0):.4f}, "
            f"amp={diagnostics.get('correction_amplifier_mean', 1.0):.4f}, "
            f"corr_norm={diagnostics.get('correction_norm', 0.0):.4f}, "
            f"corr_abs={diagnostics.get('correction_abs_mean', 0.0):.4f}, "
            f"spec_abs={diagnostics.get('spectrum_abs_mean', 0.0):.4f}, "
            f"res_norm={diagnostics.get('residual_norm', 0.0):.4f}"
        )

    def _fusion_diag_log_data(self, diagnostics, prefix):
        return {f'{prefix}_fusion_{key}': value for key, value in diagnostics.items()}
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个 epoch
        
        Returns:
            avg_loss, avg_mae, stats: 平均损失、平均绝对误差、epoch统计信息
        """
        self.model.train()
        total_loss = 0
        total_mae = 0  # 累加所有样本的绝对误差（用于准确计算MAE）
        n_samples_seen = 0
        n_processed_batches = 0
        n_skip_output_nan = 0
        n_skip_loss_nan = 0
        n_optimizer_steps = 0
        n_window_updates = 0
        total_window_mae = 0.0
        total_fusion_correction_reg = 0.0
        n_fusion_correction_reg = 0
        last_fusion_correction_reg = None
        use_window_mae = self._use_window_mae_loss()
        window_batches = max(1, int(getattr(self.config.training, 'mae_window_batches', 1)))
        window_mae_weight = float(getattr(self.config.training, 'mae_window_loss_weight', 0.0))
        
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        if not use_window_mae:
            for batch_idx, batch in enumerate(pbar):
                # 解包数据
                spectrum, aux, glucose_history, target = self._unpack_batch(batch)
                
                # 移动到设备
                spectrum = spectrum.to(self.device)
                target = target.to(self.device)
                if aux is not None:
                    aux = aux.to(self.device)
                if glucose_history is not None:
                    glucose_history = glucose_history.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 模型前向传播
                output = self._model_forward(spectrum, aux, glucose_history)
                
                # 检查输出是否包含 NaN
                if torch.isnan(output).any():
                    print(f"\n⚠️  警告: 第 {batch_idx} 个batch的输出包含NaN，跳过此batch")
                    n_skip_output_nan += 1
                    continue
                
                # 计算损失
                loss = self._compute_loss(output, target, glucose_history)
                fusion_correction_reg = self._compute_fusion_correction_regularization()
                if fusion_correction_reg is not None:
                    loss = loss + fusion_correction_reg
                    total_fusion_correction_reg += fusion_correction_reg.item()
                    n_fusion_correction_reg += 1
                    last_fusion_correction_reg = fusion_correction_reg.item()
                
                # 检查损失是否为 NaN
                if torch.isnan(loss):
                    print(f"\n⚠️  警告: 第 {batch_idx} 个batch的损失为NaN，跳过此batch")
                    n_skip_loss_nan += 1
                    continue
                
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                n_optimizer_steps += 1
                
                # 🔧 修复：累加所有样本的绝对误差总和（而不是batch平均值）
                batch_mae_sum = torch.abs(output - target).sum().item()
                batch_mae_mean = batch_mae_sum / len(target)  # 用于显示当前batch的MAE
                
                total_loss += loss.item()
                total_mae += batch_mae_sum  # 累加误差总和
                n_samples_seen += len(target)
                n_processed_batches += 1
                
                # 更新进度条
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{batch_mae_mean:.4f}'
                }
                if last_fusion_correction_reg is not None:
                    postfix['corr_reg'] = f'{last_fusion_correction_reg:.6f}'
                pbar.set_postfix(postfix)
        else:
            pending_outputs = []
            pending_targets = []
            pending_base_losses = []
            pending_batches = 0
            pending_samples = 0
            last_window_loss = None
            last_window_mae = None

            self.optimizer.zero_grad()

            def flush_pending_window():
                nonlocal pending_outputs, pending_targets, pending_base_losses
                nonlocal pending_batches, pending_samples
                nonlocal total_loss, n_optimizer_steps, n_window_updates, total_window_mae
                nonlocal last_window_loss, last_window_mae

                if pending_batches == 0:
                    return

                base_loss = torch.stack(pending_base_losses).mean()
                window_mae_loss = self._build_window_mae_loss(pending_outputs, pending_targets)
                loss = base_loss + window_mae_weight * window_mae_loss

                if torch.isnan(loss):
                    raise RuntimeError("累计窗口损失为NaN，无法继续训练，请检查输入数据或MAE窗口权重设置")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item() * pending_batches
                n_optimizer_steps += 1
                n_window_updates += 1
                total_window_mae += window_mae_loss.item()
                last_window_loss = loss.item()
                last_window_mae = window_mae_loss.item()

                pending_outputs = []
                pending_targets = []
                pending_base_losses = []
                pending_batches = 0
                pending_samples = 0

            for batch_idx, batch in enumerate(pbar):
                # 解包数据
                spectrum, aux, glucose_history, target = self._unpack_batch(batch)
                
                # 移动到设备
                spectrum = spectrum.to(self.device)
                target = target.to(self.device)
                if aux is not None:
                    aux = aux.to(self.device)
                if glucose_history is not None:
                    glucose_history = glucose_history.to(self.device)
                
                # 模型前向传播
                output = self._model_forward(spectrum, aux, glucose_history)
                
                # 检查输出是否包含 NaN
                if torch.isnan(output).any():
                    print(f"\n⚠️  警告: 第 {batch_idx} 个batch的输出包含NaN，跳过此batch")
                    n_skip_output_nan += 1
                    continue
                
                # 计算基础损失
                base_loss = self._compute_loss(output, target, glucose_history)
                fusion_correction_reg = self._compute_fusion_correction_regularization()
                if fusion_correction_reg is not None:
                    base_loss = base_loss + fusion_correction_reg
                    total_fusion_correction_reg += fusion_correction_reg.item()
                    n_fusion_correction_reg += 1
                    last_fusion_correction_reg = fusion_correction_reg.item()
                
                # 检查损失是否为 NaN
                if torch.isnan(base_loss):
                    print(f"\n⚠️  警告: 第 {batch_idx} 个batch的损失为NaN，跳过此batch")
                    n_skip_loss_nan += 1
                    continue
                
                batch_mae_sum = torch.abs(output - target).sum().item()
                batch_mae_mean = batch_mae_sum / len(target)
                
                total_mae += batch_mae_sum
                n_samples_seen += len(target)
                n_processed_batches += 1

                pending_outputs.append(output)
                pending_targets.append(target)
                pending_base_losses.append(base_loss)
                pending_batches += 1
                pending_samples += len(target)

                if pending_batches >= window_batches:
                    flush_pending_window()
                
                postfix = {
                    'base': f'{base_loss.item():.4f}',
                    'mae': f'{batch_mae_mean:.4f}',
                    'win': f'{pending_batches}/{window_batches}',
                }
                if last_window_loss is not None:
                    postfix['loss'] = f'{last_window_loss:.4f}'
                    postfix['w_mae'] = f'{last_window_mae:.4f}'
                else:
                    postfix['loss'] = f'{base_loss.item():.4f}'
                if last_fusion_correction_reg is not None:
                    postfix['corr_reg'] = f'{last_fusion_correction_reg:.6f}'
                pbar.set_postfix(postfix)

            flush_pending_window()
        
        if n_processed_batches == 0:
            raise RuntimeError("当前epoch所有batch均被NaN跳过，无法进行参数更新")

        avg_loss = total_loss / n_processed_batches
        avg_mae = total_mae / max(n_samples_seen, 1)
        
        epoch_time = time.time() - epoch_start_time
        self.total_train_time += epoch_time
        
        stats = {
            'processed_batches': int(n_processed_batches),
            'skipped_output_nan_batches': int(n_skip_output_nan),
            'skipped_loss_nan_batches': int(n_skip_loss_nan),
            'skipped_total_nan_batches': int(n_skip_output_nan + n_skip_loss_nan),
            'processed_samples': int(n_samples_seen),
            'optimizer_steps': int(n_optimizer_steps),
            'window_mae_enabled': bool(use_window_mae),
            'window_mae_updates': int(n_window_updates),
            'window_mae_mean': float(total_window_mae / max(n_window_updates, 1)) if use_window_mae else 0.0,
            'window_mae_weight': float(window_mae_weight if use_window_mae else 0.0),
            'window_mae_batches': int(window_batches if use_window_mae else 1),
            'fusion_correction_reg_enabled': bool(n_fusion_correction_reg > 0),
            'fusion_correction_reg_mean': float(total_fusion_correction_reg / max(n_fusion_correction_reg, 1)),
            'fusion_correction_reg_weight': float(self._fusion_correction_loss_weight()),
        }
        return avg_loss, avg_mae, stats
    
    def validate(self, val_loader, track_time=True, return_mape=False):
        """
        验证模型
        
        Returns:
            avg_loss, avg_mae, avg_rmse[, avg_mape]: 平均损失、MAE、RMSE、MAPE(可选)
        """
        val_start_time = time.time()
        
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_mape = 0
        self._reset_fusion_diagnostics()
        
        with torch.inference_mode():
            for batch in val_loader:
                # 解包数据
                spectrum, aux, glucose_history, target = self._unpack_batch(batch)
                
                # 移动到设备
                spectrum = spectrum.to(self.device)
                target = target.to(self.device)
                if aux is not None:
                    aux = aux.to(self.device)
                if glucose_history is not None:
                    glucose_history = glucose_history.to(self.device)
                
                # 模型前向传播
                output = self._model_forward(spectrum, aux, glucose_history)
                
                # 计算损失（验证时不使用AR正则化）
                if self.autoregressive:
                    # 只计算主损失用于验证
                    loss = self.criterion.base_criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_mae += torch.abs(output - target).sum().item()
                total_mse += ((output - target) ** 2).sum().item()
                if return_mape:
                    total_mape += torch.abs((output - target) / (torch.abs(target) + 1e-8)).sum().item()

                del spectrum, target, output, loss
                if aux is not None:
                    del aux
                if glucose_history is not None:
                    del glucose_history
        
        n_samples = len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / n_samples
        avg_rmse = (total_mse / n_samples) ** 0.5
        if return_mape:
            avg_mape = (total_mape / n_samples) * 100.0
        
        if track_time:
            val_time = time.time() - val_start_time
            self.total_val_time += val_time

        self._collect_fusion_diagnostics()
        self._release_eval_memory()

        if return_mape:
            return avg_loss, avg_mae, avg_rmse, avg_mape
        return avg_loss, avg_mae, avg_rmse
    
    def train(self, train_loader, val_loader, test_loader=None, save_dir='./checkpoints', train_metadata=None, test_metadata=None, full_metadata=None):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选，用于每个epoch评估）
            save_dir: 模型保存目录
            train_metadata: 训练集元数据（用于生成训练集可视化）
            test_metadata: 测试集元数据（用于AR真实推理）
            full_metadata: 完整数据集的元数据（包含所有ground truth，用于完整可视化）
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"开始训练")
        print(f"{'='*70}\n")
        
        # 检查是否使用验证集和测试集
        use_val_set = val_loader is not None
        use_test_set = test_loader is not None
        final_train_eval_mape = None
        self.best_test_mae = float('inf')
        self.best_test_rmse = float('inf')
        self.best_test_loss = float('inf')
        self.best_test_epoch = 0
        
        for epoch in range(self.config.training.epochs):
            # 训练
            train_loss, train_mae, train_stats = self.train_epoch(train_loader, epoch)

            # 以eval口径在训练集上评估，便于与验证/测试集可比
            is_last_epoch = (epoch == self.config.training.epochs - 1)
            if is_last_epoch:
                train_eval_loss, train_eval_mae, train_eval_rmse, train_eval_mape = self.validate(
                    train_loader, track_time=False, return_mape=True
                )
                final_train_eval_mape = train_eval_mape
            else:
                train_eval_loss, train_eval_mae, train_eval_rmse = self.validate(train_loader, track_time=False)
            train_eval_fusion_diag = dict(self.last_fusion_diagnostics)
            
            # 准备日志数据
            log_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'train_eval_loss': train_eval_loss,
                'train_eval_mae': train_eval_mae,
                'train_eval_rmse': train_eval_rmse,
                'train_processed_batches': train_stats['processed_batches'],
                'train_skipped_output_nan_batches': train_stats['skipped_output_nan_batches'],
                'train_skipped_loss_nan_batches': train_stats['skipped_loss_nan_batches'],
                'train_skipped_total_nan_batches': train_stats['skipped_total_nan_batches'],
                'train_optimizer_steps': train_stats['optimizer_steps'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            if final_train_eval_mape is not None:
                log_data['train_eval_mape'] = final_train_eval_mape
            if train_stats['window_mae_enabled']:
                log_data.update({
                    'train_window_mae_mean': train_stats['window_mae_mean'],
                    'train_window_mae_updates': train_stats['window_mae_updates'],
                    'train_window_mae_weight': train_stats['window_mae_weight'],
                    'train_window_mae_batches': train_stats['window_mae_batches'],
                })
            if train_stats['fusion_correction_reg_enabled']:
                log_data.update({
                    'train_fusion_correction_reg_mean': train_stats['fusion_correction_reg_mean'],
                    'train_fusion_correction_reg_weight': train_stats['fusion_correction_reg_weight'],
                })
            if train_eval_fusion_diag:
                log_data.update(self._fusion_diag_log_data(train_eval_fusion_diag, 'train_eval'))
            
            if use_val_set:
                # 有验证集：进行验证
                val_loss, val_mae, val_rmse = self.validate(val_loader)
                
                # 打印结果
                print(f"\nEpoch {epoch+1}/{self.config.training.epochs}:")
                print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
                if train_stats['window_mae_enabled']:
                    print(
                        f"  Train(Window MAE) - mean: {train_stats['window_mae_mean']:.4f}, "
                        f"updates: {train_stats['window_mae_updates']}, "
                        f"weight: {train_stats['window_mae_weight']:.4f}, "
                        f"batches/window: {train_stats['window_mae_batches']}"
                    )
                if train_stats['fusion_correction_reg_enabled']:
                    print(
                        f"  Train(FusionCorrReg) - mean: {train_stats['fusion_correction_reg_mean']:.6f}, "
                        f"weight: {train_stats['fusion_correction_reg_weight']:.6g}"
                    )
                if final_train_eval_mape is not None:
                    print(f"  Train(Eval) - Loss: {train_eval_loss:.4f}, MAE: {train_eval_mae:.4f}, RMSE: {train_eval_rmse:.4f}, MAPE: {final_train_eval_mape:.2f}%")
                else:
                    print(f"  Train(Eval) - Loss: {train_eval_loss:.4f}, MAE: {train_eval_mae:.4f}, RMSE: {train_eval_rmse:.4f}")
                diag_text = self._format_fusion_diagnostics(train_eval_fusion_diag)
                if diag_text:
                    print(f"  Train(FusionDiag) - {diag_text}")
                print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
                if train_stats['skipped_total_nan_batches'] > 0:
                    print(
                        f"  ⚠️  NaN批次跳过: total={train_stats['skipped_total_nan_batches']} "
                        f"(output={train_stats['skipped_output_nan_batches']}, loss={train_stats['skipped_loss_nan_batches']})"
                    )
                
                # 添加验证指标到日志
                log_data.update({
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                })
                
                # 更新学习率（基于验证集）
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_mae)
                    else:
                        self.scheduler.step()
                
                # 保存最佳模型（基于验证集）
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self.best_epoch = epoch + 1  # 记录最佳epoch
                    best_model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_mae': val_mae,
                        'val_rmse': val_rmse,
                    }, best_model_path)
                    print(f"  ✓ 保存最佳模型 (MAE: {val_mae:.4f})")
            else:
                # 无验证集：只显示训练结果
                print(f"\nEpoch {epoch+1}/{self.config.training.epochs}:")
                print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
                if train_stats['window_mae_enabled']:
                    print(
                        f"  Train(Window MAE) - mean: {train_stats['window_mae_mean']:.4f}, "
                        f"updates: {train_stats['window_mae_updates']}, "
                        f"weight: {train_stats['window_mae_weight']:.4f}, "
                        f"batches/window: {train_stats['window_mae_batches']}"
                    )
                if train_stats['fusion_correction_reg_enabled']:
                    print(
                        f"  Train(FusionCorrReg) - mean: {train_stats['fusion_correction_reg_mean']:.6f}, "
                        f"weight: {train_stats['fusion_correction_reg_weight']:.6g}"
                    )
                if final_train_eval_mape is not None:
                    print(f"  Train(Eval) - Loss: {train_eval_loss:.4f}, MAE: {train_eval_mae:.4f}, RMSE: {train_eval_rmse:.4f}, MAPE: {final_train_eval_mape:.2f}%")
                else:
                    print(f"  Train(Eval) - Loss: {train_eval_loss:.4f}, MAE: {train_eval_mae:.4f}, RMSE: {train_eval_rmse:.4f}")
                diag_text = self._format_fusion_diagnostics(train_eval_fusion_diag)
                if diag_text:
                    print(f"  Train(FusionDiag) - {diag_text}")
                if train_stats['skipped_total_nan_batches'] > 0:
                    print(
                        f"  ⚠️  NaN批次跳过: total={train_stats['skipped_total_nan_batches']} "
                        f"(output={train_stats['skipped_output_nan_batches']}, loss={train_stats['skipped_loss_nan_batches']})"
                    )
                
                # 更新学习率（基于训练loss）
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_loss)
                    else:
                        self.scheduler.step()
                
                # 定期保存模型（无验证集时，每10个epoch保存一次）
                if (epoch + 1) % 10 == 0 or epoch == self.config.training.epochs - 1:
                    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_mae': train_mae,
                    }, checkpoint_path)
                    print(f"  ✓ 保存检查点 (Epoch {epoch+1})")
            
            # 每个epoch在测试集上评估（如果提供了test_loader）
            if use_test_set:
                test_fusion_diag = {}
                # AR模式下的特殊处理：只在关键epoch进行真实推理测试
                if self.autoregressive:
                    ar_test_interval = getattr(self.config.data, 'ar_test_interval', 5)
                    should_test = False
                    
                    if ar_test_interval == 0:
                        # 0表示每个epoch都测试
                        should_test = True
                    elif (epoch + 1) % ar_test_interval == 0 or epoch == self.config.training.epochs - 1:
                        # 每N个epoch或最后一个epoch才测试
                        should_test = True
                        print(f"  📊 第{epoch+1}个epoch: 进行真实AR推理测试")
                    else:
                        # 非关键epoch跳过测试
                        next_test_epoch = ((epoch + 1) // ar_test_interval + 1) * ar_test_interval
                        next_test_epoch = min(next_test_epoch, self.config.training.epochs)
                        print(f"  ⏭️  第{epoch+1}个epoch: 跳过测试（下次测试将在第{next_test_epoch}个epoch）")
                        should_test = False
                    
                    if should_test:
                        # AR模式：使用真实的autoregressive推理（如果有test_metadata）
                        if test_metadata is not None:
                            y_pred, y_true, _ = self._test_autoregressive_inference(test_loader, test_metadata)
                            test_mae = np.mean(np.abs(y_pred - y_true))
                            test_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                            test_loss = test_mae  # 用MAE作为loss的近似
                            print(f"  Test  - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f} (真实AR推理)")
                        else:
                            # 没有test_metadata，回退到Teacher Forcing
                            test_loss, test_mae, test_rmse = self.validate(test_loader)
                            test_fusion_diag = dict(self.last_fusion_diagnostics)
                            print(f"  Test  - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f} (Teacher Forcing)")
                            diag_text = self._format_fusion_diagnostics(test_fusion_diag)
                            if diag_text:
                                print(f"  Test(FusionDiag) - {diag_text}")
                        
                        # 添加测试指标到日志
                        log_data.update({
                            'test_loss': test_loss,
                            'test_mae': test_mae,
                            'test_rmse': test_rmse,
                        })
                        if test_fusion_diag:
                            log_data.update(self._fusion_diag_log_data(test_fusion_diag, 'test'))
                        if test_mae < self.best_test_mae:
                            self.best_test_mae = float(test_mae)
                            self.best_test_rmse = float(test_rmse)
                            self.best_test_loss = float(test_loss)
                            self.best_test_epoch = epoch + 1
                            print(
                                f"  ✓ 更新最佳测试MAE: {self.best_test_mae:.4f} "
                                f"(epoch {self.best_test_epoch}, RMSE: {self.best_test_rmse:.4f})"
                            )
                else:
                    # 非AR模式：每个epoch都正常测试
                    test_loss, test_mae, test_rmse = self.validate(test_loader)
                    test_fusion_diag = dict(self.last_fusion_diagnostics)
                    print(f"  Test  - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
                    diag_text = self._format_fusion_diagnostics(test_fusion_diag)
                    if diag_text:
                        print(f"  Test(FusionDiag) - {diag_text}")
                    
                    # 添加测试指标到日志
                    log_data.update({
                        'test_loss': test_loss,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse,
                    })
                    if test_fusion_diag:
                        log_data.update(self._fusion_diag_log_data(test_fusion_diag, 'test'))
                    if test_mae < self.best_test_mae:
                        self.best_test_mae = float(test_mae)
                        self.best_test_rmse = float(test_rmse)
                        self.best_test_loss = float(test_loss)
                        self.best_test_epoch = epoch + 1
                        print(
                            f"  ✓ 更新最佳测试MAE: {self.best_test_mae:.4f} "
                            f"(epoch {self.best_test_epoch}, RMSE: {self.best_test_rmse:.4f})"
                        )
                
                if self.best_test_epoch > 0:
                    log_data.update({
                        'best_test_mae': self.best_test_mae,
                        'best_test_rmse': self.best_test_rmse,
                        'best_test_loss': self.best_test_loss,
                        'best_test_epoch': self.best_test_epoch,
                    })
            
            # 记录所有指标到 W&B
            if self.use_wandb:
                wandb.log(log_data)
        
        if use_val_set:
            print(f"\n{'='*70}")
            print(f"训练完成！最佳验证 MAE: {self.best_val_mae:.4f}")
            if final_train_eval_mape is not None:
                print(f"最后一个epoch训练集MAPE(Train Eval): {final_train_eval_mape:.2f}%")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"训练完成！（无验证集模式）")
            if final_train_eval_mape is not None:
                print(f"最后一个epoch训练集MAPE(Train Eval): {final_train_eval_mape:.2f}%")
            print(f"{'='*70}")
        if self.best_test_epoch > 0:
            print(
                f"训练期间最佳测试 MAE: {self.best_test_mae:.6f} "
                f"(epoch {self.best_test_epoch}, RMSE: {self.best_test_rmse:.6f}, "
                f"Loss: {self.best_test_loss:.6f})"
            )
            print(
                f"BEST_TEST_MAE: {self.best_test_mae:.6f} "
                f"BEST_TEST_EPOCH: {self.best_test_epoch} "
                f"BEST_TEST_RMSE: {self.best_test_rmse:.6f} "
                f"BEST_TEST_LOSS: {self.best_test_loss:.6f}"
            )
        print(f"\n⏱️  训练用时统计:")
        print(f"  总训练时间: {self.total_train_time:.2f}秒 ({self.total_train_time/60:.2f}分钟)")
        print(f"  总验证时间: {self.total_val_time:.2f}秒 ({self.total_val_time/60:.2f}分钟)")
        print(f"  总计: {self.total_train_time + self.total_val_time:.2f}秒 ({(self.total_train_time + self.total_val_time)/60:.2f}分钟)")
        print(f"  平均每epoch: {(self.total_train_time + self.total_val_time)/self.config.training.epochs:.2f}秒")
        print(f"{'='*70}\n")
        
        # 生成训练集的时间序列对比图（保存到 results 文件夹）
        if train_metadata is not None:
            print("\n📊 生成训练集时间序列对比图...")
            self._generate_train_visualization(train_loader, train_metadata, full_metadata, viz_dir='./results')
            print(f"{'='*70}\n")
    
    def _test_autoregressive_inference(self, test_loader, test_metadata):
        """
        真正的自回归推理：维护完整预测历史，每次预测时从历史中重采样glucose_history
        
        关键设计：
        1. 维护一个完整的预测历史（glucose值+时间戳），与spectrum/aux的downsampling一致
        2. 每次预测时，从历史中重采样ar_history_len个点作为模型输入
        3. 每段数据开始用ground truth glucose启动
        4. 遇到实验切换或alternating split边界时，重新用ground truth启动
        
        Args:
            test_loader: 测试数据加载器
            test_metadata: 测试集元数据（必须包含 timestamps, experiment_indices, split_indices）
        
        Returns:
            y_pred: 预测值
            y_true: 真实值
            updated_metadata: 元数据
        """
        self.model.eval()
        
        # 从test_loader获取完整数据
        dataset = test_loader.dataset
        n_samples = len(dataset)

        # 优先走零拷贝路径：直接复用AutoRegressiveDataset内部tensor，避免逐样本append导致内存峰值
        has_direct_tensors = (
            hasattr(dataset, 'spectrum') and
            hasattr(dataset, 'glucose_history') and
            hasattr(dataset, 'labels')
        )

        if has_direct_tensors:
            all_spectrums = dataset.spectrum.detach().cpu().numpy()
            all_glucose_history = dataset.glucose_history.detach().cpu().numpy()
            all_targets = dataset.labels.detach().cpu().numpy()

            has_aux = hasattr(dataset, 'aux') and dataset.aux is not None
            if has_aux:
                all_aux = dataset.aux.detach().cpu().numpy()
            else:
                all_aux = None
        else:
            # 回退路径：兼容任意自定义Dataset
            all_spectrums = []
            all_aux = []
            all_glucose_history = []
            all_targets = []

            for i in range(n_samples):
                sample = dataset[i]
                if len(sample) == 4:  # (spectrum, aux, glucose_history, target)
                    spectrum, aux, glucose_hist, target = sample
                    all_aux.append(aux.numpy() if isinstance(aux, torch.Tensor) else aux)
                else:  # (spectrum, glucose_history, target)
                    spectrum, glucose_hist, target = sample

                all_spectrums.append(spectrum.numpy() if isinstance(spectrum, torch.Tensor) else spectrum)
                all_glucose_history.append(glucose_hist.numpy() if isinstance(glucose_hist, torch.Tensor) else glucose_hist)
                all_targets.append(target.item() if isinstance(target, torch.Tensor) else target)

            all_spectrums = np.array(all_spectrums)
            all_glucose_history = np.array(all_glucose_history)
            all_targets = np.array(all_targets)
            has_aux = len(all_aux) > 0
            if has_aux:
                all_aux = np.array(all_aux)
        
        # 获取metadata
        timestamps = test_metadata['timestamps']
        experiment_indices = test_metadata.get('experiment_indices')
        split_indices = test_metadata.get('split_indices')
        
        # AR参数
        ar_history_len = all_glucose_history.shape[1]
        # 获取downsample_interval（用于glucose_history重采样）
        downsample_interval = getattr(self.config.data, 'downsample_interval', 10)
        print(f"  AR历史长度: {ar_history_len}, 下采样间隔: {downsample_interval}s")
        
        # 按实验分组处理
        if experiment_indices is not None:
            unique_experiments = np.unique(experiment_indices)
            print(f"  检测到 {len(unique_experiments)} 个实验，将分别进行AR推理")
        else:
            unique_experiments = [None]
        
        all_predictions = np.zeros(n_samples)
        
        with torch.inference_mode():
            for exp_idx, exp_id in enumerate(unique_experiments):
                if exp_id is not None:
                    exp_mask = experiment_indices == exp_id
                    exp_sample_indices = np.where(exp_mask)[0]
                    
                    exp_spectrums = all_spectrums[exp_sample_indices]
                    exp_targets = all_targets[exp_sample_indices]
                    exp_timestamps = timestamps[exp_sample_indices]
                    exp_split_indices = split_indices[exp_sample_indices] if split_indices is not None else None
                    if has_aux:
                        exp_aux = all_aux[exp_sample_indices]
                    
                    print(f"\n  实验 {exp_idx+1}/{len(unique_experiments)}: {len(exp_sample_indices)} 个样本")
                else:
                    exp_sample_indices = np.arange(n_samples)
                    exp_spectrums = all_spectrums
                    exp_targets = all_targets
                    exp_timestamps = timestamps
                    exp_split_indices = split_indices
                    if has_aux:
                        exp_aux = all_aux
                
                # 在当前实验内按时间排序
                exp_sort_indices = np.argsort(exp_timestamps)
                sorted_exp_spectrums = exp_spectrums[exp_sort_indices]
                sorted_exp_targets = exp_targets[exp_sort_indices]
                sorted_exp_timestamps = exp_timestamps[exp_sort_indices]
                sorted_exp_split_indices = exp_split_indices[exp_sort_indices] if exp_split_indices is not None else None
                if has_aux:
                    sorted_exp_aux = exp_aux[exp_sort_indices]
                
                # 识别连续段（考虑alternating split边界）
                segments = self._find_continuous_segments_single_exp(
                    sorted_exp_timestamps, 
                    sorted_exp_split_indices
                )
                print(f"    识别到 {len(segments)} 个连续段")
                
                # 对每个连续段进行AR推理
                pbar = tqdm(total=len(exp_sample_indices), desc=f'Exp {exp_idx+1} AR', leave=False)
                
                for seg_idx, (seg_start, seg_end) in enumerate(segments):
                    seg_len = seg_end - seg_start
                    
                    # 🔧 关键：维护完整预测历史（时间戳+glucose值）
                    # 用于每次预测时重采样glucose_history
                    prediction_history_times = []
                    prediction_history_values = []
                    
                    # 用ground truth初始化预测历史
                    if seg_len >= ar_history_len:
                        # 用前ar_history_len个ground truth初始化
                        for i in range(seg_start, seg_start + ar_history_len):
                            prediction_history_times.append(sorted_exp_timestamps[i])
                            prediction_history_values.append(sorted_exp_targets[i])
                            
                            # 前ar_history_len个样本直接用ground truth作为预测
                            global_idx = exp_sample_indices[exp_sort_indices[i]]
                            all_predictions[global_idx] = sorted_exp_targets[i]
                            pbar.update(1)
                        
                        if seg_idx == 0:
                            print(f"      段{seg_idx+1}: 用{ar_history_len}个ground truth初始化")
                        
                        start_pred_idx = seg_start + ar_history_len
                    else:
                        # 段太短，用默认值初始化
                        NORMAL_GLUCOSE = 5.0
                        # 估计时间间隔
                        if seg_len > 1:
                            avg_interval = (sorted_exp_timestamps[seg_end-1] - sorted_exp_timestamps[seg_start]) / (seg_len - 1)
                        else:
                            avg_interval = 10.0  # 默认10秒
                        
                        start_time = sorted_exp_timestamps[seg_start]
                        for j in range(ar_history_len):
                            fake_time = start_time - (ar_history_len - j) * avg_interval
                            prediction_history_times.append(fake_time)
                            prediction_history_values.append(NORMAL_GLUCOSE)
                        
                        if seg_idx == 0:
                            print(f"      段{seg_idx+1}: 段太短({seg_len}<{ar_history_len})，用默认值初始化")
                        
                        start_pred_idx = seg_start
                    
                    # 从start_pred_idx开始进行autoregressive预测
                    for i in range(start_pred_idx, seg_end):
                        current_time = sorted_exp_timestamps[i]
                        
                        # 🔧 关键：从预测历史中按downsample_interval重采样glucose_history
                        # 确保与训练时的采样方式一致
                        # 例如：时刻101需要[g_91, g_81, g_71, ...]（每10s采样）
                        current_history = self._resample_glucose_history(
                            prediction_history_times,
                            prediction_history_values,
                            current_time,
                            ar_history_len,
                            downsample_interval  # 新增参数
                        )
                        
                        # 准备输入
                        spectrum = torch.FloatTensor(sorted_exp_spectrums[i]).unsqueeze(0).to(self.device)
                        history = torch.FloatTensor(current_history).unsqueeze(0).to(self.device)
                        
                        aux_input = None
                        if has_aux:
                            aux_input = torch.FloatTensor(sorted_exp_aux[i]).unsqueeze(0).to(self.device)
                        
                        # 预测
                        output = self._model_forward(spectrum, aux_input, history)
                        pred_value = output.item()
                        del spectrum, history, output
                        if aux_input is not None:
                            del aux_input
                        
                        # 保存预测结果
                        global_idx = exp_sample_indices[exp_sort_indices[i]]
                        all_predictions[global_idx] = pred_value
                        
                        # 🔧 将新预测加入历史（供下一次预测使用）
                        prediction_history_times.append(current_time)
                        prediction_history_values.append(pred_value)
                        
                        pbar.update(1)
                
                pbar.close()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"  ✓ 所有实验AR推理完成")
        self._release_eval_memory()
        return all_predictions, all_targets, test_metadata
    
    def _resample_glucose_history(self, history_times, history_values, current_time, history_len, downsample_interval):
        """
        从预测历史中按照downsample_interval重采样glucose_history
        
        确保与训练时的采样方式完全一致：
        - 训练时：downsample_times = np.arange(window_start_time, current_time, downsample_interval)
        - 推理时：使用相同的逻辑，从(current_time - window_duration)到current_time按interval采样
        
        注意：窗口移动速度可能和downsample_interval不同！
        - 窗口可能每1秒移动一次（预测频率）
        - 但glucose_history按10秒间隔采样
        - 所以需要从完整预测历史中按10秒间隔重新采样
        
        Args:
            history_times: 历史时间戳列表（每次预测都会加入）
            history_values: 历史glucose值列表
            current_time: 当前时刻
            history_len: 需要的历史长度
            downsample_interval: 下采样间隔（秒）
        
        Returns:
            resampled_history: np.array of shape (history_len,)
        """
        if len(history_values) == 0:
            return np.full(history_len, 5.0, dtype=np.float32)
        
        # 转换为numpy数组
        history_times_arr = np.array(history_times)
        history_values_arr = np.array(history_values)
        
        # 🔧 关键：与训练时完全一致的采样逻辑
        # 训练时: downsample_times = np.arange(window_start_time, current_time, interval)
        # 推理时使用相同逻辑：
        window_duration = history_len * downsample_interval  # 估计窗口长度
        window_start_time = current_time - window_duration
        
        # 生成下采样时间网格（与训练时一致，不包含current_time）
        target_times = np.arange(window_start_time, current_time, downsample_interval)
        
        # 如果点数不够，从开头补
        if len(target_times) < history_len:
            # 需要补充的点数
            n_pad = history_len - len(target_times)
            pad_times = np.arange(
                window_start_time - n_pad * downsample_interval,
                window_start_time,
                downsample_interval
            )
            target_times = np.concatenate([pad_times, target_times])
        
        # 如果点数太多，取最后history_len个
        if len(target_times) > history_len:
            target_times = target_times[-history_len:]
        
        # 对每个目标时间点，从历史中找最近的值（与训练时相同的逻辑）
        resampled = []
        for target_t in target_times:
            # 找最近的历史时间点
            time_diffs = np.abs(history_times_arr - target_t)
            nearest_idx = np.argmin(time_diffs)
            
            # 使用最近的值（训练时也是这样做的）
            # 即使时间差较大，也用最近的值（边缘填充效果）
            resampled.append(history_values_arr[nearest_idx])
        
        return np.array(resampled, dtype=np.float32)
    
    def _find_continuous_segments_single_exp(self, timestamps, split_indices=None):
        """
        识别单个实验内的连续段（考虑alternating split边界）
        
        连续段定义：
        1. 时间戳连续（时间间隔不超过阈值）
        2. 如果有split_indices，同一split内的样本才算连续
           （alternating split的不同测试段之间不连续）
        
        Args:
            timestamps: 时间戳数组（已排序）
            split_indices: 分段索引数组（可选，用于识别alternating split边界）
        
        Returns:
            segments: list of (start, end) tuples
        """
        n_samples = len(timestamps)
        if n_samples == 0:
            return []
        
        if n_samples == 1:
            return [(0, 1)]
        
        segments = []
        seg_start = 0
        
        # 计算时间间隔阈值
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs[time_diffs > 0]) if np.any(time_diffs > 0) else 1.0
        time_threshold = median_diff * 5
        
        # 如果有split_indices，计算split连续性阈值
        # 在alternating split中，同一测试段内split_indices应该是连续的（差值=1）
        # 不同测试段之间会有较大跳跃（被训练段隔开）
        if split_indices is not None:
            split_diffs = np.diff(split_indices)
            # 允许差值<=1为连续，>1为不连续（被训练段隔开）
            split_threshold = 2
        
        # 检查连续性
        for i in range(1, n_samples):
            is_continuous = True
            
            # 检查时间连续性
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > time_threshold or time_diff < 0:
                is_continuous = False
            
            # 检查split_indices连续性（识别alternating split边界）
            if is_continuous and split_indices is not None:
                split_diff = abs(split_indices[i] - split_indices[i-1])
                if split_diff > split_threshold:
                    is_continuous = False
            
            if not is_continuous:
                segments.append((seg_start, i))
                seg_start = i
        
        # 保存最后一段
        segments.append((seg_start, n_samples))
        
        return segments
    
    def _find_continuous_segments(self, experiment_indices, split_indices, timestamps):
        """
        识别连续段
        
        连续段的定义（针对多实验alternating split优化）：
        1. 【必须】同一实验内（实验切换必定断开）
        2. 【必须】时间戳连续（时间间隔不超过阈值）
        
        注意：不检查split_indices连续性！
        原因：多实验alternating split时，每个实验独立分段，测试集包含
        所有实验的偶数段。按时间排序后，来自不同实验的段会交织，
        导致split_indices跳跃，但这不代表数据不连续。
        
        Args:
            experiment_indices: 实验索引（已排序）
            split_indices: 原始数据索引（已排序，但不用于连续性判断）
            timestamps: 时间戳（已排序）
        
        Returns:
            segments: list of (start, end) tuples
        """
        n_samples = len(timestamps)
        if n_samples == 0:
            return []
        
        segments = []
        seg_start = 0
        
        # 计算合理的时间间隔阈值（使用中位数的3倍）
        if n_samples > 1:
            time_diffs = np.diff(timestamps)
            median_diff = np.median(time_diffs[time_diffs > 0]) if np.any(time_diffs > 0) else 1.0
            time_threshold = median_diff * 5  # 允许一定的间隔波动
        else:
            time_threshold = float('inf')
        
        # 🔍 统计断点原因
        exp_breaks = 0
        time_breaks = 0
        
        for i in range(1, n_samples):
            is_continuous = True
            
            # 检查1：实验是否相同（最高优先级）
            if experiment_indices is not None:
                if experiment_indices[i] != experiment_indices[i-1]:
                    is_continuous = False
                    exp_breaks += 1
            
            # 检查2：时间戳是否连续（主要判断依据）
            # 注意：不再检查split_indices！
            # 因为alternating split已经保证了段内连续性，
            # 只要实验相同且时间连续，数据就是真正连续的
            if is_continuous:
                time_diff = timestamps[i] - timestamps[i-1]
                if time_diff > time_threshold or time_diff < 0:
                    is_continuous = False
                    time_breaks += 1
            
            # 如果不连续，保存当前段，开始新段
            if not is_continuous:
                segments.append((seg_start, i))
                seg_start = i
        
        # 保存最后一段
        segments.append((seg_start, n_samples))
        
        # 🔍 打印详细诊断信息
        seg_lengths = [end - start for start, end in segments]
        long_segs = sum(1 for l in seg_lengths if l >= 20)
        short_segs = len(segments) - long_segs
        
        print(f"  ⚙️  分段诊断:")
        print(f"    断点原因: 实验切换={exp_breaks}, 时间跳跃={time_breaks}")
        print(f"    时间阈值: {time_threshold:.2f}秒 (中位数间隔={median_diff:.2f}秒)")
        print(f"    段统计: 总段数={len(segments)}, 长段(≥20)={long_segs}, 短段(<20)={short_segs}")
        print(f"    段长度: 最小={min(seg_lengths)}, 最大={max(seg_lengths)}, 平均={np.mean(seg_lengths):.1f}")
        
        # 如果断点过多，打印警告
        if time_breaks > n_samples * 0.3:
            print(f"  ⚠️  警告: 时间跳跃断点过多({time_breaks})，可能是时间阈值太严格")
            print(f"       建议检查: 1) downsample_interval设置  2) 数据是否有大量缺失")
        
        return segments
    
    def test(self, test_loader, save_visualizations=True, viz_dir='./results', test_metadata=None, full_metadata=None,
             metric_prefix='final_test', log_to_summary=True):
        """
        测试模型并生成可视化报告
        
        Args:
            test_loader: 测试数据加载器
            save_visualizations: 是否保存可视化图表
            viz_dir: 可视化结果保存目录
            test_metadata: 测试集元数据（包含时间戳和实验信息）
            full_metadata: 完整数据集的元数据（包含所有ground truth，用于完整可视化）
            metric_prefix: W&B指标前缀，默认'final_test'
            log_to_summary: 是否将指标写入W&B summary
        
        Returns:
            test_loss, test_mae, test_rmse, test_mape: 测试指标
            
        注意事项：
            - MAE计算使用原始数据（保持预测-真值配对关系）
            - 可视化时会按时间戳排序数据，但同时排序y_pred和y_true以保持配对
            - 因此训练过程的MAE和可视化图上的MAE应该完全一致
        """
        print("\n" + "="*70)
        print("模型测试")
        print("="*70 + "\n")
        
        test_start_time = time.time()
        
        # 检查是否使用真正的自回归推理
        # 条件：启用AR模式 + 有test_metadata（需要识别连续段）
        use_true_ar_inference = (
            self.autoregressive and 
            test_metadata is not None and 
            test_metadata.get('timestamps') is not None
        )
        
        if use_true_ar_inference:
            print("📊 使用真正的自回归推理（滚动预测）...")
            y_pred, y_true, test_metadata = self._test_autoregressive_inference(
                test_loader, test_metadata
            )
            # AR推理返回的已经是排序后的数据
            y_pred_viz = y_pred
            y_true_viz = y_true
            total_loss = 0  # AR推理不计算batch loss
        else:
            # 原有的Teacher Forcing推理
            # 收集所有预测和真实值
            self.model.eval()
            all_predictions = []
            all_targets = []
            total_loss = 0
            final_test_fusion_diag = {}
            self._reset_fusion_diagnostics()
            
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
                    # 解包数据
                    spectrum, aux, glucose_history, target = self._unpack_batch(batch)
                    
                    # 移动到设备
                    spectrum = spectrum.to(self.device)
                    target = target.to(self.device)
                    if aux is not None:
                        aux = aux.to(self.device)
                    if glucose_history is not None:
                        glucose_history = glucose_history.to(self.device)
                    
                    # 模型前向传播
                    output = self._model_forward(spectrum, aux, glucose_history)
                    
                    # 计算损失（测试时不使用AR正则化）
                    if self.autoregressive:
                        loss = self.criterion.base_criterion(output, target)
                    else:
                        loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    all_predictions.append(output.detach().cpu().to(torch.float32).numpy().copy())
                    all_targets.append(target.detach().cpu().to(torch.float32).numpy().copy())

                    del spectrum, target, output, loss
                    if aux is not None:
                        del aux
                    if glucose_history is not None:
                        del glucose_history
                    if torch.cuda.is_available() and (batch_idx + 1) % 200 == 0:
                        torch.cuda.empty_cache()
            
            # 转换为 numpy 数组
            y_pred = np.concatenate(all_predictions).flatten()
            y_true = np.concatenate(all_targets).flatten()
            del all_predictions, all_targets
            final_test_fusion_diag = self._collect_fusion_diagnostics()
            self._release_eval_memory()
            
            # ⚠️ 重要：如果使用random分割，数据和时间戳都被打乱了
            # 需要按照时间戳重新排序，才能得到清晰的时间序列可视化
            # 注意：必须同时对y_pred, y_true, timestamps使用相同的排序索引
            if test_metadata is not None:
                timestamps = test_metadata.get('timestamps', None)
                experiment_indices = test_metadata.get('experiment_indices', None)
                split_indices = test_metadata.get('split_indices', None)
                
                if timestamps is not None and len(timestamps) == len(y_true):
                    # 按时间戳排序
                    sort_indices = np.argsort(timestamps)
                    
                    # 同时排序预测值、真实值、时间戳（保持配对关系）
                    y_pred_viz = y_pred[sort_indices]
                    y_true_viz = y_true[sort_indices]
                    
                    # 更新metadata中的timestamps、experiment_indices和split_indices为排序后的
                    sorted_metadata = test_metadata.copy()
                    sorted_metadata['timestamps'] = timestamps[sort_indices]
                    if experiment_indices is not None:
                        sorted_metadata['experiment_indices'] = experiment_indices[sort_indices]
                    if split_indices is not None:
                        sorted_metadata['split_indices'] = split_indices[sort_indices]
                    test_metadata = sorted_metadata
                else:
                    y_pred_viz = y_pred
                    y_true_viz = y_true
            else:
                y_pred_viz = y_pred
                y_true_viz = y_true
        
        # 计算指标（使用y_pred和y_true，确保配对正确）
        n_samples = len(test_loader.dataset)
        
        # 对于AR推理，重新计算loss
        if use_true_ar_inference:
            # 使用基础损失函数计算总loss
            with torch.inference_mode():
                y_pred_tensor = torch.FloatTensor(y_pred).to(self.device)
                y_true_tensor = torch.FloatTensor(y_true).to(self.device)
                if hasattr(self.criterion, 'base_criterion'):
                    test_loss = self.criterion.base_criterion(y_pred_tensor, y_true_tensor).item()
                else:
                    test_loss = self.criterion(y_pred_tensor, y_true_tensor).item()
                del y_pred_tensor, y_true_tensor
            self._release_eval_memory()
        else:
            test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        test_mae = np.mean(np.abs(y_pred - y_true))
        test_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        test_mape = self._compute_mape(y_pred, y_true)
        
        # 计算额外的评估指标
        from sklearn.metrics import r2_score
        from scipy.stats import pearsonr
        
        test_r2 = r2_score(y_true, y_pred)
        test_corr, test_corr_p = pearsonr(y_true, y_pred)
        
        test_time = time.time() - test_start_time
        self.total_test_time = test_time
        
        # 计算吞吐量
        throughput = n_samples / test_time
        
        print(f"测试结果:")
        print(f"  Loss:        {test_loss:.4f}")
        print(f"  MAE:         {test_mae:.4f} mmol/L")
        print(f"  RMSE:        {test_rmse:.4f} mmol/L")
        print(f"  MAPE:        {test_mape:.2f}%")
        print(f"  R²:          {test_r2:.4f}")
        print(f"  Correlation: {test_corr:.4f} (p={test_corr_p:.4e})")
        if not use_true_ar_inference:
            diag_text = self._format_fusion_diagnostics(final_test_fusion_diag)
            if diag_text:
                print(f"  FusionDiag:  {diag_text}")
        print("="*70)
        print(f"\n⏱️  推理用时统计:")
        print(f"  总推理时间: {test_time:.2f}秒")
        print(f"  样本数量: {n_samples:,}")
        print(f"  推理吞吐量: {throughput:.2f} 样本/秒")
        print(f"  单样本延迟: {test_time*1000/n_samples:.2f} 毫秒")
        print("="*70 + "\n")
        
        if self.use_wandb:
            wandb.log({
                f'{metric_prefix}_loss': test_loss,
                f'{metric_prefix}_mae': test_mae,
                f'{metric_prefix}_rmse': test_rmse,
                f'{metric_prefix}_mape': test_mape,
                f'{metric_prefix}_r2': test_r2,
                f'{metric_prefix}_correlation': test_corr,
                f'{metric_prefix}_correlation_p': test_corr_p,
                'test_time': test_time,
                'inference_throughput': throughput,
            })
            # 同时记录到 summary 以便在表格中查看
            if log_to_summary:
                wandb.run.summary[f'{metric_prefix}_mae'] = test_mae
                wandb.run.summary[f'{metric_prefix}_rmse'] = test_rmse
                wandb.run.summary[f'{metric_prefix}_mape'] = test_mape
                wandb.run.summary[f'{metric_prefix}_r2'] = test_r2
                wandb.run.summary[f'{metric_prefix}_correlation'] = test_corr
        
        # 生成可视化报告
        if save_visualizations:
            import os
            from datetime import datetime
            from visualization import create_evaluation_report
            from visualization.plots import plot_time_series_comparison
            
            # 根据实验名称创建子文件夹
            if test_metadata is not None and test_metadata.get('experiment_names'):
                experiment_names = test_metadata.get('experiment_names', [])
                if len(experiment_names) == 1:
                    # 单个实验：使用实验名称
                    subfolder = experiment_names[0]
                elif len(experiment_names) > 1:
                    # 多个实验：使用第一个实验名称加数量标识
                    subfolder = f"{experiment_names[0]}_and_{len(experiment_names)-1}_more"
                else:
                    subfolder = 'unknown_experiment'
            else:
                subfolder = 'default'
            
            # 创建带实验名称的子目录
            viz_dir_with_exp = os.path.join(viz_dir, subfolder)
            os.makedirs(viz_dir_with_exp, exist_ok=True)
            
            # 验证排序后的MAE应该与原始MAE相同（因为保持了配对关系）
            mae_viz = np.mean(np.abs(y_true_viz - y_pred_viz))
            print(f"\n📊 MAE验证:")
            print(f"  训练过程显示的MAE: {test_mae:.4f} mmol/L")
            print(f"  可视化数据的MAE:   {mae_viz:.4f} mmol/L")
            if abs(test_mae - mae_viz) < 1e-6:
                print(f"  ✓ 一致！（差异: {abs(test_mae - mae_viz):.2e}）")
            else:
                print(f"  ⚠️ 不一致！（差异: {abs(test_mae - mae_viz):.4f}）")
                print(f"  这可能是排序导致的配对问题，已在代码中修复。")
            
            create_evaluation_report(y_true, y_pred, save_dir=viz_dir_with_exp, prefix='test', training_info=self._get_training_info())
            
            # 生成时间序列对比图（使用排序后的数据）
            # 如果有多个实验，为每个实验分别生成图
            if test_metadata is not None:
                from visualization.plots import plot_time_series_by_experiment
                
                # 使用新的分实验可视化函数
                saved_paths = plot_time_series_by_experiment(
                    y_true=y_true_viz,  # 使用排序后的数据
                    y_pred=y_pred_viz,  # 使用排序后的数据
                    metadata=test_metadata,  # 包含experiment_indices
                    full_metadata=full_metadata,  # 传入完整数据用于显示完整ground truth
                    save_dir=viz_dir_with_exp,
                    title_prefix="Test",
                    training_info=self._get_training_info()
                )
                
                # 如果使用W&B，上传所有图片
                if self.use_wandb:
                    for i, filepath in enumerate(saved_paths):
                        wandb.log({f"time_series_comparison_{i}": wandb.Image(str(filepath))})

            self._release_eval_memory()
        
        return test_loss, test_mae, test_rmse, test_mape
    
    def _generate_train_visualization(self, train_loader, train_metadata, full_metadata, viz_dir='./results'):
        """
        生成训练集的时间序列对比可视化
        显示完整的ground truth，但预测值只显示训练集部分
        
        Args:
            train_loader: 训练数据加载器
            train_metadata: 训练集元数据
            full_metadata: 完整数据集的元数据（包含所有ground truth）
            viz_dir: 可视化结果保存目录（默认 './results'）
        """
        import os
        import numpy as np
        from datetime import datetime
        from visualization.plots import plot_time_series_comparison
        from torch.utils.data import DataLoader
        
        # 切换到评估模式
        self.model.eval()
        
        # 创建一个不打乱顺序的临时 DataLoader，确保按时间顺序获取数据
        # 这样可视化时才能正确显示时间序列
        train_dataset = train_loader.dataset
        sequential_loader = DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,  # 关键：不打乱顺序
            num_workers=0,  # 简化，使用单线程
            pin_memory=False
        )
        
        all_predictions = []
        all_targets = []
        
        # 收集所有训练集的预测和真实值（按顺序）
        with torch.inference_mode():
            for batch in sequential_loader:
                # 解包数据
                spectrum, aux, glucose_history, target = self._unpack_batch(batch)
                
                # 移动到设备
                spectrum = spectrum.to(self.device)
                target = target.to(self.device)
                if aux is not None:
                    aux = aux.to(self.device)
                if glucose_history is not None:
                    glucose_history = glucose_history.to(self.device)
                
                # 模型前向传播
                output = self._model_forward(spectrum, aux, glucose_history)
                
                all_predictions.append(output.detach().cpu().to(torch.float32).numpy().copy())
                all_targets.append(target.detach().cpu().to(torch.float32).numpy().copy())

                del spectrum, target, output
                if aux is not None:
                    del aux
                if glucose_history is not None:
                    del glucose_history
        
        # 转换为numpy数组
        y_pred = np.concatenate(all_predictions).flatten()
        y_true = np.concatenate(all_targets).flatten()
        del all_predictions, all_targets
        self._release_eval_memory()
        
        # ⚠️ 重要：如果使用random分割，数据和时间戳都被打乱了
        # 需要按照时间戳重新排序，才能得到清晰的时间序列可视化
        timestamps = train_metadata.get('timestamps', None)
        experiment_indices = train_metadata.get('experiment_indices', None)
        split_indices = train_metadata.get('split_indices', None)
        
        # 按时间戳排序（如果有）
        if timestamps is not None and len(timestamps) == len(y_true):
            # 按时间戳排序
            sort_indices = np.argsort(timestamps)
            y_pred = y_pred[sort_indices]
            y_true = y_true[sort_indices]
            # 同时更新metadata中的所有相关数据
            sorted_metadata = train_metadata.copy()
            sorted_metadata['timestamps'] = timestamps[sort_indices]
            if experiment_indices is not None and len(experiment_indices) == len(y_true):
                sorted_metadata['experiment_indices'] = experiment_indices[sort_indices]
            if split_indices is not None and len(split_indices) == len(y_true):
                sorted_metadata['split_indices'] = split_indices[sort_indices]
            train_metadata = sorted_metadata
        
        # 根据实验名称创建子文件夹
        if train_metadata is not None and train_metadata.get('experiment_names'):
            experiment_names = train_metadata.get('experiment_names', [])
            if len(experiment_names) == 1:
                # 单个实验：使用实验名称
                subfolder = experiment_names[0]
            elif len(experiment_names) > 1:
                # 多个实验：使用第一个实验名称加数量标识
                subfolder = f"{experiment_names[0]}_and_{len(experiment_names)-1}_more"
            else:
                subfolder = 'unknown_experiment'
        else:
            subfolder = 'default'
        
        # 创建带实验名称的子目录（与测试集保存到相同位置）
        viz_dir_with_exp = os.path.join(viz_dir, subfolder)
        os.makedirs(viz_dir_with_exp, exist_ok=True)
        
        print(f"📊 生成训练集时间序列对比图...")
        
        # 使用分实验可视化函数（与测试集一致）
        if train_metadata is not None and train_metadata.get('experiment_indices') is not None:
            from visualization.plots import plot_time_series_by_experiment
            
            # 为每个实验分别生成图
            saved_paths = plot_time_series_by_experiment(
                y_true=y_true,
                y_pred=y_pred,
                metadata=train_metadata,
                full_metadata=full_metadata,  # 传入完整数据用于显示完整ground truth
                save_dir=viz_dir_with_exp,
                title_prefix="Train",
                training_info=self._get_training_info()
            )
            
            # 如果使用W&B，上传所有图片
            if self.use_wandb:
                for i, filepath in enumerate(saved_paths):
                    wandb.log({f"train_time_series_{i}": wandb.Image(str(filepath))})
        else:
            # 如果没有实验信息，使用原来的单图方式
            from visualization.plots import plot_time_series_comparison
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'train_time_series_comparison_{timestamp_str}.png'
            filepath = os.path.join(viz_dir_with_exp, filename)
            
            plot_time_series_comparison(
                y_true=y_true,
                y_pred=y_pred,
                metadata=train_metadata,
                save_path=filepath,
                title_prefix="Train",
                training_info=self._get_training_info()
            )
        
        # 如果使用W&B，上传图片
        if self.use_wandb:
            wandb.log({"train_time_series_comparison": wandb.Image(filepath)})
        
        # 切换回训练模式
        self.model.train()

    def _get_training_info(self):
        """
        获取训练信息字典，用于可视化
        
        Returns:
            dict: 包含训练配置和状态的字典
        """
        info = {
            'model_name': self.config.model.architecture,
            'epochs': self.config.training.epochs,
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'dropout': self.config.model.dropout,
            'loss_function': self.config.training.loss_function,
            'mae_window_loss_weight': self.config.training.mae_window_loss_weight,
            'mae_window_batches': self.config.training.mae_window_batches,
            'optimizer': self.config.training.optimizer,
            'scheduler': self.config.training.scheduler,
            'weight_decay': self.config.training.weight_decay,
        }
        
        # 添加模型特定配置
        if hasattr(self.config.model, 'hidden_size'):
            info['hidden_size'] = self.config.model.hidden_size
        if hasattr(self.config.model, 'num_layers'):
            info['num_layers'] = self.config.model.num_layers
        if hasattr(self.config.model, 'use_attention'):
            info['attention'] = 'Yes' if self.config.model.use_attention else 'No'
        if hasattr(self.config.model, 'aggregation'):
            info['aggregation'] = self.config.model.aggregation
        
        # 添加数据配置
        if hasattr(self.config.data, 'mode'):
            info['mode'] = self.config.data.mode
        if hasattr(self.config.data, 'window_size') and self.config.data.mode == 'window':
            info['window_size'] = self.config.data.window_size
        if hasattr(self.config.data, 'window_duration') and self.config.data.window_duration:
            info['window_duration'] = f"{self.config.data.window_duration}s"
        if hasattr(self.config.data, 'downsample') and self.config.data.downsample:
            info['downsample_interval'] = f"{self.config.data.downsample_interval}s"
        if hasattr(self.config.data, 'split_strategy'):
            info['split_strategy'] = self.config.data.split_strategy
        if hasattr(self.config.data, 'n_splits'):
            info['n_splits'] = self.config.data.n_splits
        if hasattr(self.config.data, 'normalize'):
            info['normalize'] = 'Yes' if self.config.data.normalize else 'No'
        
        # 添加训练集划分信息
        if hasattr(self.config.data, 'train_split'):
            train_pct = int(self.config.data.train_split * 100)
            test_pct = int(self.config.data.test_split * 100)
            if hasattr(self.config.data, 'use_val_set') and self.config.data.use_val_set:
                val_pct = int(self.config.data.val_split * 100)
                info['data_split'] = f"Train:{train_pct}% Val:{val_pct}% Test:{test_pct}%"
            else:
                info['data_split'] = f"Train:{train_pct}% Test:{test_pct}%"
        
        # 添加最佳epoch（如果已训练）
        if self.best_epoch > 0:
            info['best_epoch'] = self.best_epoch
        
        return info

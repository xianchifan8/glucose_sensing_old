#!/usr/bin/env python3
"""
批量实验运行脚本
读取命令列表文件，按顺序执行每个实验，结果保存到独立文件夹
"""

import subprocess
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import re
import json


def extract_test_metrics(log_file: Path) -> dict:
    """从日志文件中提取测试指标"""
    metrics = {}
    
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 提取测试指标（根据 trainer.py 的输出格式）
        # 示例格式: "Test RMSE: 12.34"
        patterns = {
            'RMSE': r'Test RMSE:\s*([\d.]+)',
            'MAE': r'Test MAE:\s*([\d.]+)',
            'R2': r'Test R²:\s*([\d.]+)',
            'MAPE': r'Test MAPE:\s*([\d.]+)',
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, log_content)
            if match:
                metrics[metric_name] = float(match.group(1))
        
        # 尝试提取总体统计信息
        stats_pattern = r'Mean.*RMSE:\s*([\d.]+).*MAE:\s*([\d.]+).*R²:\s*([\d.]+)'
        stats_match = re.search(stats_pattern, log_content, re.DOTALL)
        if stats_match and not metrics:  # 如果前面没找到，用这个
            metrics['Mean_RMSE'] = float(stats_match.group(1))
            metrics['Mean_MAE'] = float(stats_match.group(2))
            metrics['Mean_R2'] = float(stats_match.group(3))
        
    except Exception as e:
        print(f"  警告: 提取指标时出错: {e}")
    
    return metrics


def parse_experiment_name(command: str) -> str:
    """从命令中提取实验名称（自动生成）- 增强版，提取更多关键参数"""
    parts = []
    
    # 提取关键参数
    tokens = command.split()
    i = 0
    
    # 用于存储各个参数
    data_source = 'bin'  # 默认是bin格式
    model = None
    mode_str = None
    lr = None
    epochs = None
    dropout = None
    use_attention = None
    normalize = False
    data_fusion = False
    fusion_method = None
    fusion_method_late = None
    fusion_stage = None
    icm_mode = None
    aux_override = False
    aux_override_value = 0.0
    split_strategy = None
    n_splits = None
    experiments = None
    
    while i < len(tokens):
        token = tokens[i]
        
        # 提取数据源类型
        if token == '--data_source' and i + 1 < len(tokens):
            data_source = tokens[i + 1]
        # 提取模型
        elif token == '--model' and i + 1 < len(tokens):
            model = tokens[i + 1]
        # 提取模式和窗口大小/时长
        elif token == '--mode' and i + 1 < len(tokens):
            mode = tokens[i + 1]
            if mode == 'window':
                # 查找 window_duration 或 window_size
                window_value = None
                for j in range(i, min(i + 15, len(tokens))):
                    if tokens[j] == '--window_duration' and j + 1 < len(tokens):
                        window_value = f"wd{tokens[j + 1]}"
                        break
                    elif tokens[j] == '--window_size' and j + 1 < len(tokens):
                        window_value = f"ws{tokens[j + 1]}"
                        break
                mode_str = f"win_{window_value}" if window_value else "win"
            else:
                mode_str = "inst"  # instant缩写
        # 提取学习率
        elif token == '--lr' and i + 1 < len(tokens):
            lr = tokens[i + 1]
        # 提取epochs
        elif token == '--epochs' and i + 1 < len(tokens):
            epochs = tokens[i + 1]
        # 提取dropout
        elif token == '--dropout' and i + 1 < len(tokens):
            dropout = tokens[i + 1]
        # 检查是否使用 attention
        elif token == '--use_attention':
            use_attention = True
        elif token == '--no_attention':
            use_attention = False
        # 检查是否 normalize
        elif token == '--normalize':
            normalize = True
        # 检查是否 data_fusion
        elif token == '--data_fusion':
            data_fusion = True
        # 提取fusion_stage
        elif token == '--fusion_stage' and i + 1 < len(tokens):
            fusion_stage = tokens[i + 1]
        # 提取fusion_method (early)
        elif token == '--fusion_method' and i + 1 < len(tokens):
            fusion_method = tokens[i + 1]
        # 提取fusion_method_late
        elif token == '--fusion_method_late' and i + 1 < len(tokens):
            fusion_method_late = tokens[i + 1]
        # 提取icm_mode
        elif token == '--icm_mode' and i + 1 < len(tokens):
            icm_mode = tokens[i + 1]
        # 检查aux_override
        elif token == '--aux_override':
            aux_override = True
        # 提取aux_override_value
        elif token == '--aux_override_value' and i + 1 < len(tokens):
            aux_override_value = float(tokens[i + 1])
        # 提取split策略
        elif token == '--split_strategy' and i + 1 < len(tokens):
            split_strategy = tokens[i + 1]
        # 提取n_splits（用于alternating策略）
        elif token == '--n_splits' and i + 1 < len(tokens):
            n_splits = tokens[i + 1]
        # 提取实验数据集
        elif token == '--experiments' and i + 1 < len(tokens):
            if tokens[i + 1] == 'all':
                experiments = 'all'
            else:
                # 提取所有实验名（去掉_Tao后缀）
                exp_list = []
                j = i + 1
                while j < len(tokens) and not tokens[j].startswith('--'):
                    exp_list.append(tokens[j].replace('_Tao', ''))
                    j += 1
                
                # 如果只有一个实验，直接使用
                if len(exp_list) == 1:
                    experiments = exp_list[0]
                # 如果有多个，使用数量标识
                elif len(exp_list) > 1:
                    experiments = f"{len(exp_list)}exps"
        
        i += 1
    
    # 构建名称（按重要性排序）
    # 0. 数据源标识（db 或 bin）
    parts.append(f"[{data_source.upper()}]")
    
    # 1. 模型
    if model:
        parts.append(model)
    
    # 2. 数据集（实验名称）
    if experiments:
        parts.append(f"exp{experiments}")
    
    # 3. 模式和窗口（重要！）
    if mode_str:
        parts.append(mode_str)
    
    # 4. data_fusion 状态和方法（重要！）
    if data_fusion:
        # 根据fusion_stage选择对应的fusion_method
        if fusion_stage == 'late':
            selected_fusion_method = fusion_method_late
        else:  # early 或 None (默认early)
            selected_fusion_method = fusion_method
        
        if selected_fusion_method:
            # 使用融合方法的缩写
            fusion_abbr = {
                'concat': 'con', 
                'film': 'film',
                'film_sensor_aware': 'filmsa',

                'attention_pool': 'ap',
                'residual_film': 'rfilm',
                'conditioned_rf': 'crf',
                'attention_fusion': 'attnfus',
                'cross_attention_film': 'cafilm',
                'aux_cross_only': 'auxonly',
                'film_aux_only': 'faux',
                'gated_residual': 'gres',
                'gated_residual_film': 'gresfilm',
                'ppg_icm_gated_residual_film': 'pigrfilm',
                'ppg_icm_gated_residual_only': 'pigronly',
                'ppg_icm_engineered_only': 'piengonly',
                'ppg_icm_segment_raw_only': 'pisegraw',
                'spectrum_only_late_head': 'solate',
                'attention': 'attn'
            }.get(selected_fusion_method, selected_fusion_method[:4])
            
            # 添加融合阶段标识
            stage_abbr = 'e' if fusion_stage == 'early' or fusion_stage is None else 'l'
            parts.append(f"fus{stage_abbr}{fusion_abbr}")
        else:
            parts.append("fus")
    
    # ICM mode (if not raw)
    if icm_mode and icm_mode != 'raw':
        icm_abbr = {
            'processed': 'proc',
            'both': 'both'
        }.get(icm_mode, icm_mode)
        parts.append(f"icm{icm_abbr}")
    
    # Aux Override (消融实验)
    if aux_override:
        # 格式化值：整数直接显示，小数保留1位
        if aux_override_value == int(aux_override_value):
            parts.append(f"auxov{int(aux_override_value)}")
        else:
            parts.append(f"auxov{aux_override_value:.1f}")
    
    # 5. attention 状态
    if use_attention is True:
        parts.append("attn")
    elif use_attention is False:
        parts.append("noattn")
    
    # 6. normalize 状态
    if normalize:
        parts.append("norm")
    
    # 学习率
    if lr:
        parts.append(f"lr{lr}")
    
    # epochs
    if epochs:
        parts.append(f"ep{epochs}")
    
    # dropout
    if dropout:
        parts.append(f"dr{dropout}")
    
    # split策略（缩写）
    if split_strategy:
        split_abbr = {'random': 'rnd', 'temporal': 'tmp', 'stratified': 'str', 
                      'experiment': 'exp', 'alternating': 'alt',
                      'hybrid_alternating': 'halt', 'cross_user': 'xusr',
                      'multi_user_independent': 'mui'}.get(split_strategy, split_strategy[:3])
        # 如果是alternating策略，添加n_splits参数
        if split_strategy == 'alternating' and n_splits:
            parts.append(f"{split_abbr}{n_splits}")
        else:
            parts.append(split_abbr)
    
    # 如果没有提取到足够信息，使用时间戳
    if len(parts) < 2:
        return f"exp_{datetime.now().strftime('%H%M%S')}"
    
    return "_".join(parts)


def run_experiment(command: str, exp_name: str, output_base_dir: Path, exp_number: int, total: int):
    """
    运行单个实验
    
    Args:
        command: 完整的命令行
        exp_name: 实验名称
        output_base_dir: 输出基础目录
        exp_number: 实验序号
        total: 总实验数
    """
    print(f"\n{'='*80}")
    print(f"[{exp_number}/{total}] 运行实验: {exp_name}")
    print(f"{'='*80}")
    print(f"命令: {command}")
    print(f"{'='*80}\n")
    
    # 创建实验目录
    exp_dir = output_base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存命令到文件
    command_file = exp_dir / "command.txt"
    with open(command_file, 'w', encoding='utf-8') as f:
        f.write(command + '\n')
    
    # 日志文件
    log_file = exp_dir / "output.log"
    
    # 修改命令以保存checkpoint到实验目录
    # 如果命令中没有指定checkpoint_dir，添加环境变量
    env = os.environ.copy()
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 创建结果目录（用于保存图片等）
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 修改命令，移除 'python main.py' 部分，直接执行
    if command.startswith('python main.py'):
        command = command.replace('python main.py', '', 1).strip()
    elif command.startswith('python3 main.py'):
        command = command.replace('python3 main.py', '', 1).strip()
    
    # 构建完整命令
    full_command = f"python main.py {command}"
    
    # 设置环境变量
    env['BATCH_RESULTS_DIR'] = str(results_dir)
    # 传递实验编号，让main.py在wandb名称前加上序号
    env['BATCH_EXP_NUMBER'] = f"{exp_number:02d}"
    
    start_time = time.time()
    
    # 执行命令
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {full_command}\n")
            f.write(f"{'='*80}\n\n")
            f.flush()
            
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd(),
                env=env
            )
            
            # 实时输出和保存
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            return_code = process.returncode
            
            duration = time.time() - start_time
            
            f.write(f"\n{'='*80}\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行时长: {duration:.2f}秒 ({duration/60:.2f}分钟)\n")
            f.write(f"返回码: {return_code}\n")
            
            if return_code == 0:
                print(f"\n✓ 实验完成: {exp_name} (耗时: {duration/60:.1f}分钟)")
                status = "success"
                
                # 提取测试指标
                metrics = extract_test_metrics(log_file)
            else:
                print(f"\n✗ 实验失败: {exp_name} (返回码: {return_code})")
                status = "failed"
                metrics = {}
            
            # 保存状态文件（包含指标）
            status_file = exp_dir / "status.txt"
            with open(status_file, 'w', encoding='utf-8') as sf:
                sf.write(f"{status}\n")
                sf.write(f"duration: {duration:.2f}\n")
                sf.write(f"return_code: {return_code}\n")
                if metrics:
                    sf.write(f"\nTest Metrics:\n")
                    for key, value in metrics.items():
                        sf.write(f"{key}: {value}\n")
            
            # 单独保存测试指标到 metrics.txt
            if metrics:
                metrics_file = exp_dir / "metrics.txt"
                with open(metrics_file, 'w', encoding='utf-8') as mf:
                    mf.write(f"Test Metrics for {exp_name}\n")
                    mf.write(f"{'='*50}\n")
                    for key, value in metrics.items():
                        mf.write(f"{key}: {value}\n")
                print(f"  指标已保存到: {metrics_file}")
            
            return status, duration, metrics
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n✗ 实验异常: {exp_name} - {str(e)}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"异常: {str(e)}\n")
        
        status_file = exp_dir / "status.txt"
        with open(status_file, 'w', encoding='utf-8') as sf:
            sf.write(f"error\n")
            sf.write(f"duration: {duration:.2f}\n")
            sf.write(f"error: {str(e)}\n")
        
        return "error", duration, {}


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量运行实验脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  1. 创建命令列表文件 experiments.txt:
     python main.py --model TCN --epochs 100 --lr 0.001 --mode window --window_size 10
     python main.py --model LSTM --epochs 100 --lr 0.001 --mode window --window_size 20
     python main.py --model CNN --epochs 100 --lr 0.005 --mode instant
  
  2. 运行批量实验:
     python run_batch.py experiments.txt
  
  3. 指定输出目录:
     python run_batch.py experiments.txt --output_dir ./my_results
        """
    )
    
    parser.add_argument('commands_file', type=str,
                       help='包含命令列表的文件，每行一个命令')
    parser.add_argument('--output_dir', type=str, default='./batch_results',
                       help='实验结果保存目录 (默认: ./batch_results)')
    parser.add_argument('--name', type=str, default=None,
                       help='本批次实验的名称（默认使用时间戳）')
    
    args = parser.parse_args()
    
    # 检查命令文件是否存在
    commands_file = Path(args.commands_file)
    if not commands_file.exists():
        print(f"错误: 命令文件不存在: {commands_file}")
        sys.exit(1)
    
    # 读取命令
    with open(commands_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤空行和注释
    commands = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            commands.append(line)
    
    if not commands:
        print("错误: 命令文件中没有有效命令")
        sys.exit(1)
    
    # 创建输出目录
    batch_name = args.name if args.name else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(args.output_dir) / batch_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"批量实验运行")
    print(f"{'='*80}")
    print(f"命令文件: {commands_file}")
    print(f"实验数量: {len(commands)}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*80}\n")
    
    # 运行所有实验
    results = []
    total_start = time.time()
    
    for i, command in enumerate(commands, 1):
        # 生成实验名称
        exp_name = f"exp{i:02d}_{parse_experiment_name(command)}"
        
        # 运行实验
        status, duration, metrics = run_experiment(command, exp_name, output_base_dir, i, len(commands))
        
        results.append({
            'number': i,
            'name': exp_name,
            'command': command,
            'status': status,
            'duration': duration,
            'metrics': metrics
        })
    
    total_duration = time.time() - total_start
    
    # 生成摘要
    print(f"\n{'='*80}")
    print(f"所有实验完成!")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n实验摘要:")
    print(f"  总数: {len(results)}")
    print(f"  成功: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  失败: {failed} ({failed/len(results)*100:.1f}%)")
    print(f"  错误: {errors} ({errors/len(results)*100:.1f}%)")
    print(f"  总耗时: {total_duration/3600:.2f} 小时")
    
    print(f"\n详细结果:")
    for r in results:
        status_symbol = "✓" if r['status'] == 'success' else "✗"
        metrics_str = ""
        if r.get('metrics'):
            mae = r['metrics'].get('MAE', r['metrics'].get('Mean_MAE', 'N/A'))
            rmse = r['metrics'].get('RMSE', r['metrics'].get('Mean_RMSE', 'N/A'))
            if mae != 'N/A':
                metrics_str = f" | MAE={mae:.2f} RMSE={rmse:.2f}"
        print(f"  {status_symbol} [{r['number']:2d}] {r['name']:<30s} {r['duration']/60:6.1f}分钟{metrics_str}")
    
    # 保存摘要文件
    summary_file = output_base_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"批量实验摘要\n")
        f.write(f"{'='*80}\n")
        f.write(f"批次名称: {batch_name}\n")
        f.write(f"开始时间: {datetime.fromtimestamp(total_start).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_duration/3600:.2f} 小时\n")
        f.write(f"\n实验统计:\n")
        f.write(f"  总数: {len(results)}\n")
        f.write(f"  成功: {successful}\n")
        f.write(f"  失败: {failed}\n")
        f.write(f"  错误: {errors}\n")
        f.write(f"\n详细结果:\n")
        f.write(f"{'='*80}\n")
        for r in results:
            status_symbol = "✓" if r['status'] == 'success' else "✗"
            f.write(f"{status_symbol} [{r['number']:2d}] {r['name']}\n")
            f.write(f"   命令: {r['command']}\n")
            f.write(f"   状态: {r['status']}\n")
            f.write(f"   耗时: {r['duration']:.2f}秒 ({r['duration']/60:.1f}分钟)\n")
            if r.get('metrics'):
                f.write(f"   指标:\n")
                for key, value in r['metrics'].items():
                    f.write(f"     {key}: {value:.4f}\n")
            f.write(f"\n")
    
    print(f"\n详细摘要已保存到: {summary_file}")
    print(f"所有结果保存在: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

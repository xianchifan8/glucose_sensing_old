"""
测试数据加载器
验证从 Dataset 读取数据的功能

改进：自动从目录名提取时间，无需手动配置
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import GlucoseDatasetLoader, split_data


def test_parse_dirname():
    """测试从目录名解析时间"""
    print("\n" + "="*70)
    print("测试 0: 从目录名解析起始时间")
    print("="*70)
    
    test_cases = [
        ("11201828_Tao", "11.20 18:28"),
        ("11141404_Tao", "11.14 14:04"),
        ("11211057_Tao", "11.21 10:57"),
    ]
    
    all_passed = True
    for dirname, expected in test_cases:
        result = GlucoseDatasetLoader.parse_start_time_from_dirname(dirname)
        if result == expected:
            print(f"  ✓ {dirname} -> {result}")
        else:
            print(f"  ✗ {dirname} -> {result} (期望: {expected})")
            all_passed = False
    
    return all_passed


def test_single_experiment():
    """测试加载单个实验（自动解析时间）"""
    print("\n" + "="*70)
    print("测试 1: 加载单个实验数据（自动解析时间）")
    print("="*70)
    
    # 使用绝对路径
    dataset_root = PROJECT_ROOT / "Dataset"
    loader = GlucoseDatasetLoader(dataset_root=str(dataset_root))
    
    # 测试 11201828_Tao 实验（无需手动指定时间）
    exp_dir = dataset_root / "Tao" / "11201828_Tao"
    
    result = loader.load_user_experiment(
        exp_dir,
        interpolation_method='linear'
    )
    
    if result:
        features, labels, metadata = result
        print("\n✓ 加载成功!")
        print(f"  特征 shape: {features.shape}")
        print(f"  标签 shape: {labels.shape}")
        print(f"  起始时间: {metadata['start_time']}")
        print(f"  元数据: {metadata}")
        return True
    else:
        print("\n✗ 加载失败")
        return False


def test_all_experiments():
    """测试加载所有实验（自动扫描和解析）"""
    print("\n" + "="*70)
    print("测试 2: 加载 Tao 用户的所有实验数据（自动扫描）")
    print("="*70)
    
    # 使用绝对路径
    dataset_root = PROJECT_ROOT / "Dataset"
    loader = GlucoseDatasetLoader(dataset_root=str(dataset_root))
    
    try:
        features, labels, metadata_list = loader.load_user_all_experiments(
            user_name="Tao",
            interpolation_method='linear'
        )
        
        print("\n✓ 所有实验加载成功!")
        print(f"  总特征 shape: {features.shape}")
        print(f"  总标签 shape: {labels.shape}")
        print(f"  加载的实验数: {len(metadata_list)}")
        
        # 显示每个实验的详情
        print("\n实验详情:")
        for i, meta in enumerate(metadata_list, 1):
            print(f"  {i}. {meta['experiment_dir']}: {meta['n_samples']} 样本, "
                  f"起始时间: {meta['start_time']}, "
                  f"血糖 {meta['glucose_min']:.2f}-{meta['glucose_max']:.2f} mmol/L")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalization():
    """测试特征归一化"""
    print("\n" + "="*70)
    print("测试 3: 特征归一化")
    print("="*70)
    
    # 使用绝对路径
    dataset_root = PROJECT_ROOT / "Dataset"
    loader = GlucoseDatasetLoader(dataset_root=str(dataset_root))
    
    try:
        features, labels, metadata = loader.load_user_all_experiments(
            user_name="Tao",
            interpolation_method='linear'
        )
        
        # 归一化
        features_normalized, scaler = loader.normalize_features(features)
        
        print("\n✓ 归一化成功!")
        print(f"  原始特征范围: [{features.min():.2f}, {features.max():.2f}]")
        print(f"  归一化后范围: [{features_normalized.min():.2f}, {features_normalized.max():.2f}]")
        print(f"  归一化后均值: {features_normalized.mean():.6f}")
        print(f"  归一化后标准差: {features_normalized.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 归一化失败: {e}")
        return False


def test_data_split():
    """测试数据划分"""
    print("\n" + "="*70)
    print("测试 4: 数据划分为训练/验证/测试集（真实数据）")
    print("="*70)
    
    # 使用真实数据进行划分测试
    dataset_root = PROJECT_ROOT / "Dataset"
    loader = GlucoseDatasetLoader(dataset_root=str(dataset_root))
    
    try:
        # 加载真实数据
        features, labels, metadata_list = loader.load_user_all_experiments(
            user_name="Tao",
            interpolation_method='linear'
        )
        
        print(f"\n加载的数据:")
        print(f"  总样本数: {len(features)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  血糖范围: {labels.min():.2f} - {labels.max():.2f} mmol/L")
        
        # 划分数据
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_data(
            features, labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        print(f"\n✓ 数据划分成功!")
        print(f"  训练集: {train_data.shape} - 血糖范围: {train_labels.min():.2f} - {train_labels.max():.2f} mmol/L")
        print(f"  验证集: {val_data.shape} - 血糖范围: {val_labels.min():.2f} - {val_labels.max():.2f} mmol/L")
        print(f"  测试集: {test_data.shape} - 血糖范围: {test_labels.min():.2f} - {test_labels.max():.2f} mmol/L")
        print(f"  比例: {len(train_data)/len(features):.2%} / "
              f"{len(val_data)/len(features):.2%} / "
              f"{len(test_data)/len(features):.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 数据划分失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Glucose Dataset Loader 测试套件 (自动时间解析版)")
    print("="*70)
    
    tests = [
        ("目录名时间解析", test_parse_dirname),
        ("单个实验加载", test_single_experiment),
        ("所有实验加载", test_all_experiments),
        ("特征归一化", test_normalization),
        ("数据划分", test_data_split),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} 出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n  总计: {passed}/{total} 通过")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

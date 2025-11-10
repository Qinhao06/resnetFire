#!/usr/bin/env python3
"""
测试数据集是否可以被正确加载
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入训练脚本中的数据集类
from train_resnet_fire_classifier import FireDataset

def test_dataset():
    """测试数据集加载"""
    data_dir = "./data/fire_dataset"
    
    print("=== 数据集测试 ===")
    print(f"数据目录: {data_dir}")
    
    # 检查目录结构
    fire_dir = os.path.join(data_dir, 'fire')
    no_fire_dir = os.path.join(data_dir, 'no_fire')
    
    print(f"\n目录检查:")
    print(f"火焰目录存在: {os.path.exists(fire_dir)}")
    print(f"非火焰目录存在: {os.path.exists(no_fire_dir)}")
    
    if os.path.exists(fire_dir):
        fire_count = len([f for f in os.listdir(fire_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"火焰图像数量: {fire_count}")
    
    if os.path.exists(no_fire_dir):
        no_fire_count = len([f for f in os.listdir(no_fire_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"非火焰图像数量: {no_fire_count}")
    
    # 测试数据集加载
    print(f"\n=== 测试数据集类 ===")
    try:
        dataset = FireDataset(data_dir)
        print(f"数据集加载成功!")
        print(f"总样本数: {len(dataset)}")
        
        # 测试获取第一个样本
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"第一个样本:")
            print(f"  - 图像形状: {image.size if hasattr(image, 'size') else 'N/A'}")
            print(f"  - 标签: {label} ({'火焰' if label == 1 else '非火焰'})")
        
        # 统计标签分布
        labels = [dataset[i][1] for i in range(min(100, len(dataset)))]  # 只检查前100个样本
        fire_samples = sum(labels)
        no_fire_samples = len(labels) - fire_samples
        print(f"\n标签分布 (前{len(labels)}个样本):")
        print(f"  - 火焰样本: {fire_samples}")
        print(f"  - 非火焰样本: {no_fire_samples}")
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return False
    
    return True

def main():
    success = test_dataset()
    if success:
        print(f"\n✅ 数据集测试通过! 可以开始训练 ResNet 模型。")
        print(f"\n使用方法:")
        print(f"python train_resnet_fire_classifier.py --data_dir ./data/fire_dataset")
    else:
        print(f"\n❌ 数据集测试失败，请检查数据集结构。")

if __name__ == '__main__':
    main()
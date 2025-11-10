#!/usr/bin/env python3
"""
简化版火焰图像提取脚本
快速从YOLO数据集中提取所有包含火焰标注的图像作为正样本
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_fire_images(yolo_dataset_path, output_fire_dir):
    """
    从YOLO数据集中提取火焰图像
    
    Args:
        yolo_dataset_path (str): YOLO数据集路径
        output_fire_dir (str): 输出火焰图像目录
    """
    yolo_path = Path(yolo_dataset_path)
    fire_dir = Path(output_fire_dir)
    fire_dir.mkdir(parents=True, exist_ok=True)
    
    total_fire_images = 0
    
    # 处理train和valid分割
    for split in ['train', 'valid']:
        split_path = yolo_path / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print(f"跳过 {split} 分割 (目录不存在)")
            continue
        
        print(f"\n处理 {split} 分割...")
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_path.glob(ext))
        
        fire_count = 0
        
        # 检查每张图像
        for image_file in tqdm(image_files, desc=f"检查{split}图像"):
            # 对应的标签文件
            label_file = labels_path / (image_file.stem + '.txt')
            
            # 如果标签文件存在且不为空，说明包含火焰标注
            if label_file.exists() and label_file.stat().st_size > 0:
                # 复制图像到火焰目录
                output_name = f"{split}_{image_file.name}"
                output_path = fire_dir / output_name
                shutil.copy2(image_file, output_path)
                fire_count += 1
        
        print(f"{split} 分割: 提取了 {fire_count} 张火焰图像")
        total_fire_images += fire_count
    
    print(f"\n总计提取了 {total_fire_images} 张火焰图像")
    print(f"保存位置: {fire_dir}")
    
    return total_fire_images

if __name__ == '__main__':
    # 配置路径
    yolo_dataset_path = './FireData_filtered'  # YOLO数据集路径
    output_fire_dir = './data/fire'            # 输出火焰图像目录
    
    print("开始提取火焰图像...")
    print("="*50)
    
    # 检查输入路径
    if not os.path.exists(yolo_dataset_path):
        print(f"错误: YOLO数据集路径不存在: {yolo_dataset_path}")
        exit(1)
    
    # 提取火焰图像
    total_images = extract_fire_images(yolo_dataset_path, output_fire_dir)
    
    print("="*50)
    print("提取完成!")
    print(f"共提取 {total_images} 张火焰图像作为正样本")
#!/usr/bin/env python3
"""
从YOLO格式的火焰检测数据集中创建二分类数据集的正样本（火焰类）
从FireData_filtered数据集中提取包含火焰标注的图像
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

class FireDatasetCreator:
    """火焰数据集创建器"""
    
    def __init__(self, yolo_dataset_path, output_path):
        """
        Args:
            yolo_dataset_path (str): YOLO数据集路径
            output_path (str): 输出数据集路径
        """
        self.yolo_dataset_path = Path(yolo_dataset_path)
        self.output_path = Path(output_path)
        
        # 创建输出目录结构
        self.fire_dir = self.output_path / 'fire'
        self.fire_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取data.yaml文件获取类别信息
        self.class_names = self._load_class_info()
        
        print(f"YOLO数据集路径: {self.yolo_dataset_path}")
        print(f"输出路径: {self.output_path}")
        print(f"类别信息: {self.class_names}")
    
    def _load_class_info(self):
        """加载类别信息"""
        data_yaml_path = self.yolo_dataset_path / 'data.yaml'
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('names', ['Fire'])
        else:
            print("警告: 未找到data.yaml文件，使用默认类别名称")
            return ['Fire']
    
    def _has_fire_annotations(self, label_file_path):
        """
        检查标签文件是否包含火焰标注
        
        Args:
            label_file_path (Path): 标签文件路径
            
        Returns:
            bool: 是否包含火焰标注
        """
        if not label_file_path.exists():
            return False
        
        try:
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:  # class_id x_center y_center width height
                        class_id = int(parts[0])
                        # 假设fire类别的ID为0（根据data.yaml）
                        if class_id == 0:
                            return True
            return False
        except Exception as e:
            print(f"读取标签文件失败: {label_file_path}, 错误: {e}")
            return False
    
    def _crop_fire_regions(self, image_path, label_path, save_crops=False):
        """
        根据标注裁剪火焰区域（可选功能）
        
        Args:
            image_path (Path): 图像路径
            label_path (Path): 标签路径
            save_crops (bool): 是否保存裁剪的火焰区域
            
        Returns:
            list: 裁剪的火焰区域图像列表
        """
        if not save_crops:
            return []
        
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            h, w = image.shape[:2]
            crops = []
            
            # 读取标注
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == 0:  # Fire类别
                            # YOLO格式: x_center, y_center, width, height (归一化坐标)
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 转换为像素坐标
                            x_center_px = int(x_center * w)
                            y_center_px = int(y_center * h)
                            width_px = int(width * w)
                            height_px = int(height * h)
                            
                            # 计算边界框
                            x1 = max(0, x_center_px - width_px // 2)
                            y1 = max(0, y_center_px - height_px // 2)
                            x2 = min(w, x_center_px + width_px // 2)
                            y2 = min(h, y_center_px + height_px // 2)
                            
                            # 裁剪火焰区域
                            if x2 > x1 and y2 > y1:
                                crop = image[y1:y2, x1:x2]
                                crops.append(crop)
            
            return crops
            
        except Exception as e:
            print(f"裁剪火焰区域失败: {image_path}, 错误: {e}")
            return []
    
    def process_split(self, split_name, save_full_images=True, save_cropped_regions=False, 
                     min_crop_size=50):
        """
        处理指定的数据集分割（train/valid）
        
        Args:
            split_name (str): 分割名称 ('train' 或 'valid')
            save_full_images (bool): 是否保存完整图像
            save_cropped_regions (bool): 是否保存裁剪的火焰区域
            min_crop_size (int): 裁剪区域的最小尺寸
        """
        split_path = self.yolo_dataset_path / split_name
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print(f"警告: {split_name} 分割的图像或标签目录不存在")
            return
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_path.glob(ext))
        
        print(f"\n处理 {split_name} 分割: 找到 {len(image_files)} 张图像")
        
        fire_images_count = 0
        cropped_regions_count = 0
        
        # 创建输出子目录
        if save_full_images:
            full_images_dir = self.fire_dir / f'{split_name}_full_images'
            full_images_dir.mkdir(exist_ok=True)
        
        if save_cropped_regions:
            cropped_dir = self.fire_dir / f'{split_name}_cropped_regions'
            cropped_dir.mkdir(exist_ok=True)
        
        # 处理每张图像
        for image_file in tqdm(image_files, desc=f"处理{split_name}图像"):
            # 对应的标签文件
            label_file = labels_path / (image_file.stem + '.txt')
            
            # 检查是否包含火焰标注
            if self._has_fire_annotations(label_file):
                fire_images_count += 1
                
                # 保存完整图像
                if save_full_images:
                    output_image_path = full_images_dir / f"{split_name}_{image_file.name}"
                    shutil.copy2(image_file, output_image_path)
                
                # 保存裁剪的火焰区域
                if save_cropped_regions:
                    crops = self._crop_fire_regions(image_file, label_file, save_crops=True)
                    for i, crop in enumerate(crops):
                        if crop.shape[0] >= min_crop_size and crop.shape[1] >= min_crop_size:
                            crop_filename = f"{split_name}_{image_file.stem}_crop_{i}.jpg"
                            crop_path = cropped_dir / crop_filename
                            cv2.imwrite(str(crop_path), crop)
                            cropped_regions_count += 1
        
        print(f"{split_name} 分割处理完成:")
        print(f"  - 包含火焰的图像: {fire_images_count} 张")
        if save_cropped_regions:
            print(f"  - 裁剪的火焰区域: {cropped_regions_count} 个")
    
    def create_dataset(self, save_full_images=True, save_cropped_regions=False, 
                      min_crop_size=50, splits=['train', 'valid']):
        """
        创建火焰数据集
        
        Args:
            save_full_images (bool): 是否保存完整图像
            save_cropped_regions (bool): 是否保存裁剪的火焰区域
            min_crop_size (int): 裁剪区域的最小尺寸
            splits (list): 要处理的数据集分割
        """
        print("开始创建火焰数据集...")
        print("="*50)
        
        total_fire_images = 0
        total_cropped_regions = 0
        
        # 处理每个分割
        for split in splits:
            if (self.yolo_dataset_path / split).exists():
                self.process_split(split, save_full_images, save_cropped_regions, min_crop_size)
            else:
                print(f"警告: {split} 分割不存在，跳过")
        
        # 统计总数
        if save_full_images:
            for split in splits:
                full_images_dir = self.fire_dir / f'{split}_full_images'
                if full_images_dir.exists():
                    count = len(list(full_images_dir.glob('*')))
                    total_fire_images += count
        
        if save_cropped_regions:
            for split in splits:
                cropped_dir = self.fire_dir / f'{split}_cropped_regions'
                if cropped_dir.exists():
                    count = len(list(cropped_dir.glob('*')))
                    total_cropped_regions += count
        
        print("\n" + "="*50)
        print("数据集创建完成!")
        print(f"输出目录: {self.output_path}")
        print(f"火焰图像总数: {total_fire_images}")
        if save_cropped_regions:
            print(f"裁剪区域总数: {total_cropped_regions}")
        
        # 创建统计信息文件
        stats = {
            'source_dataset': str(self.yolo_dataset_path),
            'output_path': str(self.output_path),
            'class_names': self.class_names,
            'total_fire_images': total_fire_images,
            'total_cropped_regions': total_cropped_regions,
            'splits_processed': splits,
            'settings': {
                'save_full_images': save_full_images,
                'save_cropped_regions': save_cropped_regions,
                'min_crop_size': min_crop_size
            }
        }
        
        stats_file = self.output_path / 'dataset_stats.yaml'
        with open(stats_file, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
        
        print(f"统计信息已保存到: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='从YOLO数据集创建火焰二分类正样本数据集')
    parser.add_argument('--yolo_dataset', type=str, 
                       default='./FireData_filtered',
                       help='YOLO数据集路径')
    parser.add_argument('--output_dir', type=str, 
                       default='./data',
                       help='输出数据集目录')
    parser.add_argument('--save_full_images', action='store_true', default=True,
                       help='保存完整图像')
    parser.add_argument('--save_cropped_regions', action='store_true', default=False,
                       help='保存裁剪的火焰区域')
    parser.add_argument('--min_crop_size', type=int, default=50,
                       help='裁剪区域的最小尺寸')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid'],
                       help='要处理的数据集分割')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.yolo_dataset):
        print(f"错误: YOLO数据集路径不存在: {args.yolo_dataset}")
        return
    
    # 创建火焰数据集
    creator = FireDatasetCreator(args.yolo_dataset, args.output_dir)
    creator.create_dataset(
        save_full_images=args.save_full_images,
        save_cropped_regions=args.save_cropped_regions,
        min_crop_size=args.min_crop_size,
        splits=args.splits
    )

if __name__ == '__main__':
    main()
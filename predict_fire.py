#!/usr/bin/env python3
"""
ResNet火焰分类器推理脚本
用于使用训练好的模型对单张图像或批量图像进行火焰检测
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
import json
import numpy as np
from train_resnet_fire_classifier import ResNetFireClassifier

class FirePredictor:
    """火焰检测预测器"""
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path (str): 训练好的模型路径
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型参数
        model_name = checkpoint.get('model_name', 'resnet50')
        input_size = checkpoint.get('input_size', 224)
        
        # 创建模型
        self.model = ResNetFireClassifier(num_classes=2, pretrained=False, model_name=model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 设置图像变换
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['无火焰', '火焰']
        
        print(f"模型加载成功: {model_name}")
        print(f"使用设备: {self.device}")
        print(f"最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    def predict_single(self, image_path, return_confidence=True):
        """
        预测单张图像
        
        Args:
            image_path (str): 图像路径
            return_confidence (bool): 是否返回置信度
            
        Returns:
            tuple: (预测类别, 置信度) 或 预测类别
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                if return_confidence:
                    return self.class_names[predicted_class], confidence_score
                else:
                    return self.class_names[predicted_class]
                    
        except Exception as e:
            print(f"预测图像 {image_path} 时出错: {e}")
            if return_confidence:
                return "错误", 0.0
            else:
                return "错误"
    
    def predict_batch(self, image_paths):
        """
        批量预测多张图像
        
        Args:
            image_paths (list): 图像路径列表
            
        Returns:
            list: 预测结果列表，每个元素为 (图像路径, 预测类别, 置信度)
        """
        results = []
        
        for image_path in image_paths:
            predicted_class, confidence = self.predict_single(image_path, return_confidence=True)
            results.append((image_path, predicted_class, confidence))
        
        return results
    
    def predict_directory(self, directory_path, output_file=None):
        """
        预测目录中的所有图像
        
        Args:
            directory_path (str): 图像目录路径
            output_file (str): 输出结果文件路径 (可选)
            
        Returns:
            list: 预测结果列表
        """
        # 获取所有图像文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = []
        
        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith(image_extensions):
                image_paths.append(os.path.join(directory_path, file_name))
        
        if not image_paths:
            print(f"在目录 {directory_path} 中未找到图像文件")
            return []
        
        print(f"找到 {len(image_paths)} 张图像，开始预测...")
        
        # 批量预测
        results = self.predict_batch(image_paths)
        
        # 统计结果
        fire_count = sum(1 for _, pred, _ in results if pred == '火焰')
        no_fire_count = len(results) - fire_count
        
        print(f"\n预测完成!")
        print(f"总图像数: {len(results)}")
        print(f"火焰图像: {fire_count}")
        print(f"无火焰图像: {no_fire_count}")
        
        # 保存结果到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("图像路径,预测结果,置信度\n")
                for image_path, prediction, confidence in results:
                    f.write(f"{image_path},{prediction},{confidence:.4f}\n")
            print(f"结果已保存到: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='ResNet火焰分类器推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--image_path', type=str,
                       help='单张图像路径')
    parser.add_argument('--image_dir', type=str,
                       help='图像目录路径 (批量预测)')
    parser.add_argument('--output_file', type=str,
                       help='输出结果文件路径 (仅用于批量预测)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建预测器
    predictor = FirePredictor(args.model_path)
    
    if args.image_path:
        # 单张图像预测
        if not os.path.exists(args.image_path):
            print(f"错误: 图像文件不存在: {args.image_path}")
            return
        
        print(f"\n预测图像: {args.image_path}")
        predicted_class, confidence = predictor.predict_single(args.image_path)
        print(f"预测结果: {predicted_class}")
        print(f"置信度: {confidence:.4f}")
        
    elif args.image_dir:
        # 批量预测
        if not os.path.exists(args.image_dir):
            print(f"错误: 目录不存在: {args.image_dir}")
            return
        
        results = predictor.predict_directory(args.image_dir, args.output_file)
        
        # 显示部分结果
        if results:
            print(f"\n前10个预测结果:")
            for i, (image_path, prediction, confidence) in enumerate(results[:10]):
                print(f"{i+1}. {os.path.basename(image_path)}: {prediction} ({confidence:.4f})")
            
            if len(results) > 10:
                print(f"... 还有 {len(results) - 10} 个结果")
    
    else:
        print("错误: 请提供 --image_path 或 --image_dir 参数")

if __name__ == '__main__':
    main()
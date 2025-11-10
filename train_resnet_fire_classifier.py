#!/usr/bin/env python3
"""
ResNet火焰二分类器训练脚本
用于训练ResNet模型判断图像中是否包含火焰
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime

class FireDataset(Dataset):
    """火焰数据集类"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据目录路径，应包含fire和no_fire两个子文件夹
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载火焰图像 (标签为1)
        fire_dir = os.path.join(data_dir, 'fire')
        if os.path.exists(fire_dir):
            for img_name in os.listdir(fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fire_dir, img_name))
                    self.labels.append(1)
        
        # 加载非火焰图像 (标签为0)
        no_fire_dir = os.path.join(data_dir, 'no_fire')
        if os.path.exists(no_fire_dir):
            for img_name in os.listdir(no_fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(no_fire_dir, img_name))
                    self.labels.append(0)
        
        print(f"加载了 {len(self.image_paths)} 张图像")
        print(f"火焰图像: {sum(self.labels)} 张")
        print(f"非火焰图像: {len(self.labels) - sum(self.labels)} 张")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            # 返回一个默认的黑色图像
            if self.transform:
                default_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                default_image = Image.new('RGB', (224, 224), (0, 0, 0))
            return default_image, label

class ResNetFireClassifier(nn.Module):
    """基于ResNet的火焰分类器"""
    
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet50'):
        super(ResNetFireClassifier, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 如果使用预训练模型，冻结前面的层
        if pretrained:
            for param in list(self.backbone.parameters())[:-10]:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

def get_transforms(input_size=224):
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="训练中")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct_predictions/total_samples:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证中")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_predictions, all_labels

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['无火焰', '火焰'], 
                yticklabels=['无火焰', '火焰'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='ResNet火焰二分类训练')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='数据目录路径 (应包含fire和no_fire子文件夹)')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                       help='ResNet模型名称')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--input_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练模型')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    train_transform, val_transform = get_transforms(args.input_size)
    
    # 加载数据集
    print("加载数据集...")
    dataset = FireDataset(args.data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("错误: 未找到数据！请确保数据目录包含fire和no_fire子文件夹")
        return
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 为验证集设置不同的变换
    val_dataset.dataset.transform = val_transform
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    # 创建模型
    print(f"创建{args.model_name}模型...")
    model = ResNetFireClassifier(num_classes=2, pretrained=args.pretrained, model_name=args.model_name)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f'best_{args.model_name}_fire_classifier.pt')
    
    print("\n开始训练...")
    print("="*50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, precision, recall, f1, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_name': args.model_name,
                'input_size': args.input_size
            }, best_model_path)
            print(f"保存最佳模型 (验证准确率: {best_val_acc:.4f})")
    
    print("\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终验证
    final_val_loss, final_val_acc, final_precision, final_recall, final_f1, final_preds, final_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    print("\n最终评估结果:")
    print(f"验证准确率: {final_val_acc:.4f}")
    print(f"精确率: {final_precision:.4f}")
    print(f"召回率: {final_recall:.4f}")
    print(f"F1分数: {final_f1:.4f}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_results': {
            'val_acc': final_val_acc,
            'precision': final_precision,
            'recall': final_recall,
            'f1': final_f1
        },
        'training_args': vars(args)
    }
    
    history_path = os.path.join(args.save_dir, f'{args.model_name}_training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 绘制训练曲线
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(args.save_dir, f'{args.model_name}_training_curves.png'))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(final_labels, final_preds,
                         os.path.join(args.save_dir, f'{args.model_name}_confusion_matrix.png'))
    
    print(f"\n模型已保存到: {best_model_path}")
    print(f"训练历史已保存到: {history_path}")

if __name__ == '__main__':
    main()
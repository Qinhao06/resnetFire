# ResNet火焰二分类器

这是一个基于ResNet的火焰检测二分类器，用于判断图像中是否包含火焰。

## 文件说明

- `train_resnet_fire_classifier.py`: 主训练脚本
- `predict_fire.py`: 推理脚本，用于预测新图像
- `requirements.txt`: 依赖包列表
- `README.md`: 使用说明

## 环境设置

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 数据准备

### 方法1: 从YOLO数据集自动提取火焰图像

如果你有YOLO格式的火焰检测数据集，可以使用提供的脚本自动提取火焰图像：

#### 快速提取（推荐）
```bash
python extract_fire_images.py
```

#### 高级提取（支持更多选项）
```bash
python create_fire_dataset.py \
    --yolo_dataset ./FireData_filtered \
    --output_dir ./data \
    --save_full_images \
    --splits train valid
```

### 方法2: 手动创建数据集

创建以下目录结构：
```
data/
├── fire/          # 包含火焰的图像
│   ├── fire1.jpg
│   ├── fire2.jpg
│   └── ...
└── no_fire/       # 不包含火焰的图像
    ├── no_fire1.jpg
    ├── no_fire2.jpg
    └── ...
```

支持的图像格式：.png, .jpg, .jpeg

### 数据集提取脚本说明

- `extract_fire_images.py`: 简化版提取脚本，快速从YOLO数据集提取所有火焰图像
- `create_fire_dataset.py`: 完整版创建脚本，支持多种提取选项：
  - 保存完整图像
  - 裁剪火焰区域
  - 自定义输出目录
  - 选择处理的数据集分割

## 训练模型

### 基本用法
```bash
python train_resnet_fire_classifier.py --data_dir ./data
```

### 高级参数
```bash
python train_resnet_fire_classifier.py \
    --data_dir ./data \
    --model_name resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --input_size 224 \
    --val_split 0.2 \
    --save_dir ./models
```

### 参数说明
- `--data_dir`: 数据目录路径
- `--model_name`: ResNet模型类型 (resnet18, resnet34, resnet50, resnet101)
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--input_size`: 输入图像尺寸
- `--val_split`: 验证集比例
- `--save_dir`: 模型保存目录

## 使用训练好的模型进行预测

### 预测单张图像
```bash
python predict_fire.py \
    --model_path ./models/best_resnet50_fire_classifier.pt \
    --image_path ./test_image.jpg
```

### 批量预测目录中的所有图像
```bash
python predict_fire.py \
    --model_path ./models/best_resnet50_fire_classifier.pt \
    --image_dir ./test_images \
    --output_file ./results.csv
```

## 训练输出

训练过程中会生成以下文件：
- `best_[model_name]_fire_classifier.pt`: 最佳模型权重
- `[model_name]_training_history.json`: 训练历史记录
- `[model_name]_training_curves.png`: 训练曲线图
- `[model_name]_confusion_matrix.png`: 混淆矩阵图

## 模型特性

- **数据增强**: 包含随机翻转、旋转、颜色抖动等
- **预训练模型**: 使用ImageNet预训练权重
- **迁移学习**: 冻结前面的层，只训练最后几层
- **正则化**: 使用Dropout防止过拟合
- **学习率调度**: 使用StepLR动态调整学习率
- **早停机制**: 保存验证集上表现最好的模型

## 性能指标

训练完成后会显示以下指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵

## 注意事项

1. 确保数据集平衡，火焰和非火焰图像数量相近
2. 图像质量要好，避免模糊或过暗的图像
3. 如果GPU内存不足，可以减小batch_size
4. 训练时间取决于数据集大小和模型复杂度
5. 建议使用GPU加速训练

## 故障排除

1. **CUDA内存不足**: 减小batch_size或换用更小的模型
2. **数据加载错误**: 检查数据目录结构和图像格式
3. **训练不收敛**: 尝试调整学习率或增加训练轮数
4. **过拟合**: 增加数据增强或调整Dropout比例
#!/usr/bin/env python3
"""
使用YOLO模型检测nofire4文件夹下的图片
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse

def detect_images():
    """检测nofire4文件夹下的所有图片,并将检测到的火焰区域剪切保存"""
    
    # 设置参数
    model_path = './best.pt'
    input_dir = './nofire4'
    output_dir = './detection_results'
    cropped_dir = './nofire'  # 保存剪切后的火焰区域
    conf_threshold = 0.3
    
    print("=" * 50)
    print("YOLO 火焰检测程序")
    print("=" * 50)
    
    try:
        # 设置torch使用单线程，避免冲突
        torch.set_num_threads(1)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # 加载YOLO模型
        print(f"加载YOLO模型: {model_path}")
        model = YOLO(model_path)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建剪切图像保存目录
        cropped_path = Path(cropped_dir)
        cropped_path.mkdir(exist_ok=True)
        print(f"剪切后的火焰区域将保存到: {cropped_dir}")
        
        # 获取输入目录
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"错误: 输入目录不存在 - {input_dir}")
            return
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"在 {input_dir} 中未找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        print("-" * 50)
        
        detection_results = []
        total_cropped = 0
        
        # 逐个处理图片
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {image_file.name}")
            
            try:
                # 读取原始图像
                img = cv2.imread(str(image_file))
                if img is None:
                    print(f"  错误: 无法读取图像")
                    continue
                
                # 进行检测
                results = model.predict(
                    source=str(image_file),
                    conf=conf_threshold,
                    save=False,
                    verbose=False
                )
                
                # 分析检测结果
                detections = []
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            # 获取检测信息
                            coords = boxes.xyxy.cpu().numpy()  # 坐标
                            confs = boxes.conf.cpu().numpy()   # 置信度
                            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                            
                            for j, (coord, conf) in enumerate(zip(coords, confs)):
                                x1, y1, x2, y2 = coord
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class': int(classes[j]) if len(classes) > j else 0
                                }
                                detections.append(detection)
                                
                                # 剪切火焰区域并保存
                                x1, y1, x2, y2 = detection['bbox']
                                # 确保坐标在图像范围内
                                h, w = img.shape[:2]
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x2)
                                y2 = min(h, y2)
                                
                                # 剪切区域
                                cropped_img = img[y1:y2, x1:x2]
                                
                                # 生成保存文件名
                                base_name = image_file.stem
                                ext = image_file.suffix
                                cropped_filename = f"{base_name}_crop{j+1}{ext}"
                                cropped_filepath = cropped_path / cropped_filename
                                
                                # 保存剪切的图像
                                cv2.imwrite(str(cropped_filepath), cropped_img)
                                total_cropped += 1
                
                # 记录结果
                result_info = {
                    'filename': image_file.name,
                    'path': str(image_file),
                    'detections': detections,
                    'fire_detected': len(detections) > 0
                }
                detection_results.append(result_info)
                
                # 输出检测结果
                if detections:
                    print(f"  ✓ 检测到 {len(detections)} 个火焰区域")
                    for j, det in enumerate(detections):
                        bbox = det['bbox']
                        conf = det['confidence']
                        print(f"    区域{j+1}: 坐标({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) 置信度:{conf:.3f}")
                        base_name = image_file.stem
                        ext = image_file.suffix
                        print(f"    已保存: {base_name}_crop{j+1}{ext}")
                else:
                    print(f"  ✗ 未检测到火焰")
                
            except Exception as e:
                print(f"  错误: 处理图片时发生异常 - {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 输出统计结果
        print("-" * 50)
        print("检测统计:")
        total_images = len(detection_results)
        fire_images = sum(1 for r in detection_results if r['fire_detected'])
        no_fire_images = total_images - fire_images
        total_detections = sum(len(r['detections']) for r in detection_results)
        
        print(f"总图片数: {total_images}")
        print(f"检测到火焰的图片: {fire_images}")
        print(f"未检测到火焰的图片: {no_fire_images}")
        print(f"总检测区域数: {total_detections}")
        print(f"剪切并保存的火焰区域: {total_cropped}")
        print(f"剪切图像保存目录: {cropped_dir}")
        
        # 保存检测结果到文件
        result_file = output_path / 'detection_results.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("YOLO火焰检测结果\n")
            f.write("=" * 50 + "\n\n")
            
            for result in detection_results:
                f.write(f"文件: {result['filename']}\n")
                f.write(f"路径: {result['path']}\n")
                f.write(f"检测到火焰: {'是' if result['fire_detected'] else '否'}\n")
                
                if result['detections']:
                    f.write(f"检测区域数: {len(result['detections'])}\n")
                    for j, det in enumerate(result['detections']):
                        bbox = det['bbox']
                        conf = det['confidence']
                        f.write(f"  区域{j+1}: 坐标({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) 置信度:{conf:.3f}\n")
                f.write("-" * 30 + "\n")
            
            f.write(f"\n统计信息:\n")
            f.write(f"总图片数: {total_images}\n")
            f.write(f"检测到火焰的图片: {fire_images}\n")
            f.write(f"未检测到火焰的图片: {no_fire_images}\n")
            f.write(f"总检测区域数: {total_detections}\n")
            f.write(f"剪切并保存的火焰区域: {total_cropped}\n")
            f.write(f"剪切图像保存目录: {cropped_dir}\n")
        
        print(f"\n检测结果已保存到: {result_file}")
        print("=" * 50)
        print("检测完成！")
        print(f"✓ 共剪切保存了 {total_cropped} 个火焰区域到 {cropped_dir} 目录")
        print("=" * 50)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='YOLO火焰检测')
    parser.add_argument('--model', default='./best.pt', help='YOLO模型路径')
    parser.add_argument('--input', default='./nofire4', help='输入图片目录')
    parser.add_argument('--output', default='./detection_results', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 可以通过参数自定义，但默认使用固定值
    detect_images()

if __name__ == '__main__':
    main()
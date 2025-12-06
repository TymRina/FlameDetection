from ultralytics import YOLO
import os
import glob
import cv2

# 类别名称映射
CLASS_NAMES = {
    0: "扩散火焰",
    1: "预混火焰", 
    2: "不完全预混火焰",
    3: "异常火焰"
}

# 目标测试类别
TARGET_CLASS = 2  # 不完全预混火焰
target_class_name = CLASS_NAMES[TARGET_CLASS]

def main():
    print(f"=== 不完全预混火焰（{target_class_name}）检测测试 ===\n")
    
    # 加载模型
    model_path_optimized = 'runs/detect/fire_train_optimized/weights/best.pt'
    model_path_original = 'runs/detect/fire_train/weights/best.pt'
    
    if os.path.exists(model_path_optimized):
        model_path = model_path_optimized
        print(f"加载优化后的模型: {model_path}")
    elif os.path.exists(model_path_original):
        model_path = model_path_original
        print(f"加载原始模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path_optimized} 或 {model_path_original}")
        print("请先运行训练脚本")
        return
    
    model = YOLO(model_path)
    
    # 创建结果保存目录
    result_dir = f'runs/detect/{target_class_name}_test'
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 首先检查验证集中的不完全预混火焰样本
    print("\n1. 检查验证集中的不完全预混火焰样本:")
    
    # 遍历验证集图像
    val_images = glob.glob('dataset/valid/images/*.PNG')
    val_images += glob.glob('dataset/valid/images/*.png')
    val_images += glob.glob('dataset/valid/images/*.jpg')
    
    # 统计验证集中的不完全预混火焰数量
    total_target_samples = 0
    detected_samples = 0
    
    for image_path in val_images:
        # 获取对应的标签文件
        label_path = image_path.replace('images', 'labels').replace('.PNG', '.txt').replace('.png', '.txt').replace('.jpg', '.txt')
        
        if os.path.exists(label_path):
            # 读取标签
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # 检查是否包含目标类别
            has_target = any(int(line.split()[0]) == TARGET_CLASS for line in labels)
            
            if has_target:
                total_target_samples += 1
                
                # 运行推理
                results = model(image_path, device='0', imgsz=640)
                
                # 检查是否检测到目标类别
                boxes = results[0].boxes
                detected_class = any(int(cls) == TARGET_CLASS for cls in boxes.cls)
                
                if detected_class:
                    detected_samples += 1
                    
                    # 保存检测结果
                    result_image = results[0].plot(labels=True, conf=True)
                    result_image_path = os.path.join(result_dir, os.path.basename(image_path))
                    cv2.imwrite(result_image_path, result_image)
                    
                    print(f"✓ 检测成功: {os.path.basename(image_path)}")
                else:
                    # 保存未检测到的样本
                    result_image = results[0].plot(labels=True, conf=True)
                    result_image_path = os.path.join(result_dir, f"未检测到_{os.path.basename(image_path)}")
                    cv2.imwrite(result_image_path, result_image)
                    
                    print(f"✗ 检测失败: {os.path.basename(image_path)}")
    
    # 输出验证集统计结果
    print(f"\n验证集测试结果:")
    print(f"总样本数: {total_target_samples}")
    print(f"检测成功: {detected_samples}")
    print(f"检测成功率: {detected_samples/total_target_samples*100:.2f}%")
    
    # 2. 检查火焰数据集中的不完全预混火焰样本
    print("\n\n2. 检查原始火焰数据集中的不完全预混火焰样本:")
    
    # 假设火焰数据集在这个路径
    flame_dataset_path = "火焰数据集"
    if os.path.exists(flame_dataset_path):
        # 遍历火焰数据集
        incomplete_flame_dir = os.path.join(flame_dataset_path, "1-预混火焰------1000---ok")  # 假设是这个目录
        if os.path.exists(incomplete_flame_dir):
            test_images = glob.glob(os.path.join(incomplete_flame_dir, "*.PNG"))
            test_images += glob.glob(os.path.join(incomplete_flame_dir, "*.png"))
            test_images += glob.glob(os.path.join(incomplete_flame_dir, "*.jpg"))
            
            print(f"找到 {len(test_images)} 张不完全预混火焰图像进行测试")
            
            # 随机选择10张进行测试
            import random
            test_images = random.sample(test_images, min(10, len(test_images)))
            
            for image_path in test_images:
                # 运行推理
                results = model(image_path, device='0', imgsz=640)
                
                # 检查是否检测到目标类别
                boxes = results[0].boxes
                detected_class = any(int(cls) == TARGET_CLASS for cls in boxes.cls)
                
                # 保存结果
                result_image = results[0].plot(labels=True, conf=True)
                result_image_path = os.path.join(result_dir, f"原始_{os.path.basename(image_path)}")
                cv2.imwrite(result_image_path, result_image)
                
                status = "✓ 检测成功" if detected_class else "✗ 检测失败"
                print(f"{status}: {os.path.basename(image_path)}")
    
    print(f"\n\n=== 测试完成！结果保存在 {result_dir} ===")
    print(f"共测试 {total_target_samples} 个不完全预混火焰样本")
    print(f"成功检测: {detected_samples}")
    print(f"检测成功率: {detected_samples/total_target_samples*100:.2f}%")

if __name__ == '__main__':
    main()

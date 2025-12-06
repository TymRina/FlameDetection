from ultralytics import YOLO
import os

def main():
    # 加载训练好的模型
    # 尝试加载优化后的模型，如果不存在则加载原始模型
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
        print("请先运行训练脚本或检查模型路径是否正确")
        return
    
    model = YOLO(model_path)
    
    # 执行验证
    results = model.val(
        data='dataset/data.yaml',  # 数据集配置文件（使用相对路径）
        split='val',  # 验证集
        batch=4,  # 增加批次大小
        workers=0,  # 工作线程数
        device='0',  # 使用GPU验证，如果没有GPU可以改为'cpu'
        save_json=True,  # 保存JSON格式的结果
        save_txt=True,  # 保存TXT格式的结果
        project='runs/detect',  # 保存项目名称
        name='fire_val',  # 验证任务名称
        exist_ok=True,  # 如果存在同名项目则覆盖
        verbose=True  # 显示详细信息
    )
    
    # 输出验证结果
    print("\n验证结果:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    print(f"精确率(Precision): {results.box.mp:.3f}")
    print(f"召回率(Recall): {results.box.mr:.3f}")
    
    # 输出各分类详细结果
    print("\n各分类详细结果:")
    for i, name in model.names.items():
        print(f"\n类别 {name} (ID: {i}):")
        print(f"  mAP50: {results.box.ap50[i]:.3f}")
        print(f"  mAP50-95: {results.box.maps[i]:.3f}")
        print(f"  精确率: {results.box.p[i]:.3f}")
        print(f"  召回率: {results.box.r[i]:.3f}")
    
    print("\n验证完成!")
    print(f"结果保存路径: {results.save_dir}")

if __name__ == '__main__':
    main()
from ultralytics import YOLO
import os

def main():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8n预训练模型
    
    # 训练模型
    results = model.train(
        data='dataset/data.yaml',  # 数据集配置文件（使用相对路径）
        epochs=30,  # 增加训练轮数到30
        batch=4,  # 适当增加批次大小
        workers=0,  # 禁用多线程数据加载以降低内存占用
        amp=False,  # 禁用混合精度训练
        deterministic=False,  # 非确定性训练
        device='0',  # 使用GPU训练
        save_period=5,  # 每5轮保存一次模型
        plots=True,  # 生成训练过程中的图表
        project='runs/detect',  # 保存项目名称
        name='fire_train_optimized',  # 优化后的训练任务名称
        exist_ok=True,  # 如果存在同名项目则覆盖
        patience=15,  # 早停机制，15轮无改进则停止
        cos_lr=True,  # 使用余弦学习率调度
        augment=True,  # 启用增强策略
        mixup=0.2,  # 混合图像增强
        copy_paste=0.3  # 复制粘贴增强
    )
    
    print("训练完成!")
    print(f"模型保存路径: {results.save_dir}")

if __name__ == '__main__':
    main()
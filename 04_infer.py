from ultralytics import YOLO
import cv2
import os

def main():
    # 加载训练好的模型
    model_path = r'E:\TymRina\课程资料\四阶段\火焰检测\runs\detect\fire_train\weights\best.pt'  # 训练好的最佳模型路径
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本或检查模型路径是否正确")
        return
    
    model = YOLO(model_path)
    
    # 设置推理参数
    conf_threshold = 0.5  # 置信度阈值
    iou_threshold = 0.5  # IoU阈值
    
    # 测试图片路径
    test_image_path = 'e:\\TymRina\\课程资料\\四阶段\\火焰检测\\火焰数据集\\0-扩散火焰---1000---ok\\NPX-GS130UC-Snapshot-20250714-211224-709-2484852850274.PNG'
    
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return
    
    # 执行推理
    results = model(test_image_path, conf=conf_threshold, iou=iou_threshold)
    
    # 显示结果
    for r in results:
        im_array = r.plot()  # 绘制带有预测框的图像
        cv2.imshow('Fire Detection Result', im_array)
        
        # 保存结果图像
        save_path = 'result_fire.jpg'
        cv2.imwrite(save_path, im_array)
        print(f"结果图像已保存到: {save_path}")
    
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口

if __name__ == '__main__':
    main()
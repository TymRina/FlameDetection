import sys
import os
import time
import psutil
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QTextEdit, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class FireDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("火焰检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 摄像头和视频相关
        self.cap = None
        self.is_camera_running = False
        
        # 初始化UI
        self.init_ui()
        
        # 定时器用于更新系统资源信息
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resource_info)
        self.resource_timer.start(1000)  # 每秒更新一次
    
    def load_model(self):
        """加载训练好的模型"""
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建模型文件的绝对路径
        model_path = os.path.join(script_dir, 'runs/detect/fire_train/weights/best.pt')
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.statusBar().showMessage(f"模型加载成功: {model_path}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"模型加载失败: {str(e)}")
                self.model = None
        else:
            QMessageBox.warning(self, "警告", f"模型文件不存在: {model_path}")
            self.model = None
    
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧布局：视频显示区域
        left_layout = QVBoxLayout()
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # 控制按钮布局
        button_layout = QHBoxLayout()
        
        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_btn)
        
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_btn)
        
        self.quit_btn = QPushButton("退出")
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.quit_btn)
        
        left_layout.addLayout(button_layout)
        
        # 右侧布局：结果和系统信息
        right_layout = QVBoxLayout()
        
        # 检测结果显示
        result_label = QLabel("检测结果")
        result_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setFixedHeight(200)
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)
        
        # 系统资源信息
        resource_label = QLabel("系统资源占用")
        resource_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(resource_label)
        
        # CPU使用率
        self.cpu_label = QLabel("CPU使用率: 0%")
        right_layout.addWidget(self.cpu_label)
        
        self.cpu_progress = QProgressBar()
        right_layout.addWidget(self.cpu_progress)
        
        # 内存使用率
        self.memory_label = QLabel("内存使用率: 0%")
        right_layout.addWidget(self.memory_label)
        
        self.memory_progress = QProgressBar()
        right_layout.addWidget(self.memory_progress)
        
        # 空白占位符
        right_layout.addStretch()
        
        # 将左右布局添加到主布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        # 定时器用于处理视频帧
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.process_frame)
    
    def toggle_camera(self):
        """开关摄像头"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先训练并加载模型")
            return
        
        if not self.is_camera_running:
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开摄像头")
                return
            
            self.is_camera_running = True
            self.camera_btn.setText("关闭摄像头")
            self.video_timer.start(30)  # 大约30fps
        else:
            # 关闭摄像头
            self.is_camera_running = False
            self.camera_btn.setText("打开摄像头")
            self.video_timer.stop()
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.video_label.setText("摄像头已关闭")
    
    def upload_image(self):
        """上传图片进行检测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先训练并加载模型")
            return
        
        # 停止摄像头（如果正在运行）
        if self.is_camera_running:
            self.toggle_camera()
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # 读取图片
            image = cv2.imread(file_path)
            
            if image is None:
                QMessageBox.warning(self, "警告", "无法读取图片")
                return
            
            # 执行检测
            results = self.model(image, conf=0.5, iou=0.5)
            
            # 显示检测结果
            self.display_detection_results(results[0], image)
    
    def process_frame(self):
        """处理视频帧"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # 执行检测
        results = self.model(frame, conf=0.5, iou=0.5)
        
        # 显示检测结果
        self.display_detection_results(results[0], frame)
    
    def display_detection_results(self, result, frame):
        """显示检测结果"""
        # 绘制预测框
        annotated_frame = result.plot()
        
        # 将OpenCV图像转换为PyQt图像
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            annotated_frame.data, width, height, bytes_per_line,
            QImage.Format_BGR888
        )
        
        # 显示图像
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # 更新检测结果文本
        result_text = f"检测到 {len(result.boxes)} 个火焰目标\n"
        result_text += "-" * 30 + "\n"
        
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf = float(box.conf[0])
            
            result_text += f"目标 {i+1}: {class_name}\n"
            result_text += f"  置信度: {conf:.2f}\n"
        
        self.result_text.setPlainText(result_text)
    
    def update_resource_info(self):
        """更新系统资源信息"""
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent()
        self.cpu_label.setText(f"CPU使用率: {cpu_percent:.1f}%")
        self.cpu_progress.setValue(int(cpu_percent))
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_label.setText(f"内存使用率: {memory_percent:.1f}%")
        self.memory_progress.setValue(int(memory_percent))
    
    def closeEvent(self, event):
        """关闭窗口时的事件处理"""
        # 释放摄像头资源
        if self.cap is not None:
            self.cap.release()
        
        # 停止定时器
        self.video_timer.stop()
        self.resource_timer.stop()
        
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec_())
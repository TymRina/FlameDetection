import os
import shutil
import random
import yaml

# 数据集根目录
dataset_root = "e:\\TymRina\\课程资料\\四阶段\\火焰检测\\火焰数据集"
# 输出目录
target_root = "e:\\TymRina\\课程资料\\四阶段\\day14\\fire_detection\\dataset"

# 类别映射
class_map = {
    "0-扩散火焰---1000---ok": 0,
    "1-预混火焰------1000---ok": 1,
    "2-不完全预混-过渡---------1000---ok": 2,
    "3-异常火焰-熄灭-------1000---ok": 3
}

# 创建目标目录结构
for split in ["train", "test", "valid"]:
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(target_root, split, folder), exist_ok=True)

# 收集所有图片和标签文件
data_files = []
for class_name, class_id in class_map.items():
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.exists(class_dir):
        continue
    
    # 获取所有图片文件
    for file in os.listdir(class_dir):
        if file.endswith(".PNG") or file.endswith(".jpg") or file.endswith(".jpeg"):
            img_path = os.path.join(class_dir, file)
            # 查找对应的标签文件
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(class_dir, label_file)
            
            if os.path.exists(label_path):
                data_files.append((img_path, label_path, class_id))

# 随机打乱数据
random.seed(42)
random.shuffle(data_files)

# 划分数据集
total = len(data_files)
train_split = int(total * 0.7)
test_split = int(total * 0.2)
valid_split = total - train_split - test_split

train_files = data_files[:train_split]
test_files = data_files[train_split:train_split+test_split]
valid_files = data_files[train_split+test_split:]

# 复制文件到目标目录
def copy_files(files, split):
    for img_path, label_path, class_id in files:
        # 复制图片
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(target_root, split, "images", img_name))
        
        # 修改标签文件中的类别ID并复制
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 将原类别ID替换为新的类别ID
                    parts[0] = str(class_id)
                    new_line = " ".join(parts) + "\n"
                    new_lines.append(new_line)
        
        label_name = os.path.basename(label_path)
        with open(os.path.join(target_root, split, "labels", label_name), 'w') as f:
            f.writelines(new_lines)

# 复制文件
print("Copying train files...")
copy_files(train_files, "train")
print("Copying test files...")
copy_files(test_files, "test")
print("Copying valid files...")
copy_files(valid_files, "valid")

# 创建data.yaml文件
data_config = {
    "path": "e:\\TymRina\\课程资料\\四阶段\\day14\\fire_detection\\dataset",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": [
        "扩散火焰",
        "预混火焰",
        "不完全预混火焰",
        "异常火焰"
    ]
}

with open(os.path.join(target_root, "data.yaml"), 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, allow_unicode=True)

print("Data preprocessing completed!")
print(f"Total files: {total}")
print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")
print(f"Valid files: {len(valid_files)}")
print(f"data.yaml created at: {os.path.join(target_root, 'data.yaml')}")
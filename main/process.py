# 所有原图在data/img文件夹内

# 所有标注在data/img/labels文件夹内
# 所有标注的文件名同图片文件，后缀为.txt

# 首先删除所有data/img中没有被打标注的图片
# 按照8:2的比例，将data/img中的图片分为train和val两部分
# 按照8:2的比例，将data/img/labels中的标注分为train和val两部分
# 生成data.yaml文件
# 生成data.yaml文件的格式参考入如下：
# train: ../data/dataset/images/train  # Adjusted path relative to data.yaml location
# val: ../data/dataset/images/val    # Adjusted path relative to data.yaml location
# nc: 4
# names: ['nanzhu', 'nvzhu', 'nanpei', 'nvpei']
# 注意：原始注释中labels指向一个目录，但通常labels也需要划分train/val。
# 本脚本将labels也划分，并在yaml中省略labels根目录。YOLO会自动查找。

import os
import shutil
import random
import yaml
from pathlib import Path

def process_dataset(img_dir_rel='../data/img', label_dir_rel='../data/img/labels', output_dir_rel='../data/dataset', train_ratio=0.8):
    """
    处理图像数据集，清理、划分并生成配置文件。

    Args:
        img_dir_rel (str): 相对于脚本的原始图片目录路径。
        label_dir_rel (str): 相对于脚本的原始标签目录路径。
        output_dir_rel (str): 相对于脚本的输出数据集目录路径。
        train_ratio (float): 训练集所占比例。
    """
    script_dir = Path(__file__).parent
    img_dir = (script_dir / img_dir_rel).resolve()
    label_dir = (script_dir / label_dir_rel).resolve()
    output_dir = (script_dir / output_dir_rel).resolve()
    data_yaml_path = output_dir.parent / 'data.yaml' # Place data.yaml in ../data/

    print(f"图片目录: {img_dir}")
    print(f"标签目录: {label_dir}")
    print(f"输出目录: {output_dir}")

    if not img_dir.exists():
        print(f"错误: 图片目录 {img_dir} 不存在。")
        return
    if not label_dir.exists():
        print(f"错误: 标签目录 {label_dir} 不存在。")
        return

    # 1. 清理没有对应标签的图片
    print("开始清理没有标签的图片...")
    img_files = [f for f in img_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
    label_files_basenames = {f.stem for f in label_dir.glob('*.txt') if f.is_file()}
    cleaned_count = 0
    valid_img_files = []

    for img_file in img_files:
        if img_file.stem in label_files_basenames:
            valid_img_files.append(img_file)
        else:
            print(f"删除图片 (无对应标签): {img_file}")
            img_file.unlink()
            cleaned_count += 1
    print(f"清理完成，共删除 {cleaned_count} 张图片。剩余 {len(valid_img_files)} 张有效图片。")

    if not valid_img_files:
        print("错误：没有有效的图片和标签对可供处理。")
        return

    # 2. 划分数据集
    print("开始划分数据集...")
    random.shuffle(valid_img_files)
    num_train = int(len(valid_img_files) * train_ratio)
    train_files = valid_img_files[:num_train]
    val_files = valid_img_files[num_train:]
    print(f"划分完成：训练集 {len(train_files)} 张，验证集 {len(val_files)} 张。")

    # 3. 创建输出目录
    output_img_train_dir = output_dir / 'images' / 'train'
    output_img_val_dir = output_dir / 'images' / 'val'
    output_label_train_dir = output_dir / 'labels' / 'train'
    output_label_val_dir = output_dir / 'labels' / 'val'

    # 清空已存在的输出目录内容
    print("清空旧的输出目录...")
    for dir_path in [output_img_train_dir, output_img_val_dir, output_label_train_dir, output_label_val_dir]:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"已删除: {dir_path}")
            except OSError as e:
                print(f"删除目录时出错 {dir_path}: {e}")
        # 重新创建目录
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             print(f"创建目录时出错 {dir_path}: {e}")
             return # 如果无法创建目录，则停止处理

    print("输出目录创建/清空完成。")


    # 4. 复制文件 (原为移动文件)
    print("开始复制文件...")
    def copy_files(file_list, target_img_dir, target_label_dir): # 函数名改为 copy_files
        copied_count = 0 # 变量名改为 copied_count
        for img_path in file_list:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                try:
                    # 使用 shutil.copy2 复制图片和标签文件
                    shutil.copy2(str(img_path), str(target_img_dir / img_path.name))
                    shutil.copy2(str(label_path), str(target_label_dir / label_path.name))
                    copied_count += 1
                except Exception as e:
                    # 错误信息更新
                    print(f"复制文件时出错 {img_path} 或 {label_path}: {e}")
            else:
                 print(f"警告：找不到标签文件 {label_path} 对应图片 {img_path}")
        return copied_count # 返回复制的数量

    # 调用修改后的函数
    copied_train = copy_files(train_files, output_img_train_dir, output_label_train_dir)
    copied_val = copy_files(val_files, output_img_val_dir, output_label_val_dir)
    # 更新打印信息
    print(f"文件复制完成：复制了 {copied_train} 对训练文件，{copied_val} 对验证文件。")
    print(f"原始目录 {img_dir} 和 {label_dir} 中的文件保持不变。")


    # 5. 生成 data.yaml 文件
    print(f"开始生成 {data_yaml_path}...")
    # Calculate relative paths from the location of data.yaml
    train_img_rel_path = os.path.relpath(output_img_train_dir, data_yaml_path.parent)
    val_img_rel_path = os.path.relpath(output_img_val_dir, data_yaml_path.parent)

    data_yaml_content = {
        'path': os.path.relpath(output_dir, data_yaml_path.parent), # Path relative to data.yaml
        'train': train_img_rel_path,
        'val': val_img_rel_path,
        'nc': 4,
        'names': ['nanzhu', 'nvzhu', 'nanpei', 'nvpei']
    }

    try:
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml_content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"{data_yaml_path} 生成成功。")
    except Exception as e:
        print(f"生成 {data_yaml_path} 时出错: {e}")

if __name__ == '__main__':
    # --- 配置参数 ---
    IMG_DIR_RELATIVE = '../data/img'          # 原始图片目录（相对于本脚本）
    LABEL_DIR_RELATIVE = '../data/img/labels' # 原始标签目录（相对于本脚本）
    OUTPUT_DIR_RELATIVE = '../data/dataset'   # 输出数据集目录（相对于本脚本）
    TRAIN_RATIO = 0.8                         # 训练集比例
    # --- 配置结束 ---

    process_dataset(
        img_dir_rel=IMG_DIR_RELATIVE,
        label_dir_rel=LABEL_DIR_RELATIVE,
        output_dir_rel=OUTPUT_DIR_RELATIVE,
        train_ratio=TRAIN_RATIO
    )
    print("处理完成。")
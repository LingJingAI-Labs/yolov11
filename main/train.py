import sys
import site
import torch
from ultralytics import YOLO
import os
from config import PROJECT_ROOT, DATA_YAML_PATH, BASE_MODEL, RUNS_DIR, TRAIN_PROJECT_NAME, TRAIN_RUN_NAME

def print_env_info():
    print(f"Python路径: {sys.executable}")
    print(f"site-packages路径: {site.getsitepackages()}")
    print(torch.__version__)
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else '不可用'}")
    print(f"GPU设备数: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")

def train_yolo_model():
    # --- 配置参数 ---
    project_root = str(PROJECT_ROOT)
    dataset_yaml_path = str(DATA_YAML_PATH)

    # --- 训练模式选择 ---
    RESUME_TRAINING = False  # 设置为 True 以继续上次训练，设置为 False 以开始新训练

    # --- 基础模型与继续训练配置 ---
    # 正确指定YOLO11m模型
    base_model_name = "yolo11m.pt"  # 使用标准命名格式
    model_path = os.path.join('models', base_model_name)
    
    # 检查本地是否存在模型文件
    if not os.path.exists(model_path):
        print(f"注意：本地未找到模型文件 {model_path}，将尝试从官方仓库下载。")
        # 直接使用模型名称，让Ultralytics自动下载
        model_to_load = base_model_name
    else:
        model_to_load = model_path
        
    previous_project_name = os.path.join('runs', 'detect', TRAIN_PROJECT_NAME)
    previous_run_name = TRAIN_RUN_NAME

    # --- 训练参数 ---
    total_epochs = 100      # 增加轮数以充分学习
    img_size = 640         # 统一使用640尺寸提高训练稳定性
    batch_size = 4         # 批处理大小，根据RTX 3060显存调整
    
    # 结果保存路径配置
    if RESUME_TRAINING:
        project_name = previous_project_name
        run_name = previous_run_name
        model_to_load = os.path.join(project_root, previous_project_name, previous_run_name, 'weights', 'last.pt')
        print(f"模式：继续训练。将加载权重: {model_to_load}")
        if not os.path.exists(model_to_load):
            print(f"错误：找不到用于继续训练的权重文件 '{model_to_load}'。")
            print("请检查路径或将 RESUME_TRAINING 设置为 False。")
            return
    else:
        project_name = os.path.join('runs', 'detect', TRAIN_PROJECT_NAME)
        run_name = 'yolo11m_exp'  # 更改运行名以反映模型类型
        print(f"模式：开始新训练。将加载基础模型: {model_to_load}")

    # --- 设备检查 (优先使用 CUDA，因为我们有RTX 3060) ---
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"将使用的设备: {device}")

    # --- 加载模型 ---
    try:
        model = YOLO(model_to_load)
        print(f"成功加载模型: {model_to_load}")
        
    except Exception as e:
        print(f"加载模型 '{model_to_load}' 时出错: {e}")
        print("请确保模型文件路径正确或模型名称可被自动下载。")
        return

    # --- 开始训练 ---
    print(f"开始训练模型，使用数据集配置文件: {dataset_yaml_path}")
    if not os.path.exists(dataset_yaml_path):
        print(f"错误：找不到数据集配置文件 '{dataset_yaml_path}'。")
        print("请确认 process.py 已成功运行并在 data/ 目录下生成了 data.yaml。")
        return

    try:
        # --- 第一阶段：冻结backbone训练 ---
        print("第一阶段：冻结主干网络层进行训练...")
        results = model.train(
            data=dataset_yaml_path,
            epochs=30,              # 第一阶段训练30轮
            imgsz=img_size,
            batch=batch_size,
            project=project_name,
            name=run_name,
            device=device,
            resume=False,
            
            # 冻结参数 - 冻结backbone
            freeze=[0, 1, 2, 3, 4, 5],  # YOLO11模型结构可能不同，调整冻结层
            
            # 优化器和学习率参数
            optimizer='AdamW',      # 优化器选择
            lr0=0.0005,             # 初始学习率
            lrf=0.01,               # 最终学习率因子
            momentum=0.937,         # 动量
            weight_decay=0.001,     # 增加权重衰减
            warmup_epochs=5,        # 增加热身轮数
            warmup_momentum=0.8,    # 热身动量
            cos_lr=True,            # 使用余弦学习率调度
            
            # 数据增强参数，增强各项参数
            augment=True,           # 启用数据增强
            degrees=15.0,           # 增加旋转角度范围
            translate=0.3,          # 增加平移比例
            scale=0.3,              # 增加缩放比例
            shear=10.0,             # 增加剪切角度
            fliplr=0.5,             # 水平翻转概率
            flipud=0.2,             # 垂直翻转概率
            mosaic=0.7,             # 增加马赛克增强概率
            mixup=0.3,              # 增加混合增强概率
            copy_paste=0.3,         # 添加复制粘贴增强
            auto_augment='randaugment', # 添加自动增强
            erasing=0.4,            # 随机擦除增强
            
            # 训练策略参数
            verbose=True,           # 详细输出
            patience=30,            # 增加早停耐心值，避免过早停止
            
            # 额外参数
            exist_ok=True,          # 如果输出目录已存在则覆盖
            pretrained=True,        # 使用预训练权重
            dropout=0.2,            # 增加dropout以减少过拟合
            val=True,               # 训练时进行验证
            rect=False,             # 关闭矩形训练，提高稳定性
            overlap_mask=True,      # 重叠的掩码
            mask_ratio=4,           # 掩码下采样比例
            
            # 类别平衡参数
            nbs=64,                 # 标称批量大小
            cls=1.0,                # 增加分类损失权重
            box=8.0,                # 增加边界框损失权重
        )
        
        # --- 第二阶段：解冻全部层微调 ---
        print("第二阶段：解冻所有层进行微调...")
        # 加载第一阶段训练的最佳模型
        best_model_path = os.path.join(project_name, run_name, 'weights', 'best.pt')
        if not os.path.exists(best_model_path):
            print(f"无法找到第一阶段训练的最佳模型: {best_model_path}")
            best_model_path = os.path.join(project_name, run_name, 'weights', 'last.pt')
            
        # 加载第一阶段训练的模型
        second_stage_model = YOLO(best_model_path)
            
        # 开始第二阶段训练 - 注意这里resume=False
        results = second_stage_model.train(
            data=dataset_yaml_path,
            epochs=total_epochs,
            imgsz=img_size,
            batch=batch_size,
            project=project_name,
            name=f"{run_name}_stage2",
            device=device,
            resume=False,   
            
            # 优化器和学习率参数
            optimizer='AdamW',
            lr0=0.0001,  # 较低的学习率
            lrf=0.005,
            momentum=0.937,
            weight_decay=0.0005,
            
            # 训练策略参数
            patience=50,
            verbose=True,
            exist_ok=True
        )
        
        print("训练完成！")
        save_dir = os.path.join(project_root, project_name, run_name)
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        print(f"训练结果保存在: {save_dir}")
        if os.path.exists(best_model_path):
            print(f"最佳模型权重保存在: {best_model_path}")
        else:
            print("警告：未找到最佳模型权重 best.pt。请检查训练日志。")
    except FileNotFoundError:
        print(f"错误：找不到数据集配置文件 '{dataset_yaml_path}'。")
        print("请确保该文件存在并且路径正确。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    print_env_info()
    train_yolo_model()
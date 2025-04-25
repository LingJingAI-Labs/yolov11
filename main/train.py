import torch
from ultralytics import YOLO
import os

def train_yolo_model():
    # --- 配置参数 ---
    project_root = '/Users/snychng/Work/code/yolov11'
    # 确认 data.yaml 的路径 (应在 data/ 目录下)
    dataset_yaml_path = os.path.join(project_root, 'data', 'dataset', 'data.yaml')

    # --- 训练模式选择 ---
    RESUME_TRAINING = True  # 设置为 True 以继续上次训练，设置为 False 以开始新训练

    # --- 基础模型与继续训练配置 ---
    base_model_name = 'models/yolo11n.pt' # 或者 'yolov8n.pt' 让库自动下载
    previous_project_name = 'runs/detect/train-0424' # *仅在 RESUME_TRAINING = True 时使用*：上次训练的项目名
    previous_run_name = 'exp3'                     # *仅在 RESUME_TRAINING = True 时使用*：上次训练的运行名

    # --- 训练参数 ---
    total_epochs = 10      # 目标总训练轮数 (无论是新训练还是继续训练)
    img_size = 640         # 输入图像大小
    batch_size = 8         # 批处理大小
    # 结果保存路径配置
    if RESUME_TRAINING:
        project_name = previous_project_name # 继续训练时，通常保存在同一项目下
        run_name = previous_run_name         # 继续训练时，通常保存在同一运行下
        model_to_load = os.path.join(project_root, previous_project_name, previous_run_name, 'weights', 'last.pt')
        print(f"模式：继续训练。将加载权重: {model_to_load}")
        if not os.path.exists(model_to_load):
            print(f"错误：找不到用于继续训练的权重文件 '{model_to_load}'。")
            print("请检查路径或将 RESUME_TRAINING 设置为 False。")
            return
    else:
        project_name = 'runs/detect/train-0424' # 新训练的项目名 (可以修改)
        run_name = 'exp'                     # 新训练的运行名 (会自动递增, 如 exp, exp2, ...)
        model_to_load = base_model_name
        print(f"模式：开始新训练。将加载基础模型: {model_to_load}")
        # 可选：检查基础模型文件是否存在（如果不是让库自动下载的话）
        # if not os.path.exists(os.path.join(project_root, model_to_load)) and model_to_load != 'yolov8n.pt': # 假设 'yolov8n.pt' 是自动下载的
        #     print(f"警告：基础模型文件 '{model_to_load}' 在指定路径未找到。")
        #     print("如果不是自动下载的模型，请确保文件存在。")


    # --- 设备检查 (优先使用 MPS) ---
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"将使用的设备: {device}")

    # --- 加载模型 ---
    try:
        # 根据选择的模式加载模型
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
        # 调用 train
        results = model.train(
            data=dataset_yaml_path,
            epochs=total_epochs, # 使用总轮数
            imgsz=img_size,
            batch=batch_size,
            project=project_name,
            name=run_name,
            device=device,
            resume=False # 加载 .pt 文件通常已包含状态，无需显式 resume=True
            # patience=10,
            # workers=8,
        )
        print("训练完成！")
        # 结果保存路径现在由 project_name 和 run_name 决定
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
    train_yolo_model()
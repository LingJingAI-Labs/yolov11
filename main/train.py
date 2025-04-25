import torch
from ultralytics import YOLO
import os
from config import PROJECT_ROOT, DATA_YAML_PATH, BASE_MODEL, RUNS_DIR, TRAIN_PROJECT_NAME, TRAIN_RUN_NAME

def train_yolo_model():
    # --- 配置参数 ---
    project_root = str(PROJECT_ROOT)
    dataset_yaml_path = str(DATA_YAML_PATH)

    # --- 训练模式选择 ---
    RESUME_TRAINING = True  # 设置为 True 以继续上次训练，设置为 False 以开始新训练

    # --- 基础模型与继续训练配置 ---
    base_model_name = str(BASE_MODEL)
    previous_project_name = os.path.join('runs', 'detect', TRAIN_PROJECT_NAME)
    previous_run_name = TRAIN_RUN_NAME

    # --- 训练参数 ---
    total_epochs = 10      # 目标总训练轮数 (无论是新训练还是继续训练)
    img_size = 640         # 输入图像大小
    batch_size = 8         # 批处理大小
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
        run_name = 'exp'
        model_to_load = base_model_name
        print(f"模式：开始新训练。将加载基础模型: {model_to_load}")
        # 可选：检查基础模型文件是否存在（如果不是让库自动下载的话）
        # if not os.path.exists(os.path.join(project_root, model_to_load)) and model_to_load != 'yolov8n.pt':
        #     print(f"警告：基础模型文件 '{model_to_load}' 在指定路径未找到。")
        #     print("如果不是自动下载的模型，请确保文件存在。")

    # --- 设备检查 (优先使用 MPS) ---
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
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
        results = model.train(
            data=dataset_yaml_path,
            epochs=total_epochs,
            imgsz=img_size,
            batch=batch_size,
            project=project_name,
            name=run_name,
            device=device,
            resume=False
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
    train_yolo_model()
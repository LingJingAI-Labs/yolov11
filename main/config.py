import os
from pathlib import Path

# 项目根目录（假设config.py在main目录下）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据集相关路径
DATASET_DIR = os.environ.get('YOLO_DATASET_DIR', PROJECT_ROOT / 'data' / 'dataset')
DATA_YAML_PATH = os.environ.get('YOLO_DATA_YAML', DATASET_DIR / 'data.yaml')

# 原始图片和标签路径
IMG_DIR = os.environ.get('YOLO_IMG_DIR', PROJECT_ROOT / 'data' / 'img')
LABEL_DIR = os.environ.get('YOLO_LABEL_DIR', PROJECT_ROOT / 'data' / 'img' / 'labels')

# 训练输出路径
RUNS_DIR = os.environ.get('YOLO_RUNS_DIR', PROJECT_ROOT / 'runs' / 'detect')

# 模型路径
BASE_MODEL = os.environ.get('YOLO_BASE_MODEL', PROJECT_ROOT / 'models' / 'yolo11x.pt')

# 其它参数
TRAIN_PROJECT_NAME = os.environ.get('YOLO_TRAIN_PROJECT', 'train-0424')
TRAIN_RUN_NAME = os.environ.get('YOLO_TRAIN_RUN', 'exp3')
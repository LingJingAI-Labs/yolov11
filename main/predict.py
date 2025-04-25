import os
import cv2
import torch
from ultralytics import YOLO
from config import PROJECT_ROOT, RUNS_DIR, TRAIN_PROJECT_NAME, TRAIN_RUN_NAME

def video_inference_with_custom_model():

    # 训练好的模型权重路径
    model_weights_path = os.path.join(RUNS_DIR, TRAIN_PROJECT_NAME, TRAIN_RUN_NAME, 'weights', 'best.pt')

    # 需要进行预测的视频源路径（可考虑也放到 config.py 统一管理）
    video_path = os.path.join(PROJECT_ROOT, 'data', 'video', 'test-05.mp4')

    # 置信度阈值
    confidence_threshold = 0.25
    
    # 设置显示窗口的最大宽度
    display_width = 1024  # 可根据你的屏幕大小调整这个值

    # --- 设备检查 ---
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"将使用的设备: {device}")

    # --- 加载训练好的模型 ---
    if not os.path.exists(model_weights_path):
        print(f"错误：找不到模型权重文件 '{model_weights_path}'。")
        print("请确保路径正确，并且模型已经训练完成。")
        return

    try:
        model = YOLO(model_weights_path)
        print(f"成功加载训练好的模型: {model_weights_path}")
        # 将模型移动到指定的设备
        model.to(device)
        print(f"模型已转移到设备: {device}")
    except Exception as e:
        print(f"加载模型或将其转移到设备时出错: {e}")
        return

    # --- 打开视频文件 ---
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        print("请检查文件路径和格式是否正确。")
        return

    print(f"成功打开视频: {video_path}")
    print(f"开始逐帧推理。按 'q' 键退出。")

    # --- 逐帧处理视频 ---
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频结束或读取错误
        if not ret:
            print("视频处理完成或读取帧时出错。")
            break

        frame_count += 1

        # 使用 YOLO 模型进行推理
        results = model(frame, conf=confidence_threshold, verbose=True)

        # 处理推理结果
        annotated_frame = frame
        for result in results:
            annotated_frame = result.plot()

        # 调整显示帧的大小
        frame_height, frame_width = annotated_frame.shape[:2]
        # 计算调整后的高度以保持宽高比
        display_height = int(frame_height * (display_width / frame_width))
        # 调整帧大小
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # 显示调整大小后的帧
        cv2.imshow('YOLOv11 Video Inference (Custom Model)', resized_frame)

        # 检测按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户请求退出。")
            break

    # --- 释放资源 ---
    cap.release()
    cv2.destroyAllWindows()

    print("资源已释放。推理结束。")

if __name__ == '__main__':
    video_inference_with_custom_model()
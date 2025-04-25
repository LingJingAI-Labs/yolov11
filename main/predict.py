import os
import cv2
import torch
from ultralytics import YOLO

def video_inference_with_custom_model():

    # 训练好的模型权重路径
    project_root = '/Users/snychng/Work/code/yolov11'
    project_name = 'train-0424'                       
    run_name = 'exp34'                              
    model_weights_path = os.path.join(project_root, 'runs', 'detect', project_name, run_name, 'weights', 'best.pt')

    # 需要进行预测的视频源路径
    video_path = '/Users/snychng/Work/code/yolov11/data/video/test-05.mp4'

    # 置信度阈值
    confidence_threshold = 0.25

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
        # print(f"正在处理帧: {frame_count}") # 可选：打印帧数显示进度

        # 使用 YOLO 模型进行推理 (基于代码2的模型调用方式，并应用置信度阈值)
        # model(frame, ...) 是 model.predict(frame, ...) 的简写形式，适用于单输入
        # conf=confidence_threshold 应用了阈值
        # verbose=False 可以减少控制台输出，避免每帧都打印日志
        results = model(frame, conf=confidence_threshold, verbose=True)

        # 处理推理结果
        # results 是一个列表，通常包含一个 Results 对象对于单帧输入
        annotated_frame = frame # 初始化带标注的帧为原始帧
        for result in results:
            # result.plot() 会在当前帧上绘制检测框和标签
            annotated_frame = result.plot() # 获取绘制了检测结果的帧

            # 您可以在这里进一步处理 result 对象，例如打印框坐标、类别、置信度等
            # for box in result.boxes:
            #     print(f"  Detected: Class={box.cls.item()}, Conf={box.conf.item():.2f}, XYXY={box.xyxy.squeeze().tolist()}")


        # 显示处理后的帧
        cv2.imshow('YOLOv11 Video Inference (Custom Model)', annotated_frame)

        # 检测按键，如果按下 'q' 键则退出循环
        # cv2.waitKey(1) 表示等待键盘输入 1 毫秒
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户请求退出。")
            break

    # --- 释放资源 (基于代码2的风格) ---
    cap.release()          # 释放视频捕捉对象
    cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口

    print("资源已释放。推理结束。")

if __name__ == '__main__':
    video_inference_with_custom_model()
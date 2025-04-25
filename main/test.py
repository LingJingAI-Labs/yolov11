import cv2
import torch
from ultralytics import YOLO

# 检查 MPS 是否可用
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    device = 'cpu' # 如果 MPS 不可用，回退到 CPU
else:
    device = 'mps'
print(f"将使用的设备: {device}")

try:
    # 在这里添加 device=device 参数
    model = YOLO('models/yolo11s.pt', task='detect')
    model.to(device) # 确保模型移动到指定设备
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("请确保 YOLO 模型文件 ('models/yolo11m.pt') 存在。")
    exit()

# 视频文件路径
video_path = '/Users/snychng/Work/code/yolov11/data/video/test-05.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

print("按 'q' 键退出实时推理...")

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果 ret 为 False，表示视频结束或读取错误
    if not ret:
        print("视频处理完成或读取帧时出错。")
        break

    # 使用 YOLOv11 模型进行推理
    # 'stream=True' 适用于视频流处理，可能更高效
    # 'verbose=False' 可以减少控制台输出
    # ultralytics 会自动使用模型所在的设备 (model.device)
    results = model(frame, stream=True, verbose=True)

    # 处理推理结果
    for result in results:
        # result.plot() 会在原图上绘制检测框和标签
        # 如果您的库不支持 plot()，您需要手动绘制
        # 例如，遍历 result.boxes 获取边界框坐标和类别
        annotated_frame = result.plot() # 获取绘制了检测结果的帧

        # 显示处理后的帧
        cv2.imshow('YOLOv11 Real-Time Inference (MPS)', annotated_frame)

    # 检测按键，如果按下 'q' 键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("用户请求退出。")
        break

# 释放视频捕捉对象
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()

print("资源已释放。")
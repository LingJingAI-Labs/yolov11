import cv2
import os
import math

def extract_frames(video_path, output_folder, interval_sec=1):
    """
    从视频中每隔指定秒数提取一帧并保存。

    Args:
        video_path (str): 输入视频文件的路径。
        output_folder (str): 保存提取帧的文件夹路径。
        interval_sec (int): 提取帧的时间间隔（秒）。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    # 获取视频的帧率 (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("错误：无法获取视频的有效帧率。")
        cap.release()
        return

    print(f"视频信息: 路径='{video_path}', FPS={fps:.2f}")

    # 计算需要跳过的帧数
    frame_interval = int(round(fps * interval_sec))
    if frame_interval <= 0:
        print("错误：计算出的帧间隔小于或等于 0，请检查视频 FPS 或间隔设置。")
        frame_interval = 1 # 至少处理每一帧

    print(f"将每隔 {frame_interval} 帧提取一帧 (约等于 {interval_sec} 秒)。")

    frame_count = 0
    saved_frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频结束或读取错误
        if not ret:
            print("视频处理完成或读取帧时出错。")
            break

        # 检查是否到达提取帧的时间点
        # (frame_count % frame_interval == 0) 确保我们从第0帧开始，并按间隔提取
        if frame_count % frame_interval == 0:
            # 构建输出文件名 (例如: frame_0000.jpg, frame_0001.jpg, ...)
            output_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            # 保存帧
            cv2.imwrite(output_filename, frame)
            print(f"已保存: {output_filename} (原视频第 {frame_count} 帧)")
            saved_frame_count += 1

        frame_count += 1

    # 释放视频捕捉对象
    cap.release()
    print(f"总共处理了 {frame_count} 帧，提取并保存了 {saved_frame_count} 帧到 {output_folder}")

if __name__ == '__main__':
    # --- 配置 ---
    project_root = '/Users/snychng/Work/code/yolov11' # 你的项目根目录
    input_video = os.path.join(project_root, 'data', 'video', 'test-05.mp4')
    output_dir = os.path.join(project_root, 'data', 'img')
    extract_interval_seconds = 1 # 每隔 1 秒提取一帧
    # --- 执行 ---
    extract_frames(input_video, output_dir, extract_interval_seconds)
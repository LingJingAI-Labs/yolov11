import torch
import numpy as np
import sys
from ultralytics import YOLO

print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print(f"NumPy版本: {np.__version__}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    # 测试GPU计算
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    z = torch.matmul(x, y)
    end.record()
    torch.cuda.synchronize()
    print(f"矩阵乘法耗时: {start.elapsed_time(end):.2f} ms")
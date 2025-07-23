# Reference:
# Ultralytics YOLO Docs - https://docs.ultralytics.com/reference/engine/model/

import sys
import traceback

import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

# region 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）
context = rs.context()
if len(context.query_devices()) == 0:
    print("未檢測到已連接的深度攝影機！")
    sys.exit(1)

print(f"PyTorch 是否可用 cuda：{torch.cuda.is_available()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Main GPU: {torch.cuda.get_device_name(0)}")
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）

try:
    # 初始化 YOLOv8 模型
    # 名稱後面有個n代表這是 Nano 模型，也是 YOLOv8 裡面最輕量、運行速度最快的模型（運行時，如果指定的路徑找不到該模型檔案，會自動從網路下載）
    model: YOLO = YOLO("models/yolov8n.pt")

    rs_config: rs.config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline: rs.pipeline = rs.pipeline()
    pipeline.start(rs_config)

    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image: np.ndarray = np.asanyarray(color_frame.get_data())

        # 使用 YOLO 模型進行物件偵測
        # conf=0.25 代表置信度閾值，低於這個值的預測結果將被忽略
        model(color_image, conf=0.25)
except Exception as e:
    traceback.print_exc()

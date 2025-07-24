import sys
import traceback
from typing import Generator

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult

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
    # 使用 SAHI 初始化 YOLOv8 模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="models/yolov8n.pt",
        confidence_threshold=0.25,
        # 這裡的 device 會自動選擇可用的 GPU，如果沒有可用的 GPU，則使用 CPU
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    rng: Generator = np.random.default_rng(1)
    colors: np.ndarray = rng.integers(
        0, 255, size=(len(detection_model.category_mapping), 3), dtype=np.int32
    )

    rs_config: rs.config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline: rs.pipeline = rs.pipeline()
    pipeline.start(rs_config)

    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        depth_frame: rs.depth_frame = frames.get_depth_frame()
        color_frame: rs.video_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image: np.ndarray = np.asanyarray(color_frame.get_data())
        # 用於標註的影像
        annotated_image: np.ndarray = color_image.copy()

        # 使用 SAHI 來進行物件偵測，這裡會將影像切割成小塊進行處理
        # 這裡的切割大小是 320x320，重疊比例是 20%，可以根據需要調整
        # verbose=False 代表不輸出詳細的處理過程
        result: PredictionResult = get_sliced_prediction(
            color_image,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=False,
        )

        for pred in result.object_prediction_list:
            class_id: int = pred.category.id  # 類別 ID
            score: float = pred.score.value  # 置信度
            bbox: list[int] = list(map(int, pred.bbox.to_xyxy()))  # 邊界框
            label: str = pred.category.name  # 類別名稱
            # 隨機顏色
            color: list[int] = colors[class_id].astype(int).tolist()

            x1, y1, x2, y2 = bbox
            annotated_image = cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), color, 2
            )

            depth: float = depth_frame.get_distance(
                int((x1 + x2) / 2), int((y1 + y2) / 2)
            )

            annotated_image = cv2.putText(
                annotated_image,
                f"{label} {round(score * 100)}% {depth:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.namedWindow("small object detect", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("small object detect", annotated_image)

        key: int = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("small object detect", cv2.WND_PROP_VISIBLE) < 1:
            break

    pipeline.stop()
    cv2.destroyAllWindows()
except Exception as e:
    traceback.print_exc()

import sys
import traceback
from typing import List

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

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
    # 名稱後面有個n代表這是 Nano 模型，也是 YOLOv8 裡面最輕量、運行速度最快的模型（運行時，如果指定的路徑找不到該模型檔案，會自動從網路下載）
    model: YOLO = YOLO("models/yolov8n.pt")

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline = rs.pipeline()
    pipeline.start(rs_config)

    box_annotator: sv.BoxAnnotator = sv.BoxAnnotator()
    trace_annotator: sv.TraceAnnotator = sv.TraceAnnotator(trace_length=10, thickness=2)

    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        depth_frame: rs.depth_frame = frames.get_depth_frame()
        color_frame: rs.video_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image: np.ndarray = np.asanyarray(color_frame.get_data())
        # 用於標註的影像
        annotated_image: np.ndarray = color_image.copy()

        # 使用 YOLOv8 模型進行物件追蹤
        # tracker="botsort.yaml" 代表使用 YOLOv8 內建的 BotSort 追蹤器
        results: List[Results] = model.track(
            color_image,
            conf=0.25,
            iou=0.5,
            agnostic_nms=True,
            verbose=False,
            persist=True,
            tracker="botsort.yaml",
        )
        result: Results = results[0]

        detections = sv.Detections.from_ultralytics(result)

        # 檢查 detections.tracker_id 是否為 None，如果是則將其設置為一個空陣列
        if detections.tracker_id is None:
            detections.tracker_id = np.array([])

        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = trace_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        cv2.namedWindow("object track", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("object track", annotated_image)

        key: int = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("object track", cv2.WND_PROP_VISIBLE) < 1:
            break

    pipeline.stop()
    cv2.destroyAllWindows()
except Exception as e:
    traceback.print_exc()

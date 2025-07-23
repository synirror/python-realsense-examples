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


def get_median_depth(
    depth_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, samples=30
) -> float:
    """
    這個函數用來從邊界框中獲取指定區域的深度距離，並計算中間值
    :param depth_frame: 深度影像的 rs.depth_frame 物件
    :param x1: 邊界框左上角的 x 座標
    :param y1: 邊界框左上角的 y 座標
    :param x2: 邊界框右下角的 x 座標
    :param y2: 邊界框右下角的 y 座標
    :param samples: 用來計算中間值的樣本數量，預設為 30
    :return: 返回指定區域的深度距離中間值，單位為公尺。如果沒有有效的深度數據則返回 0.0
    """
    step_x = max(1, (x2 - x1) // samples)
    step_y = max(1, (y2 - y1) // samples)

    coords = [(x, y) for x in range(x1, x2, step_x) for y in range(y1, y2, step_y)]

    distances = np.array([depth_frame.get_distance(x, y) for x, y in coords])

    valid_distances = distances[distances > 0]

    if valid_distances.size == 0:
        return 0.0

    return float(np.median(valid_distances))


try:
    # 初始化 YOLOv8 模型
    model: YOLO = YOLO("models/yolov8n.pt")

    rs_config: rs.config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline: rs.pipeline = rs.pipeline()
    pipeline.start(rs_config)

    # 初始化 Supervision 的繪圖工具
    box_annotator: sv.BoxAnnotator = sv.BoxAnnotator()
    label_annotator: sv.LabelAnnotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.5
    )

    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        depth_frame: rs.depth_frame = frames.get_depth_frame()
        color_frame: rs.video_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image: np.ndarray = np.asanyarray(color_frame.get_data())
        annotated_image: np.ndarray = color_image.copy()

        results: List[Results] = model(color_image, verbose=False, conf=0.25)
        result: Results = results[0]

        # 將 YOLOv8 的偵測結果轉換為 Supervision 的 Detections 物件
        detections: sv.Detections = sv.Detections.from_ultralytics(result)

        # 獲取深度資訊並添加到 detections 物件中
        depth_values: List[str] = []
        for bbox in detections.xyxy:
            bbox: np.ndarray = bbox.astype(int)
            x1, y1, x2, y2 = [int(i) for i in bbox]
            depth: float = get_median_depth(depth_frame, x1, y1, x2, y2, samples=30)
            depth_values.append(f"{depth:.2f}m")

        # 將深度資訊添加到 detections 的 data 屬性中
        detections.data["depth"] = np.array(depth_values)

        # 準備標籤，包含類別名稱、置信度和深度資訊
        labels: List[str] = [
            f"{model.names[class_id]} {confidence:.2f} {depth}"
            for class_id, confidence, depth in zip(
                detections.class_id, detections.confidence, detections.data["depth"]
            )
        ]

        # 使用 Supervision 繪製邊界框和標籤
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        cv2.namedWindow("object detect with supervision", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("object detect with supervision", annotated_image)

        key: int = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            break
        if (
            cv2.getWindowProperty(
                "object detect with supervision", cv2.WND_PROP_VISIBLE
            )
            < 1
        ):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
except Exception as e:
    traceback.print_exc()

# Reference:
# Ultralytics YOLO Docs - https://docs.ultralytics.com/reference/engine/model/

import sys
import traceback
from typing import Generator, List

import cv2
import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs
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
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.get_device_name(0)=}")
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）


def get_median_depth(
    depth_frame: npt.NDArray[np.uint16], x1: int, y1: int, x2: int, y2: int, samples=30
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
    # 確保 x1, y1 是左上角，x2, y2 是右下角
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

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
    # 名稱後面有個n代表這是 Nano 模型，也是 YOLOv8 裡面最輕量、運行速度最快的模型（運行時，如果指定的路徑找不到該模型檔案，會自動從網路下載）
    model: YOLO = YOLO("models/yolov8n.pt")

    # 產生對應類別數量的隨機顏色
    rng: Generator = np.random.default_rng(1)
    colors: npt.NDArray[np.int32] = rng.integers(
        0, 255, size=(len(model.names), 3), dtype=np.int32
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

        color_image: npt.NDArray[np.uint16] = np.asanyarray(color_frame.get_data())
        # 用於標註的影像
        annotated_image: npt.NDArray[np.uint16] = color_image.copy()

        # 使用 YOLO 模型進行物件偵測
        # verbose=False 代表關閉預設會在終端機輸出的推理資訊，conf=0.25 代表置信度閾值，低於這個值的預測結果將被忽略
        results: List[Results] = model(color_image, verbose=False, conf=0.25)
        result: Results = results[0]

        # result.boxes 是一個物件，包含了所有預測的邊界框、類別和置信度
        classes: npt.NDArray[np.float32] = result.boxes.cls.cpu().numpy()
        confidences: npt.NDArray[np.float32] = result.boxes.conf.cpu().numpy()
        # 邊界框格式可選擇 xywh、xywhn、xyxy、xyxyn
        # xyxy 是左上角座標及右下角座標 [x1, y1, x2, y2]
        # xywh 是左上角座標及邊界框的長寬 [x, y, w, h]
        # 多一個 n 的話，代表是歸一化的座標，表示邊界框的座標是相對於影像大小的比例。範圍在 0 到 1 之間
        boxes: npt.NDArray[np.float32] = result.boxes.xyxy.cpu().numpy()

        for prediction in zip(classes, confidences, boxes):
            # 類別 ID
            class_id: int = prediction[0].astype(int)
            # 置信度，數字區間為 0 到 1 有小數點，1 代表 100% 確定
            score: float = prediction[1].astype(float)
            # 邊界框，格式為 [x1, y1, x2, y2]，x1, y1 是左上角座標（最小值），x2, y2 是右下角座標（最大值）
            # 座標會有小數點，需使用 astype(int) 方法轉換為整數，避免 OpenCV 報錯
            bbox: npt.NDArray[np.int32] = prediction[2].astype(int)
            # 類別名稱，這是從模型的類別名稱列表中獲取的
            label: str = model.names[class_id]
            # 根據類別 ID 獲取隨機出來的固定顏色
            color: list[int] = colors[class_id].astype(int).tolist()

            # 將邊界框座標轉換為整數（OpenCV 的座標必須是整數）
            x1, y1, x2, y2 = [int(i) for i in bbox]
            # 在影像上畫出邊界框
            annotated_image = cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), color, 2
            )

            # 從邊界框範圍內隨機取 30 個點的深度距離並取中間值
            # 不建議使用取平均值的方式，因為有可能會有極端值影響結果
            depth: float = get_median_depth(depth_frame, x1, y1, x2, y2, samples=30)

            # 在邊界框上方標註類別名稱和置信度
            text: str = f"{label} {round(score * 100)}% {depth:.2f}m"
            # 使用 OpenCV 的 putText 函數來寫文字
            font_face: int = cv2.FONT_HERSHEY_SIMPLEX
            font_scale: float = 0.5
            thickness: int = 1

            # 計算文字的寬度和高度
            (tw, th), _ = cv2.getTextSize(
                text=text,
                fontFace=font_face,
                fontScale=font_scale,
                thickness=1,
            )
            th = int(th * 1.2)

            # 在邊界框上方畫出文字的背景矩形
            annotated_image = cv2.rectangle(
                annotated_image, (x1, y1), (x1 + tw, y1 + th), color, -1
            )
            # 在邊界框上方寫上類別名稱和置信度
            # 注意：這裡的文字位置是從左上角開始計算的
            annotated_image = cv2.putText(
                annotated_image,
                text,
                (x1, y1 + th),
                font_face,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        cv2.namedWindow("object detect", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("object detect", annotated_image)

        key: int = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("object detect", cv2.WND_PROP_VISIBLE) < 1:
            break

    pipeline.stop()
    cv2.destroyAllWindows()
except Exception as e:
    traceback.print_exc()

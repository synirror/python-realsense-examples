import sys
import traceback
from collections import defaultdict
from typing import List, Optional

import cv2
import numpy as np
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
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Main GPU: {torch.cuda.get_device_name(0)}")
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）


try:
    # 名稱後面有個n代表這是 Nano 模型，也是 YOLOv8 裡面最輕量、運行速度最快的模型（運行時，如果指定的路徑找不到該模型檔案，會自動從網路下載）
    model: YOLO = YOLO("models/yolov8n.pt")

    rs_config: rs.config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline: rs.pipeline = rs.pipeline()
    pipeline.start(rs_config)

    track_history = defaultdict(lambda: [])

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
        boxes: np.ndarray = result.boxes.xyxy.cpu().numpy()
        # 如果沒有追蹤到物件，則 track_ids 會是空的；如果有追蹤到物件，則 track_ids 會是一個包含所有追蹤 ID 的列表
        track_ids: np.ndarray = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.empty(0)
        )

        for bbox, track_id in zip(boxes, track_ids):
            bbox: np.ndarray = bbox.astype(int)
            track_id: Optional[int]

            x1, y1, x2, y2 = [int(i) for i in bbox]
            annotated_image = cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), (255, 255, 255), 2
            )

            if track_id is not None:  # 檢查 track_id 是否存在
                # 如果有追蹤 ID，則使用該 ID 來獲取追蹤歷史
                track: List[List[int]] = track_history[track_id]

                track.append(bbox.tolist())
                # 限制追蹤歷史的長度，防止記憶體溢出
                if len(track) > 10:
                    track.pop(0)

                # 在邊界框上方顯示追蹤 ID
                cv2.putText(
                    annotated_image,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # 繪製追蹤歷史
                for i in range(len(track) - 1):
                    x1, y1, x2, y2 = track[i]
                    next_x1, next_y1, next_x2, next_y2 = track[i + 1]
                    cv2.line(
                        annotated_image,
                        ((x1 + x2) // 2, (y1 + y2) // 2),
                        ((next_x1 + next_x2) // 2, (next_y1 + next_y2) // 2),
                        (0, 255, 0),
                        2,
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

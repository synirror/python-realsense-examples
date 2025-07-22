import sys
import traceback

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

# region 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）
context = rs.context()
if len(context.query_devices()) == 0:
    print("未檢測到已連接的深度攝影機！")
    sys.exit(1)
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）

try:
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

        depth_image: npt.NDArray[np.uint16] = np.asanyarray(depth_frame.get_data())
        color_image: npt.NDArray[np.uint16] = np.asanyarray(color_frame.get_data())

        height, width = depth_image.shape
        msg = ""
        # 用正方形符號在終端機繪製深度圖
        for y in range(0, height, height // 20):
            for x in range(0, width, width // 60):
                # 取得當前像素的深度值
                depth_value: int = int(depth_image[y, x])
                # 將深度值轉換為距離（單位：公尺）
                distance: float = depth_value * 0.001
                # 根據距離值決定符號的顏色
                if distance == 0:  # 超出深度攝影機的偵測範圍（無法測量）
                    symbol = "X"
                elif distance < 0.5:  # 0.5 公尺以下
                    symbol = "█"
                elif distance < 1:  # 0.5 到 1 公尺
                    symbol = "▓"
                elif distance < 2:  # 1 到 2 公尺
                    symbol = "▒"
                elif distance < 3:  # 2 到 3 公尺
                    symbol = "░"
                else:  # 3 公尺以上
                    symbol = " "
                msg += symbol
            msg += "\n"
        print(msg)
except Exception as e:
    traceback.print_exc()

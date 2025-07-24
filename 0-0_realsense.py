import sys
import traceback

import numpy as np
import pyrealsense2 as rs

# region 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）
context = rs.context()
if len(context.query_devices()) == 0:
    print("未檢測到已連接的深度攝影機！")
    sys.exit(1)
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）


try:
    rs_config = rs.config()
    """
    ===== enable_stream 方法說明 =====
    第1個參數：要啟用的深度攝影機功能，可以代入 rs.stream.depth、rs.stream.color、rs.stream.accel、rs.stream.gyro、rs.stream.pose
    第2個參數：畫面寬度
    第3個參數：畫面高度
    第4個參數：根據你第1個參數要啟用的功能，會有不同對應的格式可以選
    第5個參數：FPS 幀數
    這裡參數全部都不能隨便填，都是需要填入該深度攝影機所支持的參數
    如果需要知道有哪些參數支持，可以從 Intel Realsense Viewer 裡面看，展開下拉選單出現的選項就會是這個深度攝影機所支持的參數了
    載點（從裡面找 Intel.RealSense.Viewer.exe）：https://github.com/IntelRealSense/librealsense/releases
    多個程式不能互相占用深度攝影機，否則最後啟動的程式如果要調用深度攝影機會無法成功調用
    ==============================
    """
    # 啟用色彩影像
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 啟用深度影像
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 啟動深度攝影機
    pipeline = rs.pipeline()
    pipeline.start(rs_config)
    while 1:
        # 等待新的幀數據到來
        # 如果沒有連接深度攝影機，就會卡在這行程式碼，超過 5 秒就會觸發 timeout 而報錯
        # 如果在這行報錯，需檢查 enable_stream 方法的參數是否正確，或是深度攝影機是否有連接到電腦
        frames: rs.composite_frame = pipeline.wait_for_frames()

        depth_frame: rs.depth_frame = frames.get_depth_frame()
        color_frame: rs.color_frame = frames.get_color_frame()

        # 剛開始啟動時，有些情況下會無法從幀裡面提取部分數據，因此這個判斷是必須要有的
        if not depth_frame or not color_frame:
            continue

        depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())
        color_image: np.ndarray = np.asanyarray(color_frame.get_data())

        # shape 第一個值是高度，第二個值是寬度，如果是 640x480，從 shape 取得的會是 [480, 640]
        height, width = depth_image.shape

        # 取得畫面正中心的座標
        center_x, center_y = int(width / 2), int(height / 2)
        # 取畫面正中心的深度距離（單位：km 公尺）
        center_distance: float = depth_frame.get_distance(center_x, center_y)
        center_distance *= 1000  # 將公尺轉成毫米

        # 也可以用這種方式直接從二維陣列取深度距離（單位：mm 毫米）
        # center_distance = depth_image[center_y, center_x]  # 高度在前 寬度在後，[0, 0] 起點在畫面的最左上角

        # 如果出現剛好 0 毫米，代表有可能是距離過近、過遠、過曝或過暗導致超出深度攝影機的偵測範圍，這時候就無法測量距離
        # 由於 D435i 會藉由紅外線感測讓距離更精準，因此不會有環境過暗而判斷不出距離的問題，但是仍有亮度過曝的問題
        print(f"中心點距離：{center_distance:.0f} 毫米(mm)")
except Exception as e:
    traceback.print_exc()

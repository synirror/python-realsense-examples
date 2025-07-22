# Reference:
# Intel Realsense Examples - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
import sys
import traceback

import cv2
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
    # 啟用色彩影像
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 啟用深度影像
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline: rs.pipeline = rs.pipeline()
    pipeline.start(rs_config)  # 啟動深度攝影機
    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        depth_frame: rs.depth_frame = frames.get_depth_frame()
        color_frame: rs.video_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image: npt.NDArray[np.uint16] = np.asanyarray(depth_frame.get_data())
        color_image: npt.NDArray[np.uint16] = np.asanyarray(color_frame.get_data())
        # 如果想要展示深度圖給別人看，直接 imshow depth_image 的話會是呈現黑白的
        # 所以需要使用 OpenCV 的 applyColorMap 函數來映射顏色
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # 使用 OpenCV 來顯示深度攝影機的畫面

        # 將色彩影像調整到與深度影像相同的大小，避免合併時出現錯誤
        resized_color_image: npt.NDArray[np.uint16] = cv2.resize(
            color_image,
            dsize=(depth_colormap.shape[1], depth_colormap.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        # 可以使用 np.hstack 將兩個畫面橫向合併成一個畫面一起顯示，需注意要合併的兩個畫面"高度像素"必須一模一樣
        # 想要垂直合併的話可以使用 np.vstack，要合併的兩個畫面"寬度像素"也是必須一模一樣
        # np.hstack 跟 np.vstack 同時使用的話可以拼湊成網格畫面
        images = np.hstack((resized_color_image, depth_colormap))

        cv2.namedWindow("color and depth", cv2.WINDOW_AUTOSIZE)
        # 如果想要顯示單一畫面，直接 imshow color_image 或 depth_colormap 即可
        cv2.imshow("color and depth", images)

        # 無窮迴圈的最尾端一定要加上 cv2.waitKey(1) ，不加上的話會導致 imshow 沒有時間渲染畫面
        key: int = cv2.waitKey(1)
        # 按下 esc 或 q 關閉程式
        if key & 0xFF == ord("q") or key == 27:
            break

        # 這裡是用來判斷 imshow 視窗是否有關掉，只要主動關掉其中一個視窗就會關閉整個程式而不是又再彈出關掉的視窗。這個判斷可有可無
        # cv2.getWindowProperty 第一個參數是"視窗名稱"，必須對應上面 imshow 一樣的視窗名稱
        if cv2.getWindowProperty("color and depth", cv2.WND_PROP_VISIBLE) < 1:
            break

    # 如果跳出無窮迴圈則釋放資源
    pipeline.stop()
    cv2.destroyAllWindows()
except Exception as e:
    traceback.print_exc()

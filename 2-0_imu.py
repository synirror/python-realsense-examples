import sys
import traceback

import pyrealsense2 as rs

# region 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）
context = rs.context()
devices = context.query_devices()
if len(devices) == 0:
    print("未檢測到已連接的深度攝影機！")
    sys.exit(1)

device = devices[0]
print(f"設備: {device.get_info(rs.camera_info.name)}")
print(f"序列號: {device.get_info(rs.camera_info.serial_number)}")
# 檢查所有可用的感測器
sensors = device.query_sensors()

has_gyro = False
has_accel = False
has_motion = False

for sensor in sensors:
    sensor_name = sensor.get_info(rs.camera_info.name)
    print(f"  感測器: {sensor_name}")

    # 檢查感測器的串流配置檔
    profiles = sensor.get_stream_profiles()
    for profile in profiles:
        stream_type = profile.stream_type()

        if stream_type == rs.stream.gyro and not has_gyro:
            has_gyro = True
            print(f"    ✓ 支援陀螺儀 (Gyroscope)")
        elif stream_type == rs.stream.accel and not has_accel:
            has_accel = True
            print(f"    ✓ 支援加速度計 (Accelerometer)")
        elif stream_type == rs.stream.pose and not has_motion:
            has_motion = True
            print(f"    ✓ 支援運動感測器")

if not has_accel and not has_gyro:
    print("\n檢測到當前深度攝影機不支援加速度計或是陀螺儀，因此終止程式！")
    sys.exit(1)
# endregion 檢查設備是否支援運行這支程式（這裡面程式是不需要的，可略過不看）

try:
    pipeline: rs.pipeline = rs.pipeline()
    rs_config: rs.config = rs.config()
    # 啟用加速度計
    rs_config.enable_stream(rs.stream.accel)
    # 啟用陀螺儀
    rs_config.enable_stream(rs.stream.gyro)

    pipeline.start(rs_config)
    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()

        """
        frames 是一個陣列，順序是根據你上面程式 config.enable_stream 的啟用順序而定
        假設你是依照這順序啟用：color、depth、accel、gyro
        color 就必須用 frames[0] 取，而 accel 就必須用 frames[2] 取，以此類推
        在這個範例因為只有啟用 accel 和 gyro，所以 frames[0] 會是 accel，而 frames[1] 會是 gyro
        """
        accel: rs.motion_frame = frames[0].as_motion_frame().get_motion_data()
        gyro: rs.motion_frame = frames[1].as_motion_frame().get_motion_data()

        # 取得到的資料會是三維向量 Vector3（x、y、z）
        print(f"加速度計數據: {accel}")
        print(f"陀螺儀數據: {gyro}")
        print()
except Exception as e:
    traceback.print_exc()

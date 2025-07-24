# Reference: https://github.com/IntelRealSense/librealsense/issues/4391#issuecomment-510701377

import math
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
    pipeline = rs.pipeline()
    rs_config = rs.config()
    # 啟用加速度計
    rs_config.enable_stream(rs.stream.accel)
    # 啟用陀螺儀
    rs_config.enable_stream(rs.stream.gyro)

    pipeline.start(rs_config)

    # region 這些都是計算攝影機歐拉角所需要保存的數值
    first = True
    alpha = 0.98
    total_gyro_angle_y = 0.0

    accel_angle_x: float
    accel_angle_y: float
    accel_angle_z: float
    last_ts_gyro: float
    # endregion 這些都是計算攝影機歐拉角所需要保存的數值

    while 1:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        accel: rs.motion_frame = frames[0].as_motion_frame().get_motion_data()
        gyro: rs.motion_frame = frames[1].as_motion_frame().get_motion_data()

        timestamp = frames.get_timestamp()

        # 計算第一幀（防止第一幀缺失陀螺儀資料）
        if first:
            first = False
            last_ts_gyro = timestamp

            # 計算加速度計
            accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            accel_angle_x = math.degrees(
                math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z))
            )
            accel_angle_y = math.degrees(math.pi)
            continue

        # 從第二幀開始計算

        # 計算陀螺儀
        dt_gyro = (timestamp - last_ts_gyro) / 1000
        last_ts_gyro = timestamp

        gyro_angle_x = gyro.x * dt_gyro
        gyro_angle_y = gyro.y * dt_gyro
        gyro_angle_z = gyro.z * dt_gyro

        dangleX = gyro_angle_x * 57.2958
        dangleY = gyro_angle_y * 57.2958
        dangleZ = gyro_angle_z * 57.2958

        total_gyro_angle_x = accel_angle_x + dangleX
        # total_gyro_angle_y = accel_angle_y + dangleY
        total_gyro_angle_y = accel_angle_y + dangleY + total_gyro_angle_y
        total_gyro_angle_z = accel_angle_z + dangleZ

        # 計算加速度計
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(
            math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z))
        )
        # accel_angle_y = math.degrees(math.pi)
        accel_angle_y = 0.0

        # 結合陀螺儀和加速度計角度
        combined_angle_x = total_gyro_angle_x * alpha + accel_angle_x * (1 - alpha)
        combined_angle_z = total_gyro_angle_z * alpha + accel_angle_z * (1 - alpha)
        combined_angle_y = total_gyro_angle_y

        pitch = combined_angle_z  # 上下
        yaw = combined_angle_y  # 左右（該角度會有偏移問題，需依賴 magnetometer 三軸磁力計矯正才不會隨時間偏移越來越嚴重，但是 D435i 的 IMU 不具備）
        roll = combined_angle_x  # 傾斜

        print(f"{pitch=:.2f}, {yaw=:.2f}, {roll=:.2f}")

except Exception as e:
    traceback.print_exc()

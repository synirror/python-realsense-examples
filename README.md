# Python Intel Realsense D435i Example Code

這是一個提供有關深度攝影機的範例程式碼的專案，
變數均有類型標註，以便於理解和維護。
本專案使用 D435i 進行測試，其餘型號未經測試。

---

## 環境設定與安裝

建議使用 PyCharm 編輯器閱覽這裡的程式碼，因為 VSCode 在某些地方可能無法推測出變數的類型，並且函數註釋不易閱讀。

### Python 版本要求

Python >= 3.8

### 安裝依賴套件

如果需要使用 CUDA 版 PyTorch，請先優先至 [PyTorch 官方網站](https://pytorch.org/get-started/locally/) 執行官網提供的套件下載指令，然後再使用
`pip install -r requirements.txt`。

如果不需要使用 CUDA 版 PyTorch，則可以直接使用以下指令安裝所有依賴套件。預設會安裝 CPU 版的 PyTorch：

```bash
pip install -r requirements.txt
```

> [!WARNING]
> RTX50 系列顯示卡必須安裝 PyTorch CUDA 12.8 版本才支援。
> 最新版 PyTorch CUDA 12.8 版本要求 Python >= 3.9。
> 安裝時請務必注意 Python 和 CUDA 版本的兼容性。

---

## 專案結構與範例

專案包含多個範例程式碼，展示如何使用不同的感測器和技術進行環境感知和物件偵測。以下是主要的範例：
> [!NOTE]
> 檔名前面的數字表示主要章節，後面的數字表示子章節。
> 主要章節可以跳著看，子章節則是需要從最前的子章節開始看起。
> 如果主要章節跟子章節的數字相同，則表示運行結果相同但是使用了不同的寫法。

### 0. 深度攝影機初始化與數據獲取
- `0-0_realsense.py`: RealSense 相機初始化與數據獲取範例。
- `0-1_loop.py`: 使用迴圈遍歷深度圖深度數據與終端機顯示範例。

### 1. OpenCV 基礎影像處理
- `1-0_opencv.py`: OpenCV 基礎影像處理範例

### 2. IMU 數據處理與相機角度計算
- `2-0_imu.py`: IMU 數據讀取與處理範例
- `2-1_calculate_camera_angle.py`: 透過 IMU 數據計算相機角度的範例

### 3. YOLOv8 物件偵測與追蹤
- `3-0_object_detect.py`: YOLOv8 物件偵測基礎範例
- `3-1_yolo_opencv_sv.py`: YOLOv8 與 OpenCV 結合的物件偵測範例（使用 Supervision 標註工具）
- `3-1_yolo_opencv.py`: YOLOv8 與 OpenCV 結合的物件偵測範例（不依賴額外套件的標註方式）
- `3-2_object_track_sv.py`: YOLOv8 物件追蹤範例（使用 Supervision 標註工具）
- `3-2_object_track.py`: YOLOv8 物件追蹤範例（不依賴額外套件的標註方式）
- `3-3_small_object_detect.py`: SAHI 小型物件偵測範例

專案中的 `models/` 目錄包含了 YOLOv8 所需的模型檔案，程式碼會在執行時自動下載所需的模型：

- `yolov8n.pt`: YOLOv8n 模型檔案，用於物件偵測

---

## 執行範例程式碼

在 PyCharm 中，您可以直接打開任何一個範例程式碼檔案，然後點擊右上角的綠色執行按鈕來運行程式。
如果您使用 VSCode，請確保已安裝 Python 套件，然後在終端中執行以下命令：

```bash
python <範例檔案名稱>.py
```

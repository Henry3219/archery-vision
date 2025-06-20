# 箭靶识别

一个基于 Python 和 OpenCV 的 CV 应用，用于从图像中自动检测箭靶。

## 项目结构

```
.
├── images/         # 存放输入图像
├── results/        # 存放输出结果图像
├── main.py         # 主运行脚本
├── vision.py       # 核心计算机视觉逻辑
├── settings.py     # 参数配置文件
├── utils.py        # 绘图和文件保存等辅助函数
└── README.md       # 本文档
```

## 处理流程

<table border="0" cellpadding="0" cellspacing="0" style="margin:0 auto; border-collapse:collapse; border:none;">
  <tbody>
    <tr style="text-align:center;">
      <td style="padding:0 8px; vertical-align:middle;">
        <img src="https://raw.githubusercontent.com/Henry3219/archery-vision/main/results/1_debug_1_edges.jpg" alt="边缘检测" width="150">
        <br><small>1. 边缘检测</small>
      </td>
      <td style="padding:0 8px; vertical-align:middle;">➡️</td>
      <td style="padding:0 8px; vertical-align:middle;">
        <img src="https://raw.githubusercontent.com/Henry3219/archery-vision/main/results/1_debug_2_cand.jpg" alt="轮廓识别" width="150">
        <br><small>2. 轮廓识别</small>
      </td>
      <td style="padding:0 8px; vertical-align:middle;">➡️</td>
      <td style="padding:0 8px; vertical-align:middle;">
        <img src="https://raw.githubusercontent.com/Henry3219/archery-vision/main/results/1_debug_3_warped.jpg" alt="透视校正" width="150">
        <br><small>3. 透视校正</small>
      </td>
      <td style="padding:0 8px; vertical-align:middle;">➡️</td>
      <td style="padding:0 8px; vertical-align:middle;">
        <img src="https://raw.githubusercontent.com/Henry3219/archery-vision/main/results/1_debug_4_y_mask.jpg" alt="靶心定位" width="150">
        <br><small>4. 靶心定位</small>
      </td>
      <td style="padding:0 8px; vertical-align:middle;">➡️</td>
      <td style="padding:0 8px; vertical-align:middle;">
        <img src="https://raw.githubusercontent.com/Henry3219/archery-vision/main/results/1_result.jpg" alt="最终结果" width="150">
        <br><small>5. 最终结果</small>
      </td>
    </tr>
  </tbody>
</table>

## 安装与使用

**第一步：克隆仓库**
```bash
git clone https://github.com/Henry3219/archery-vision.git
cd archery-vision
```

**第二步：创建并激活 Conda 环境**
```bash
conda create -n vision_env python=3.9 -y
conda activate vision_env
```

**第三步：安装依赖**
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

**第四步：配置**

-   将靶纸图像 (`.jpg`, `.png` 格式) 放入 `images` 文件夹。
-   在 `settings.py` 文件中调整颜色范围、输出尺寸等参数。

**第五步：运行程序**
```bash
python main.py
```

## 输出结果

处理完成的图像将被保存到 `results` 目录。

## 拓展：品字靶识别
**第一步：靶纸识别与校正**

忽略靶心数量，将含有多个靶面的整张纸作为一个对象进行识别和透视校正。
*   定位与识别：通过颜色找到一个靶环作为基准，然后在更大的范围内通过边缘检测和轮廓拟合，识别出整张靶纸的四个角点。
*   透视校正：根据找到的四个角点进行透视变换，将倾斜的靶纸校正为一张标准正视的矩形图像。

**第二步：多目标分割与独立分析**

在校正后的图像上，对每个靶心进行单独的识别与分析。

*   目标分割：在校正后的大图上，通过颜色特征找出所有独立的圆形靶。接着，为每个靶环创建一个独立的矩形切割区域 (ROI)，将其从大图中分离出来。
*   循环分析：遍历每一个切割出的ROI，在其中独立完成找靶心、测量环半径、计算得分环等所有分析步骤。最后，将所有靶心的分析结果统一绘制到校正后的大图上。
# 箭靶视觉识别系统

一个基于 Python 和 OpenCV 的计算机视觉应用，用于从图像中自动检测和分析射箭靶。它可以识别靶纸、校正其透视形变，并检测得分环及靶心。

## 项目结构

```
.
├── images/         # 存放输入图像
├── results/        # 存放输出结果图像
├── main.py         # 主运行脚本
├── vision.py       # 核心计算机视觉逻辑
├── settings.py     # 参数配置文件
├── utils.py        # 绘图和文件保存等辅助函数
├── requirements.txt # 项目依赖
└── README.md       # 本文档
```

## 处理流程

```mermaid
graph TD
    A[加载原始图像] --> B{寻找蓝色信标};
    B --> C{确定ROI};
    C --> D[ROI内边缘检测<br><font size=1><i>例: 1_debug_1_edges.jpg</i></font>];
    D --> E[寻找靶纸四边形<br><font size=1><i>例: 1_debug_2_cand.jpg</i></font>];
    E --> F[透视校正<br><font size=1><i>例: 1_debug_3_warped.jpg</i></font>];
    F --> G[靶心定位 (黄色区域)<br><font size=1><i>例: 1_debug_4_y_mask.jpg</i></font>];
    G --> H{测量主色带半径};
    H --> H_Y[黄色带掩码<br><font size=1><i>例: 1_debug_mask_yellow.jpg</i></font>];
    H --> H_R[红色带掩码<br><font size=1><i>例: 1_debug_mask_red.jpg</i></font>];
    H --> H_B[蓝色带掩码<br><font size=1><i>例: 1_debug_mask_blue.jpg</i></font>];
    H_Y --> I[推算所有环半径];
    H_R --> I;
    H_B --> I;
    I --> J[绘制并保存结果<br><font size=1><i>例: 1_result.jpg</i></font>];
```

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
pip install opencv-python numpy
```

**第四步：配置**

-   将靶纸图像 (`.jpg`, `.png` 格式) 放入 `images` 文件夹。
-   在 `settings.py` 文件中调整算法参数（如颜色范围、输出尺寸等）。

**第五步：运行程序**
```bash
python main.py
```

## 输出结果

处理完成的图像将被保存到 `results` 目录。
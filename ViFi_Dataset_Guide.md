# Vi-Fi 数据集详解

> 数据集来源：**Vi-Fi: Associating Moving Subjects across Vision and Wireless Sensors**（IPSN 2022）
> 代码仓库：[ViTag (SECON 2022)](https://github.com/bryanbocao/vitag)
> 官方原始数据：[Vi-Fi Dataset](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)
> 预处理版本下载：[RAN4model_dfv4p4 (Google Drive)](https://drive.google.com/drive/folders/1fZmWWQoNIhkd7Fk9HhziCPotaCppHOLt)

---

## 一、数据集是什么

Vi-Fi 是一个**视觉 + 无线传感器双模态数据集**，每条序列同时包含：

- **摄像头端**：RGB-D 视频，带边界框标注（谁在哪里）
- **手机端**：持有手机的行人产生的 WiFi/IMU/GPS 信号（谁在移动）

**核心设计理念：** 每个合法参与者同时被相机看到 AND 手持手机发送传感器信号，两路数据**通过 NTP 时间戳天然配对**。这是 OOSTraj 等后续工作能用它做跨模态监督训练的根本前提。

---

## 二、采集硬件

```
                         ┌─────────────────────────────────┐
                         │          采集系统                  │
                         │                                   │
  ┌─────────────┐        │  ┌──────────────────────────┐    │
  │  Google     │  FTM   │  │   Stereolabs ZED2 RGB-D  │    │
  │  Pixel 3a   │───────►│  │   10 fps，深度 0.2~20m    │    │
  │  智能手机    │  IMU   │  │   安装高度 2.4~2.8m       │    │
  │             │───────►│  └──────────────────────────┘    │
  │  FTM @ 3Hz  │        │                                   │
  │  IMU @ 50Hz │        │  ┌──────────────────────────┐    │
  │  GPS @ 1Hz  │  GPS   │  │  Google Nest WiFi AP     │    │
  │ （户外）    │───────►│  │  紧邻相机放置              │    │
  └─────────────┘        │  │  接收 FTM 测距信号         │    │
                         │  └──────────────────────────┘    │
                         └─────────────────────────────────┘
```

| 设备 | 型号 | 采集内容 | 频率 |
|------|------|---------|------|
| RGB-D 摄像头 | Stereolabs ZED2 | 彩色视频 + 深度图 | 10 fps |
| 智能手机 | Google Pixel 3a | WiFi FTM、IMU（加速度+陀螺+磁力计）、GPS | FTM: 3Hz / IMU: 50Hz / GPS: 1Hz |
| WiFi 接入点 | Google Nest AP | 接收 FTM 测距 | — |

---

## 三、数据集规模与场景

### 总体规模

| 项目 | 数量 |
|------|------|
| 总序列数 | **90 条**（室内 15 + 室外 75） |
| 每条序列时长 | 约 3 分钟 |
| 总帧数 | 约 **162,000 帧** |
| 场景数 | **6 个**（1 室内 + 5 室外） |

### 子集说明

| 子集 | 场景 | 合法用户数 | 旁观者数 | 序列数 |
|------|------|-----------|---------|--------|
| **Dataset A**（室内） | 1 个受控办公室环境 | 5 人 | 0 | 15 |
| **Dataset B**（户外） | 5 个真实户外地点 | 2~3 人 | 1~9 人 | 75 |

> **合法用户（Legitimate User）**：持有 Pixel 手机并同意参与实验的人，同时被相机和手机传感器记录。
> **旁观者（Passersby）**：路过的真实行人，只被相机看到，没有手机数据。

---

## 四、传感器信号详解

每个合法用户在每条序列中产生以下对齐的多模态数据：

### 4.1 视觉模态（来自 ZED2 摄像头）

| 字段名 | 维度 | 含义 |
|--------|------|------|
| `BBX5` | 5维 | 边界框 `(x, y, w, h, depth)`，中心坐标 + 宽高 + 深度 |
| `BBX5_Others` | 5维 | 旁观者的边界框（仅户外） |
| `BBXC3` | 3维 | 边界框中心点 `(cx, cy, depth)` |
| `BBXC3_Others` | 3维 | 旁观者中心点 |

> `BBX5` 用于模型训练；`BBXC3` 是简化版，只用中心点坐标，OOSTraj 使用的视觉轨迹即 `BBXC3` 的 `(cx, cy)` 部分。

### 4.2 无线/传感器模态（来自 Pixel 手机）

| 字段名 | 维度 | 含义 | 频率 | 噪声特性 |
|--------|------|------|------|---------|
| `FTM` | 2维 | WiFi 测距结果 `(range, std)`，range 单位毫米 | 3 Hz | 多路径衰落，精度约 1m |
| `FTM_li` | 2维 | FTM 线性插值版（补全到视频帧率） | 10 fps | 同上 |
| `IMU19` | 19维 | 完整 IMU：加速度(3) + 陀螺(3) + 磁力计(3) + 四元数(4) + 旋转矩阵(9)？ | 50 Hz | 积分漂移 |
| `IMUagm9` | 9维 | 简化 IMU：加速度(3) + 陀螺(3) + 磁力计(3) | 50 Hz | — |
| `RSSI` | 1维 | WiFi 信号强度（dBm） | 3 Hz | 波动大 |
| `RSSI_li` | 1维 | RSSI 线性插值版 | 10 fps | — |
| GPS | 2维 | 经纬度（仅 Dataset B 户外） | 1 Hz | 1~4m 误差 |

---

## 五、数据格式（同步版 RAN4model_dfv4p4）

### 5.1 文件夹结构

```
RAN4model_dfv4p4/
├── exps/
│   └── exp1/
│       ├── start_end_ts16_dfv4p4_indoor.json    # 室内各序列的有效起止时间戳
│       └── start_end_ts16_dfv4p4_outdoor.json   # 户外各序列的有效起止时间戳
└── seqs/
    ├── indoor/
    │   └── scene0/
    │       └── <seq_id>/          # 例: 20201223_140951（采集日期_时间）
    │           ├── RGBg_ts16_dfv4p4_ls.json      # 视频帧时间戳列表（g=去匿名化版）
    │           ├── RGBh_ts16_dfv4p4_ls.json      # 视频帧时间戳列表（h=半匿名版）
    │           └── sync_ts16_dfv4p4/             # 同步数据目录
    │               ├── BBX5H_sync_dfv4p4.pkl
    │               ├── FTM_sync_dfv4p4.pkl
    │               ├── FTM_li_sync_dfv4p4.pkl
    │               ├── IMU19_sync_dfv4p4.pkl
    │               ├── IMUagm9_sync_dfv4p4.pkl
    │               ├── RSSI_sync_dfv4p4.pkl
    │               └── RSSI_li_sync_dfv4p4.pkl
    └── outdoor/
        ├── scene1/  scene2/  scene3/
        └── scene4/
            └── <seq_id>/          # 例: 20211007_135415
                ├── RGBg_ts16_dfv4p4_ls.json
                ├── RGB_ts16_dfv4p4_ls.json
                └── sync_ts16_dfv4p4/
                    ├── BBX5_sync_dfv4p4.pkl        # 合法用户边界框
                    ├── BBX5_Others_sync_dfv4p4.pkl # 旁观者边界框
                    ├── BBXC3_sync_dfv4p4.pkl
                    ├── BBXC3_Others_sync_dfv4p4.pkl
                    ├── FTM_sync_dfv4p4.pkl
                    ├── FTM_li_sync_dfv4p4.pkl
                    ├── IMU19_sync_dfv4p4.pkl
                    ├── IMUagm9_sync_dfv4p4.pkl
                    ├── RSSI_sync_dfv4p4.pkl
                    ├── RSSI_li_sync_dfv4p4.pkl
                    └── Others_id_ls.pkl            # 旁观者 ID 列表
```

### 5.2 数据 Tensor 维度格式

所有 `.pkl` 文件加载后形状统一为：

```
(WIN_IDX, SUBJ_IDX, DIM_PER_FRAME, FEAT_DIM)
   窗口索引   用户索引   帧内采样数     特征维度
```

以 `scene4` 的序列 `20211007_144525`（3 个合法用户，1831 个窗口）为例：

| 文件 | Shape | 说明 |
|------|-------|------|
| `BBX5_sync_dfv4p4` | `(1831, 3, 1, 5)` | 1831帧 × 3用户 × 1 × 5维BBX |
| `BBXC3_sync_dfv4p4` | `(1831, 3, 1, 3)` | 边界框中心点 |
| `IMU19_sync_dfv4p4` | `(1831, 3, 1, 19)` | 完整 IMU |
| `IMUagm9_sync_dfv4p4` | `(1831, 3, 1, 9)` | 简化 IMU |
| `FTM_sync_dfv4p4` | `(1831, 3, 1, 2)` | FTM 测距 |
| `FTM_li_sync_dfv4p4` | `(1831, 3, 1, 2)` | FTM 线性插值 |
| `RSSI_sync_dfv4p4` | `(1831, 3, 1, 1)` | RSSI 强度 |
| `RGB_ts16_dfv4p4_ls` | `len=2001` | 视频帧时间戳列表 |

> `WIN_IDX=1831` 而 `RGB 帧数=2001`，是因为同步后部分边界帧被裁剪。

### 5.3 数据读取示例

```python
import pickle as pkl

# 读取任意模态（将 BBX5 替换为其他模态名即可）
with open('path/to/BBX5_sync_dfv4p4.pkl', 'rb') as f:
    BBX5 = pkl.load(f)
    # shape: (WIN_IDX, SUBJ_IDX, 1, 5)

# 取第 0 个用户的所有帧 BBX5 数据
user0_bbx = BBX5[:, 0, 0, :]   # shape: (WIN_IDX, 5)

# 读取时间戳列表（JSON）
import json, numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open('path/to/RGB_ts16_dfv4p4_ls.json', 'r') as f:
    ts_list = json.load(f)   # 长度 = 视频总帧数
```

---

## 六、时间同步机制

多传感器采样率不同，原始数据需要对齐：

```
视频帧（RGB）   ████████████████  10 fps  ← 锚定时间轴
FTM 信号       █   █   █         3 fps
IMU 数据       ████████████████  50 Hz（降采样到 10fps）
GPS 信号       █                 1 fps

同步方式：
  ├── 默认：最近邻对齐（nearest timestamp alignment）
  │         → FTM_sync, IMU_sync, RSSI_sync
  └── 线性插值：在两个相邻采样点之间线性补全
              → FTM_li_sync, RSSI_li_sync
```

时间戳格式：`ts16`，即 16 位 Unix 时间戳字符串，精确到微秒，例如：
```
1633628794.773000
```

对于多用户时钟漂移，每条序列记录了各用户手机相对于相机的时钟偏移（offset，单位毫秒），预处理时校正。

---

## 七、原始数据目录结构（预处理前的 RAN）

```
RAN/seqs/
├── indoor/scene0/
│   └── 20201223_140951/
│       ├── Depth/              # 深度图序列（.png）
│       ├── GND/                # 标注文件（ground truth，边界框 JSON）
│       ├── IMU/                # 手机 IMU 原始 CSV
│       ├── IMU_NED_pos/        # IMU 积分得到的 NED 坐标系位置
│       ├── RGB_ts16_dfv4_anonymized/  # 人脸打码后的视频帧（.jpg）
│       └── WiFi/               # FTM/RSSI 原始 CSV
└── outdoor/scene1~4/
    └── <同结构>
```

> 原始数据因隐私（IRB 协议）不公开，只提供预处理后的 `RAN4model_dfv4p4`。

---

## 八、数据预处理流水线

```
原始采集数据（RAN）
       │
       ▼
Step 1: sync_1 ── 将各传感器数据粗对齐到视频时间轴
       │          脚本: IMU_sync1_all.py / FTM_sync1_all.py
       │
       ▼
Step 2: sync_2 ── 精确时钟偏移校正 + 降采样/插值
       │          脚本: RGBh_dfv4_IMU_sync2_FTM_sync2_all.py
       │
       ▼
Step 3: 生成 RAN4model_dfv4p4
          ├── 标准化命名（ts16_dfv4p4）
          ├── 生成 FTM_li（线性插值版）
          ├── 生成 Others 旁观者字段
          └── 保存为 .pkl 格式
```

---

## 九、与后续工作的数据使用对应

| 后续工作 | 使用的字段 | 用途 |
|---------|-----------|------|
| **Vi-Fi**（IPSN 2022） | `BBX5` + `FTM` + `IMU19` | 视觉-手机身份关联（主任务） |
| **ViTag**（SECON 2022） | `BBX5` + `IMU19` + `FTM` | 跨模态翻译 + 关联（X-Translator） |
| **OOSTraj**（CVPR 2024） | `GPS`（noisy 传感器轨迹） + `BBXC3` 中心点（视觉轨迹真值） | 视野外轨迹预测，GPS 作为带噪输入，BBX 中心点作为去噪目标 |

> OOSTraj 使用的 GPS 信号在 Vi-Fi 原始论文中属于附加字段（仅 Dataset B 户外），同步处理脚本由 Hai-chao Zhang 贡献，见 [Vi-FiDatasetProcessing](https://github.com/Hai-chao-Zhang/Vi-FiDatasetProcessing)。

---

## 十、关键数值速查

| 参数 | 值 |
|------|----|
| 总序列数 | 90（15 室内 + 75 户外） |
| 总帧数 | ~162,000 |
| 视频帧率 | 10 fps |
| FTM 频率 | 3 Hz（插值后 10 fps） |
| IMU 频率 | 50 Hz（同步后 10 fps） |
| GPS 频率 | 1 Hz（仅户外） |
| 最大同时参与用户数 | 5 人（室内） / 3 人（户外） |
| 最大旁观者数 | 9 人（户外） |
| 深度测量范围 | 0.2~20 m |
| FTM 精度 | ~1 m（视多路径环境） |
| GPS 误差 | 1~4 m |

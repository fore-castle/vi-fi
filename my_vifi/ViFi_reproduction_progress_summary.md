# Vi-Fi 论文复现

 Vi-Fi 论文中跨视觉与无线传感器行人关联任务的复现进展

## 1. 数据集与任务

当前复现使用的是 Vi-Fi 的同步预处理数据集 `RAN4model_dfv4p4`。该数据集将摄像头检测轨迹与手机无线传感器序列按时间同步，用于判断某个 phone entity 是否对应某个 camera entity。

数据集中主要包含两类模态：

- 视觉端：摄像头检测框中心、深度等信息，深度可近似反映目标与摄像头之间的距离。
- 无线端：FTM 测距、FTM 标准差，以及 IMUagm9 中的加速度、陀螺仪、磁力计信息。

## 2. 二分图方法复现结果

二分图方法复现的是论文中的传统匹配基线。该方法先计算 camera entity 与 phone entity 之间的相似度，再使用 Hungarian algorithm 做二分图匹配。

相似度主要由三部分组成：

- 朝向相似度 `I_phi`
- 轨迹或位置变化相似度 `I_j`
- 距离相似度 `I_d`

带距离版本的加权形式为：

```text
E_uv = 0.1 * I_phi + 0.1 * I_j + 0.8 * I_d
```

这说明在作者原始二分图基线中，距离项本身就是最主要的匹配依据。

当前复现结果如下：

| 数据集 | Online IDP | Offline IDP | 论文参考 |
| --- | ---: | ---: | --- |
| Dataset A indoor | 0.682 | 0.854 | 约 0.78 / 0.83 |
| Dataset B outdoor | 0.540 | 0.565 | 约 0.60 / 0.68 |
| Outdoor scene4 | 0.565 | 0.616 | - |

仅从传统方法看，距离信息已经是非常强的匹配信号，且作者原方法中也给了距离项 0.8 的高权重。

## 3. 深度学习方法复现结果

深度学习方法复现的是 Vi-Fi 的 affinity model。模型结构大致为：

1. 使用 Bi-LSTM 分别编码 camera 序列和 phone 序列。
2. 构造 camera-phone 两两组合特征。
3. 通过 1x1 convolution 输出 affinity matrix。
4. 额外加入 unmatched row/column，用于处理未匹配的 camera 或 phone entity。
5. 使用行、列两个方向的交叉熵和一致性约束进行训练。

当前默认输入输出形式为：

- camera 输入：最多 15 个 camera entity，每个 entity 为 `10 * 3 + 1 = 31` 维，即 10 帧 `(x, y, depth)` 加有效长度。
- phone 输入：最多 5 个 phone entity，每个 entity 为 `10 * 11 + 1 = 111` 维，即 10 帧 11 维无线特征加有效长度。
- 输出：`(Np + 1, Nc + 1)` 的 affinity matrix，当前为 `(6, 16)`。

| 方法 | Online Accuracy | Offline Accuracy |
| --- | ---: | ---: |
| 深度学习 full model | 85.26% | 92.54% |

## 4. 消融实验发现

为了判断距离信息是否主导模型性能，当前进行了 4 组消融实验。实验均为短程训练，主要用于观察趋势。

| 实验 | 输入设置 | Online Accuracy | Offline Accuracy | 结论 |
| --- | --- | ---: | ---: | --- |
| `distance_only` | camera depth + phone FTM range | 74.73% | 83.12% | 仅距离信息已经很强 |
| `no_distance` | 移除 camera depth 和 FTM range/std，仅保留 x/y 与 IMU | 51.36% | 59.24% | 去掉距离后性能大幅下降 |
| `ftm_only` | camera x/y/depth + phone FTM range/std，移除 IMU | 80.30% | 89.07% | FTM 与视觉几何几乎恢复大部分性能 |

消融实验支持以下判断：

1. depth 与 FTM range 是 Vi-Fi 匹配任务中最关键的信号。
2. 只依靠距离信息已经可以获得较高 offline 匹配效果。
3. 移除距离后模型性能明显下降，说明 IMU、轨迹方向等信息不是当前复现结果的主要来源。

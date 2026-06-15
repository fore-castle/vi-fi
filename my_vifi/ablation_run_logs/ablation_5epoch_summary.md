# Vi-Fi 特征消融实验总结

本文总结 `my_vifi` 复现代码中的特征消融实验。

本实验主要回答一个问题：

> 当前 Vi-Fi 复现结果较高，是否主要由“距离信号”贡献，即相机侧的 `depth` 和无线侧的 `FTM range`？

## 实验设置

- 数据集根目录：`RAN4model_dfv4p4`
- Fold：`fold1`
- 测试集划分：按作者 `fold_info.txt` 配置
- 当前实际使用的测试序列：4 条
- 缺失的官方测试序列：`20210907_145202`
- 训练集划分：除配置测试序列之外的所有可用序列
- 注意：当前训练集仍包含 indoor 序列
- 评估方式：
  - Online：逐帧预测，不使用投票
  - Offline：30 帧一致性投票

同一本地划分下，完整模型此前测试结果为：

| 模型 | Online | Offline 30 帧投票 |
|---|---:|---:|
| 完整模型 | 89.37% | 94.20% |

## 消融实验定义

| 消融实验 | Camera 输入 | Phone 输入 | 含义 |
|---|---|---|---|
| `distance_only` | `depth` | `FTM range` | 最小距离模型，只保留视觉深度和无线测距 |
| `no_distance` | `x, y` | `IMUagm9` | 去掉 camera depth 和 FTM range/std |
| `ftm_only` | `x, y, depth` | `FTM range, FTM std` | 使用视觉位置/深度 + FTM，去掉 IMU |
| `single_frame_distance` | 当前帧 `depth` | 当前帧 `FTM range` | 尝试只用单帧距离进行匹配 |

命名说明：

`ftm_only` 并不是“只用无线信号”。它的准确含义是“不使用 IMU”。该实验仍然使用 camera 侧的 `x, y, depth`。

更准确的命名可以是：

```text
distance_only  -> depth_range_only
ftm_only       -> vision_plus_ftm_no_imu
no_distance    -> no_depth_no_ftm
```

## 实验结果

| 消融实验 | Checkpoint | Online | Offline 30 帧投票 |
|---|---|---:|---:|
| `distance_only` | `fold1_best_epoch8_acc0.7646.pth` | 74.73% | 83.12% |
| `no_distance` | `fold1_best_epoch5_acc0.5407.pth` | 51.36% | 59.24% |
| `ftm_only` | `fold1_best_epoch5_acc0.8088.pth` | 80.30% | 89.07% |
| `single_frame_distance` | `fold1_best_epoch1_acc0.1758.pth` | 0.81% | 89.46% |

## 主要结论

### 1. 距离是非常重要的信号

`distance_only` 只使用：

```text
camera depth + FTM range
```

结果达到：

```text
Online:  74.73%
Offline: 83.12%
```

这个结果已经明显高于随机匹配，说明视觉深度和 FTM 测距之间的对应关系本身就是很强的跨模态关联线索。

也就是说，即使完全去掉：

```text
camera x/y
FTM std
IMU
```

模型仍然能获得较好的匹配效果。

### 2. 去掉距离后性能明显下降

`no_distance` 同时去掉：

```text
camera depth
FTM range / FTM std
```

只保留：

```text
camera x/y + IMU
```

结果下降到：

```text
Online:  51.36%
Offline: 59.24%
```

这比 `distance_only` 低很多，也比完整模型低很多。因此，该实验支持如下判断：

> 距离通道是当前任务中的主导信息之一。

如果没有 `depth` 和 `FTM range`，模型仍能利用部分横向位置和运动信号，但效果已经明显不足。

### 3. FTM + 视觉几何几乎恢复完整模型

`ftm_only` 使用：

```text
camera x/y/depth + FTM range/std
```

去掉：

```text
IMU accel/gyro/mag
```

结果为：

```text
Online:  80.30%
Offline: 89.07%
```

这个结果已经接近论文中 Vi-Fi 的参考值，也明显接近完整模型。

这说明：

> FTM 测距信号加上视觉几何信息，已经解释了模型的大部分性能；IMU 在当前划分上的边际贡献可能较小。

这并不表示 IMU 没有用。IMU 可能在更困难的样本中有帮助，例如：

- 多个行人的距离非常接近
- 行人交叉或遮挡
- FTM 噪声较大
- 短时间视觉检测缺失
- 多人以相似距离同时移动

但在当前本地测试划分上，IMU 似乎不是最主要的性能来源。

### 4. 单帧距离实验暂时不可靠

`single_frame_distance` 的结果是：

```text
Online:  0.81%
Offline: 89.46%
```

这个结果非常异常：online 几乎为零，但 offline 投票后很高。

目前发现一个具体实现问题：当 `window_size = 1` 时，当前 `compute_norm_stats()` 统计归一化参数时只统计 `valid_len > 1` 的样本。因此单帧窗口下没有有效样本被纳入统计，导致均值和标准差变成：

```text
Camera mean: [0.]
Camera std:  [0.]
Phone mean:  [0.]
Phone std:   [0.]
```

因此，`single_frame_distance` 当前结果不能直接用于结论分析。尤其是 offline 89.46% 不能被简单解释为“单帧距离就足够好”。这个实验需要修复归一化统计后重新运行。

## 总体解释

消融结果支持如下解释：

1. Vi-Fi 任务中存在很强的几何距离线索。
2. `camera depth` 和 `FTM range` 是模型性能的重要来源。
3. `camera x/y/depth + FTM range/std` 已经能够接近完整模型。
4. 去掉距离后性能显著下降。
5. 在当前 split 上，IMU 的贡献可能小于 FTM-depth/视觉几何信号。

简而言之：

> 当前复现代码的高准确率不一定是“作弊”，但任务本身很大程度上可以由距离和几何关系解决。高性能很可能主要来自 FTM-depth 这一强信号。

## 与论文二分图基线的关系

论文中的二分图基线方法给距离项较高权重，例如距离相关权重达到 0.8，这与本次消融观察一致。

这说明作者的方法设计中也承认：

```text
距离 / 轨迹距离 是最强的匹配线索之一
```

深度学习模型虽然使用了 LSTM 和多模态特征，但它很可能也优先学习到了：

```text
camera depth trajectory <-> FTM range trajectory
```

这种强对应关系。

## 注意事项

- 当前本地数据缺少 `20210907_145202`，因此测试集不是完整官方 fold。
- 当前训练集仍包含 indoor 序列。
- 这些消融多数只训练了 5 epoch，适合观察趋势，不适合作为最终严格指标。
- `distance_only` 使用的是更长一些的 epoch8 checkpoint，因为其 loss/val acc 后续基本趋于平稳。
- `single_frame_distance` 存在归一化统计问题，应修复后重跑。
- 当前 checkpoint 仍是基于 test split 的 validation accuracy 选择的，因此结果应视为诊断性实验，而不是严格最终测试结果。

## 后续建议

1. 修复 `window_size = 1` 时的归一化统计逻辑。
2. 重跑 `single_frame_distance`。
3. 对最关键的两个消融做更长训练：
   - `distance_only`
   - `ftm_only`
4. 如果要严格写论文式复现报告，应划分独立 validation set，只在最后使用 test set 评估一次。

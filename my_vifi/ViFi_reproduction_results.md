# Vi-Fi Deep Affinity Learning — PyTorch 复现结果

## 概述

基于 PyTorch 复现 Vi-Fi 论文（IPSN 2022）的深度亲和矩阵学习方法，使用 RAN4model_dfv4p4 数据集。

## 交叉验证结果（4-fold leave-one-scene-out）

| Fold | 测试场景 | 最佳 Epoch | Val Acc (在线) | 论文在线 | 提升 |
|------|---------|-----------|--------------|---------|------|
| 1 | CAC spot2 (scene1) | 175 | **91.85%** | ~81% | +10.8% |
| 2 | Busch (scene2) | 67 | **93.83%** | ~81% | +12.8% |
| 3 | Liv spot1 (scene3) | ~130 | **~88.8%** | ~81% | +7.8% |
| 4 | Liv spot2 (scene4) | 196 | **89.42%** | ~81% | +8.4% |
| **平均** | | | **~91.0%** | ~81% | +10% |

## 离线推理（30 帧一致性投票）

| Fold | 无投票 | 30 帧投票 | 论文离线 |
|------|--------|----------|---------|
| 1 | 85.6% | **98.94%** | ~90% |

## 关键优化

| 优化 | 效果 |
|------|------|
| 特征 Z-score 归一化 | acc 61% → 84.5%（最关键） |
| 批量 LSTM（20 次循环 → 2 次） | ~10× LSTM 加速 |
| AMP 混合精度 | ~1.5× 加速 |
| num_workers=2 | ~2× 数据加载加速 |
| batch_size 64 | GPU 利用率提升 |

**训练速度**：~4 分钟/epoch → ~1 分钟/epoch（~4× 提升）

## 模型配置

- Bi-LSTM (2 层, 双向, hidden=32)
- 相机输入: (x, y, depth) × 10 帧
- 手机输入: FTM range/std + IMU accel/gyro/mag × 10 帧
- 1×1 Conv 压缩: 64→128→64→32→16→1
- 4 项损失: L_pc + L_cp + L_cons + L_aff
- Optimizer: SGD, lr=1e-3, momentum=0.9
- LR 衰减: epoch 100 → 1e-4

## 数据集

- RAN4model_dfv4p4（同步预处理版）
- 15 室内序列 + 67 室外序列
- 125k 训练样本，~1800 验证样本/fold

## 文件结构

```
my_vifi/
├── config.py          # 超参数
├── dataset.py         # 数据加载 + 滑动窗口构建
├── model.py           # Bi-LSTM + 1×1 Conv
├── loss.py            # 4 项亲和矩阵损失
├── train.py           # 训练脚本 (AMP, 批量 LSTM)
├── inference.py       # 离线推理 + 一致性投票
└── checkpoints/       # 模型文件
```

# Strict Vi-Fi Full Model Experiment

- Fold: `1`
- Epochs: `30`
- Batch size: `64`
- Validation split: `val_ratio=0.1`, `val_seed=2026`
- Checkpoint selected by validation only: `D:\buaa_study\research\vi-fi\my_vifi\checkpoints_strict\full\fold1_best_epoch29_acc0.8552.pth`
- Test set is evaluated once after training.

## Test Results

- Online accuracy (no voting):   85.26%
- Offline accuracy (30-frame vote): 92.54%

## Train Log Tail

```text
Epoch 29, Iter 1700: loss=0.3492, acc=0.8564, lr=0.000100
Epoch 29/30: train_loss=0.3725, train_acc=0.8543, val_loss=0.3545, val_acc=0.8552
New best: fold1_best_epoch29_acc0.8552.pth
Epoch 30, Iter 50: loss=0.3358, acc=0.8786, lr=0.000100
Epoch 30, Iter 100: loss=0.3339, acc=0.8858, lr=0.000100
Epoch 30, Iter 150: loss=0.3761, acc=0.8791, lr=0.000100
Epoch 30, Iter 200: loss=0.4102, acc=0.8471, lr=0.000100
Epoch 30, Iter 250: loss=0.3730, acc=0.8527, lr=0.000100
Epoch 30, Iter 300: loss=0.3627, acc=0.8564, lr=0.000100
Epoch 30, Iter 350: loss=0.3884, acc=0.8559, lr=0.000100
Epoch 30, Iter 400: loss=0.3881, acc=0.8718, lr=0.000100
Epoch 30, Iter 450: loss=0.3641, acc=0.8680, lr=0.000100
Epoch 30, Iter 500: loss=0.3848, acc=0.8404, lr=0.000100
Epoch 30, Iter 550: loss=0.4473, acc=0.8156, lr=0.000100
Epoch 30, Iter 600: loss=0.5182, acc=0.7887, lr=0.000100
Epoch 30, Iter 650: loss=0.3156, acc=0.8690, lr=0.000100
Epoch 30, Iter 700: loss=0.3113, acc=0.8842, lr=0.000100
Epoch 30, Iter 750: loss=0.3757, acc=0.8391, lr=0.000100
Epoch 30, Iter 800: loss=0.4272, acc=0.8198, lr=0.000100
Epoch 30, Iter 850: loss=0.3891, acc=0.8492, lr=0.000100
Epoch 30, Iter 900: loss=0.4095, acc=0.8322, lr=0.000100
Epoch 30, Iter 950: loss=0.4409, acc=0.8246, lr=0.000100
Epoch 30, Iter 1000: loss=0.3430, acc=0.8698, lr=0.000100
Epoch 30, Iter 1050: loss=0.3277, acc=0.8634, lr=0.000100
Epoch 30, Iter 1100: loss=0.4225, acc=0.8250, lr=0.000100
Epoch 30, Iter 1150: loss=0.3572, acc=0.8332, lr=0.000100
Epoch 30, Iter 1200: loss=0.3922, acc=0.8369, lr=0.000100
Epoch 30, Iter 1250: loss=0.3171, acc=0.8740, lr=0.000100
Epoch 30, Iter 1300: loss=0.3332, acc=0.8556, lr=0.000100
Epoch 30, Iter 1350: loss=0.3672, acc=0.8528, lr=0.000100
Epoch 30, Iter 1400: loss=0.3982, acc=0.8480, lr=0.000100
Epoch 30, Iter 1450: loss=0.3363, acc=0.8641, lr=0.000100
Epoch 30, Iter 1500: loss=0.4389, acc=0.8097, lr=0.000100
Epoch 30, Iter 1550: loss=0.3995, acc=0.8571, lr=0.000100
Epoch 30, Iter 1600: loss=0.3009, acc=0.9069, lr=0.000100
Epoch 30, Iter 1650: loss=0.3622, acc=0.8360, lr=0.000100
Epoch 30, Iter 1700: loss=0.3737, acc=0.8654, lr=0.000100
Epoch 30/30: train_loss=0.3721, train_acc=0.8541, val_loss=0.3560, val_acc=0.8544
Saved: D:\buaa_study\research\vi-fi\my_vifi\checkpoints_strict\full\fold1_epoch30.pth
Training complete. Best val acc: 0.8552
```

Full train log: `D:\buaa_study\research\vi-fi\my_vifi\strict_run_logs\full.fold1.strict.train.log`

Full test log: `D:\buaa_study\research\vi-fi\my_vifi\strict_run_logs\full.fold1.strict.test.log`

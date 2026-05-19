# Traditional Baseline 使用说明

这份文档写给负责传统事件表示 baseline 的同学。目标是把传统方法接到和
learning-based 方法相同的 benchmark 流程里，方便最后做公平对比。

## 现在支持哪些传统表示

当前已经实现了 5 类常用传统事件表示：

| 方法名 | 含义 | 输出通道 |
|---|---|---:|
| `event_frame` / `event_count` | 正负极性事件计数图 | 2 |
| `binary_event_image` | 正负极性二值事件图，有事件就是 1 | 2 |
| `timestamp_image` | 每个像素最近一次事件的归一化时间 | 2 |
| `time_surface` | 指数衰减时间表面，表示事件有多“新” | 2 |
| `voxel_grid` | 按时间分 bin 的体素网格，默认 5 个 bin | 10 |

所有方法都使用同一个输入输出接口：

```text
输入: events, shape = Nx4, columns = [x, y, t, p]
输出: representation, shape = CxHxW, dtype = float32
```

通道顺序统一为：正极性在前，负极性在后。例如 `voxel_grid` 的前 5 个
channel 是正极性时间 bin，后 5 个 channel 是负极性时间 bin。

## 覆盖哪些任务

traditional baseline 计划覆盖 4 条线：

| 数据集 | 任务 | 下游模型 |
|---|---|---|
| N-MNIST | 分类 | ResNet18 |
| N-Caltech101 | 分类 | ResNet18 |
| GEN1 | 目标检测 | YOLOv6 |
| MVSEC | 光流估计 | EV-FlowNet-like decoder |

目前代码已经完成这些接入：

- N-MNIST / N-Caltech101: `tasks/classification/scripts/train_traditional_classification.py`
- GEN1 detection: `tasks/detection/scripts/train_gen1_detection.py --method <traditional_method>`
- MVSEC optical flow: `tasks/optical_flow/scripts/run_original_protocol.py --adapter <traditional_method>`

## 跑实验前先检查环境

在 GPU 机器上先检查 PyTorch 和 CUDA：

```bash
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果 `cuda` 是 `False`，先不要正式跑。需要换 PyTorch/CUDA 镜像，或者重新安装
匹配 CUDA 版本的 `torch` 和 `torchvision`。

然后安装项目依赖：

```bash
pip install -r requirements.txt
```

## 先跑测试，不要直接跑满 100 epoch

每次上新机器，先跑轻量测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v

cd tasks/optical_flow
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
cd ..
```

如果本地或服务器没有安装 torch，部分 torch 相关测试会显示 `skipped`。这不代表
traditional representation 出错，只是训练相关测试没有执行。

## N-MNIST 分类 smoke test

N-MNIST 比较小，最适合先检查训练闭环。第一次只跑 1 个 epoch 和少量样本：

```bash
python tasks/classification/scripts/train_traditional_classification.py \
  --dataset nmnist \
  --root /mnt/datasets \
  --method event_frame \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 128 \
  --test-limit 128 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda \
  --output-dir /mnt/artifacts/classification/traditional/nmnist/event_frame_smoke
```

这个命令的目的不是追求高准确率，而是检查：

- 数据能不能正常下载或读取
- representation 能不能正常构建
- ResNet18 能不能 forward/backward
- loss 和 accuracy 是不是有限数值
- checkpoint 和日志能不能正常保存

跑完后检查输出目录：

```bash
ls /mnt/artifacts/classification/traditional/nmnist/event_frame_smoke
```

应该能看到：

```text
config.json
history.jsonl
metrics.json
progress.json
representation_stats.json
train.log
checkpoints/
```

## N-MNIST 正式 baseline

smoke test 通过后，再跑 5 个方法：

```bash
for method in event_frame binary_event_image timestamp_image time_surface voxel_grid; do
  python tasks/classification/scripts/train_traditional_classification.py \
    --dataset nmnist \
    --root /mnt/datasets \
    --method $method \
    --epochs 100 \
    --early-stop-patience 10 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda \
    --resume \
    --output-dir /mnt/artifacts/classification/traditional/nmnist/$method
done
```

训练采用 `max 100 epochs + early stopping`。也就是说，不一定每个方法都会跑满
100 轮；如果验证集长期不提升，会提前停止。最终报告应使用 best checkpoint 的
test metric，而不是最后一轮的结果。

## N-Caltech101 分类

N-Caltech101 比 N-MNIST 更大，建议先跑 smoke test：

```bash
python tasks/classification/scripts/train_traditional_classification.py \
  --dataset ncaltech101 \
  --root /mnt/datasets \
  --method event_frame \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 128 \
  --test-limit 128 \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda \
  --output-dir /mnt/artifacts/classification/traditional/ncaltech101/event_frame_smoke
```

如果显存足够，再正式跑：

```bash
for method in event_frame binary_event_image timestamp_image time_surface voxel_grid; do
  python tasks/classification/scripts/train_traditional_classification.py \
    --dataset ncaltech101 \
    --root /mnt/datasets \
    --method $method \
    --epochs 100 \
    --early-stop-patience 10 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda \
    --resume \
    --output-dir /mnt/artifacts/classification/traditional/ncaltech101/$method
done
```

如果遇到 CUDA OOM，把 `--batch-size 32` 改成 `16` 或 `8`。

## GEN1 detection

GEN1 检测更重，建议放在分类 baseline 跑通之后。先构建 window index：

```bash
python tasks/detection/scripts/build_gen1_window_index.py \
  --root /path/to/detection_dataset_duration_60s_ratio_1.0
```

然后先跑小规模 smoke test：

```bash
python tasks/detection/scripts/train_gen1_detection.py \
  --root /path/to/detection_dataset_duration_60s_ratio_1.0 \
  --method event_frame \
  --epochs 1 \
  --train-limit 32 \
  --val-limit 16 \
  --test-limit 16 \
  --batch-size 4 \
  --num-workers 0 \
  --device cuda \
  --output-dir outputs/debug/gen1_event_frame_smoke
```

GEN1 依赖 YOLOv6 third-party 代码和数据路径，跑之前要确认 benchmark 组的检测
环境已经配好。

## MVSEC optical flow

MVSEC 的 traditional adapter 已经接入 `optical-flow`。先跑 mock smoke：

```bash
cd tasks/optical_flow
python scripts/run_smoke.py
python scripts/run_linear_benchmark.py --adapter event_frame --use-mock
```

真实 MVSEC formal protocol 可以显式指定 adapter：

```bash
python scripts/run_original_protocol.py \
  --adapter event_frame \
  --data-root /path/to/mvsec \
  --epochs 100 \
  --early-stop-patience 10 \
  --device cuda
```

具体真实数据路径和参数以 optical-flow 组当前文档为准。传统方法不要默认混进原有
six-method learning-based suite；需要跑时显式指定 adapter。

## 训练过程中要记录什么

每个 run 的输出目录里，重点看这些文件：

| 文件 | 用途 |
|---|---|
| `config.json` | 本次实验完整配置 |
| `history.jsonl` | 每一轮 train/val 指标 |
| `metrics.json` | 最终汇总结果 |
| `progress.json` | 当前或最终进度 |
| `representation_stats.json` | 表示张量统计，例如非零比例、构建时间 |
| `train.log` | 人能直接读的训练日志 |
| `checkpoints/best.pt` | 验证集最佳模型 |
| `checkpoints/last.pt` | 最近一轮模型，方便 resume |

写报告时建议记录：

- method
- dataset
- downstream model
- best epoch
- test accuracy / mAP / AEE
- early stopping 是否触发
- representation shape
- nonzero ratio
- mean build time

这样结果不是黑盒，后面可以解释为什么某个传统表示更快、更稀疏，或者准确率更高。

## 推荐执行顺序

最省钱、最稳的顺序是：

1. N-MNIST `event_frame` smoke test
2. N-MNIST 五个方法正式跑
3. N-Caltech101 `event_frame` smoke test
4. N-Caltech101 五个方法正式跑
5. MVSEC mock smoke
6. MVSEC formal protocol
7. GEN1 detection smoke
8. GEN1 detection formal run

不要先跑 GEN1。它依赖重、显存和时间成本都更高，适合放在最后。

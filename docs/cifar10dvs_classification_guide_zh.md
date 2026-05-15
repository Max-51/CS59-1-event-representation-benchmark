# CIFAR10-DVS Classification Benchmark

This guide runs CIFAR10-DVS recognition with the same shared classification
interfaces used by the N-MNIST and N-Caltech101 benchmark runs.

## Dataset

Official download URL:

```text
https://figshare.com/ndownloader/files/38023437
```

Expected local filename:

```text
CIFAR10DVS.zip
```

On the GPU server:

```bash
cd ~/CS59-1-event-representation-benchmark
mkdir -p data/cifar10dvs
wget -O data/cifar10dvs/CIFAR10DVS.zip https://figshare.com/ndownloader/files/38023437
```

If the server cannot reach Figshare reliably, download the same file elsewhere
and upload it to:

```text
data/cifar10dvs/CIFAR10DVS.zip
```

## Split And Protocol

CIFAR10-DVS has no official train/test split in Tonic, so the scripts use a
deterministic 80/20 split with `seed=42`. Traditional baselines then reserve
10% of the training split as validation for early stopping.

Shared full-run parameters:

```text
epochs=100
early_stop_patience=10
batch_size=32
lr=0.0001
weight_decay=0.0001
num_workers=4
seed=42
max_events=50000
```

## Smoke Test

```bash
cd ~/CS59-1-event-representation-benchmark
DATA_ROOT=$PWD/data/cifar10dvs RUN_TAG=smoke_cifar10dvs EPOCHS=1 PATIENCE=0 \
  bash scripts/run_cifar10dvs_classification_benchmark.sh
```

## Full Run

```bash
cd ~/CS59-1-event-representation-benchmark
DATA_ROOT=$PWD/data/cifar10dvs RUN_TAG=20260515_cifar10dvs_gpu_full_aligned \
  bash scripts/run_cifar10dvs_classification_benchmark.sh
```

## Small Result Archive

This archives metrics only and excludes checkpoints:

```bash
cd ~/CS59-1-event-representation-benchmark
tar -czf cifar10dvs_classification_results_only.tar.gz \
  outputs/learning_classification/cifar10dvs/20260515_cifar10dvs_gpu_full_aligned/results \
  outputs/traditional/classification/cifar10dvs/20260515_cifar10dvs_gpu_full_aligned
```

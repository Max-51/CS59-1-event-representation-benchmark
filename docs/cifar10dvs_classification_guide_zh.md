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

If the uploaded archive has another name, rename or symlink it to
`CIFAR10DVS.zip` before launching the benchmark. Tonic will reuse this local
zip instead of downloading again.

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
DATA_ROOT=$PWD/data/cifar10dvs RUN_TAG=smoke_cifar10dvs \
  EPOCHS=1 PATIENCE=0 TRAIN_LIMIT=256 TEST_LIMIT=128 \
  LEARNING_METHODS="est" TRADITIONAL_METHODS="event_frame" \
  bash scripts/run_cifar10dvs_classification_benchmark.sh
```

After the smoke test succeeds, remove the limits and method filters for the
full run.

## Full Run

```bash
cd ~/CS59-1-event-representation-benchmark
DATA_ROOT=$PWD/data/cifar10dvs RUN_TAG=20260515_cifar10dvs_gpu_full_aligned \
  bash scripts/run_cifar10dvs_classification_benchmark.sh
```

Useful environment overrides:

```bash
DEVICE=cuda:0
NUM_WORKERS=8
BATCH_SIZE=32
RESUME=1
LEARNING_METHODS="est ergo event_pretraining matrix_lstm evrepsl get omnievent"
TRADITIONAL_METHODS="event_frame binary_event_image timestamp_image time_surface voxel_grid"
```

## Small Result Archive

This archives metrics only and excludes checkpoints:

```bash
cd ~/CS59-1-event-representation-benchmark
tar -czf cifar10dvs_classification_results_only.tar.gz \
  artifacts/classification/learning/cifar10dvs/20260515_cifar10dvs_gpu_full_aligned/results \
  artifacts/classification/traditional/cifar10dvs/20260515_cifar10dvs_gpu_full_aligned
```

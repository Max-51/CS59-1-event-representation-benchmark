# Traditional 方法总索引（任务化重构版）

## 1) 核心实现

- `src/representations/traditional/representation.py`
- `src/representations/traditional/__init__.py`

方法名：

- `event_frame`
- `event_count`
- `binary_event_image`
- `timestamp_image`
- `time_surface`
- `voxel_grid`

## 2) 三条任务入口

### A. Classification

- `tasks/classification/scripts/train_traditional_classification.py`
- `tasks/classification/configs/traditional_nmnist.yaml`
- `tasks/classification/configs/traditional_ncaltech101.yaml`

### B. GEN1 Detection

- `tasks/detection/scripts/train_gen1_detection.py --method <traditional_method>`
- `tasks/detection/scripts/run_all_gen1_methods.py`

### C. MVSEC Optical Flow

- `tasks/optical_flow/src/mvsec_benchmark/adapters/traditional.py`
- `tasks/optical_flow/scripts/run_original_protocol.py`
- `tasks/optical_flow/scripts/run_mvsec_100e_all_early_stop.sh`

## 3) 结果目录（新规范）

- 分类：`artifacts/classification/traditional/`
- 检测：`artifacts/detection/gen1/`
- 光流：`artifacts/optical_flow/mvsec/`
- 跨任务报告：`artifacts/cross_task_reports/latest/`
- 历史：`artifacts/archive/`

## 4) 兼容层

旧路径保留兼容：

- `train_traditional_classification.py`
- `optical-flow/`（软链接）
- `artifacts/traditional_classification`（软链接）

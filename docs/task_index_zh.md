# 仓库任务索引（2026-05-20 重构版）

本仓库按三条任务线组织：

1. 分类（Classification）
2. 目标识别（GEN1 Detection）
3. 光流（MVSEC Optical Flow）

## 1) 分类

- 目录：`tasks/classification/`
- 学习方法入口：`tasks/classification/scripts/train_classification.py`
- EST 端到端入口：`tasks/classification/scripts/train_est_e2e_classification.py`
- 传统方法入口：`tasks/classification/scripts/train_traditional_classification.py`
- CIFAR10-DVS 批量脚本：`tasks/classification/scripts/run_cifar10dvs_classification_benchmark.sh`
- 配置：`tasks/classification/configs/`
- 结果：`artifacts/classification/learning/`、`artifacts/classification/traditional/`

## 2) 目标识别（GEN1）

- 目录：`tasks/detection/`
- 单方法训练：`tasks/detection/scripts/train_gen1_detection.py`
- 全方法批跑：`tasks/detection/scripts/run_all_gen1_methods.py`
- 汇总：`tasks/detection/scripts/summarize_gen1_results.py`
- 预处理索引：`tasks/detection/scripts/build_gen1_window_index.py`
- 配置：`tasks/detection/configs/yolov6n_gen1.py`
- 结果：`artifacts/detection/gen1/`

## 3) 光流（MVSEC）

- 目录：`tasks/optical_flow/`
- 对齐检查：`tasks/optical_flow/scripts/check_mvsec_alignment.py`
- 主跑脚本：`tasks/optical_flow/scripts/run_mvsec_100e_all_early_stop.sh`
- 汇总构建：`tasks/optical_flow/scripts/build_mvsec_e100_outputs.py`
- 结果：`artifacts/optical_flow/mvsec/`

## 4) 兼容层（旧命令仍可用）

以下旧入口仍可执行，会自动转到新目录：

- `train_classification.py`
- `train_est_e2e_classification.py`
- `train_traditional_classification.py`
- `train_gen1_detection.py`
- `run_all_gen1_methods.py`
- `summarize_gen1_results.py`
- `scripts/run_cifar10dvs_classification_benchmark.sh`
- `scripts/build_gen1_window_index.py`

## 5) 历史与归档

- 跨任务最新报告：`artifacts/cross_task_reports/latest/`
- 历史报告：`artifacts/archive/`
- `misc/`：非三任务核心目录（如 `paper_overleaf`、`metadata`）

# Traditional 方法总索引（仓库整理版）

这份文档用于把仓库中所有与 Traditional event representation 相关的代码、脚本、结果、文档统一收口，减少“到处找文件”的成本。

## 1) 核心代码（唯一实现入口）

- 表示实现（统一注册）  
  `src/representations/traditional/representation.py`  
  `src/representations/traditional/__init__.py`

- 方法名（统一）  
  `event_frame` / `event_count` / `binary_event_image` / `timestamp_image` / `time_surface` / `voxel_grid`

## 2) 三条任务线入口

### A. Classification

- 训练入口（Traditional）  
  `train_traditional_classification.py`
- 传统配置模板  
  `configs/traditional_nmnist.yaml`  
  `configs/traditional_ncaltech101.yaml`
- 关键测试  
  `tests/test_traditional_representations.py`  
  `tests/test_traditional_classification_helpers.py`

### B. GEN1 Detection

- 训练入口（learning + traditional 统一入口）  
  `train_gen1_detection.py`
- 表示工厂（含 traditional 分支）  
  `src/detection/gen1_representations.py`
- 检测基准流水线  
  `src/detection/gen1_benchmark.py`  
  `src/detection/gen1_yolov6.py`
- 批量运行与汇总  
  `run_all_gen1_methods.py`  
  `summarize_gen1_results.py`

### C. MVSEC Optical Flow

- traditional adapter  
  `optical-flow/src/mvsec_benchmark/adapters/traditional.py`
- 统一跑数入口  
  `optical-flow/scripts/run_original_protocol.py`  
  `optical-flow/scripts/run_mvsec_100e_all_early_stop.sh`
- 对齐检查与结果汇总  
  `optical-flow/scripts/check_mvsec_alignment.py`  
  `optical-flow/scripts/build_mvsec_e100_outputs.py`

## 3) 结果目录（建议作为“规范位置”）

### 已归档的传统分类总结果

- N-MNIST / N-Caltech101 汇总  
  `artifacts/traditional_classification/`
- CIFAR10-DVS（2026-05-19 对齐版）  
  `artifacts/traditional_classification/cifar10dvs/20260519_cifar10dvs_aligned_cls_v2_traditional/`

### 传统跨任务分析（报告与图表）

- 当前主报告（float64 + timestamp 对齐）  
  `artifacts/traditional_baseline_analysis/20260516_float64/`
- 历史快照（可追溯，但非当前主交付）  
  `artifacts/traditional_baseline_analysis/20260508_full_aligned/`

## 4) 目录使用规范（后续统一）

1. 新实验的结果放在：
   - `artifacts/traditional_classification/<dataset>/<run_tag>/...`
2. `outputs/` 仅用于训练机临时产物，不进 Git。
3. 提交结果时优先保留轻量文件：
   - `config.json`
   - `history.jsonl`
   - `metrics.json`
   - `progress.json`
   - `representation_stats.json`
   - `result.json`
   - `train.log`
4. checkpoint（`*.pt/*.pth`）不进 Git。

## 5) 当前仓库中与 Traditional 相关的高频入口

- 总说明（中文）  
  `docs/traditional_baseline_guide_zh.md`
- CIFAR10-DVS 分类流程说明  
  `docs/cifar10dvs_classification_guide_zh.md`
- 项目总览  
  `README.md`

---

如果后续要继续“物理重构目录”（例如把 legacy 分析目录迁移到 `archive/`），建议单独开一次结构重构提交，避免与实验结果提交混在同一个 commit 里。

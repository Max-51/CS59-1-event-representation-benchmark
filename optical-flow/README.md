# MVSEC Optical Flow Benchmark

This folder contains the optical-flow part of the COMP5703 benchmark group work.
It implements a unified downstream benchmark on MVSEC for six runnable event
representations, all evaluated with the same EVFlowNet-like decoder.

This is an adapted reproduction benchmark. It is not a bit-for-bit rerun of
each paper's original optical-flow decoder, training stack, or private
downstream head.

## Current Status

- Dataset: MVSEC optical-flow sequences.
- Train split: `outdoor_day1 + outdoor_day2`.
- Test split: `indoor_flying1/2/3`.
- Runnable methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`.
- Reported-only method: OmniEvent.
- Decoder: shared `EVFlowNetLike`.
- Training: max 100 epochs, early-stop patience 10.
- Validation: block-random validation sampled from outdoor training windows.
- Metrics: AEE/EPE and outlier percentage.
- Result state: main experiment completed for all six runnable methods.

## Main Result

Protocol: train on `outdoor_day1 + outdoor_day2`, evaluate on
`indoor_flying1/2/3`, event window 6M, full generated GT flow frames, max 100
epochs, early-stop patience 10, block-random validation from outdoor train set.

Lower is better for AEE and Outlier %.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 | 16329 | 3583 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 | 16329 | 3583 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 | 16329 | 3583 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 | 16329 | 3583 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 | 16329 | 3583 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 | 16329 | 3583 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper | paper | paper |

Among the six local runnable methods, the best test AEE is EST at `2.8654`,
with `37.04%` outliers. The local method gaps are small, so the main claim
should be that the six runnable representations are compared under one
consistent downstream protocol.

✳ OmniEvent is added as a paper-reported reference row, not a local run in this
pipeline. The displayed AEE and Outlier values are simple averages over
OmniEvent paper Table 2 results on `indoor_flying1/2/3`
([arXiv:2508.01842](https://arxiv.org/abs/2508.01842)).

## Result Files

- Summary table: `results/summary/mvsec_e100_earlystop_summary.csv`
- Markdown summary: `results/summary/mvsec_e100_earlystop_summary.md`
- AEE figure: `results/figures/mvsec_e100_earlystop_aee.svg`
- Outlier figure: `results/figures/mvsec_e100_earlystop_outlier.svg`
- Train-loss curve: `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- Validation-AEE curve: `results/figures/mvsec_e100_earlystop_val_curve.svg`
- Per-method JSON/log/curve artifacts: `artifacts/e100_earlystop_20260501/`

## Reporting Notes

- Describe this as a unified downstream optical-flow benchmark / adapted
  reproduction.
- The six runnable event representations use the same EVFlowNet-like decoder
  and the same train/eval split.
- Event windows are paired to flow frames by fixed window order/index. This is
  consistent across methods, but it is not strict timestamp interpolation.
- Do not directly compare these numbers as official-paper reproduction numbers,
  because several papers do not release the same optical-flow downstream code.
- Treat `OmniEvent✳` as reported-only. It is useful for context, but it is not
  directly comparable to the six local runs because the exact pipeline and
  evaluation details differ.
- W&B hooks exist in the code, but this result package relies on local CSV
  curves and SVG figures.

## Code Layout

```text
optical-flow/
├── configs/
│   └── envs/
├── docs/
├── results/
│   ├── figures/
│   └── summary/
├── scripts/
├── src/
│   └── mvsec_benchmark/
└── tests/
```

Key entry points:

- `scripts/run_original_protocol.py`
- `src/mvsec_benchmark/pipeline.py`
- `src/mvsec_benchmark/models/evflownet_like.py`
- `docs/PROJECT_STATUS.md`
- `docs/MVSEC_E100_EARLYSTOP_RESULTS_20260501.md`

## Local Smoke Test

The smoke tests use synthetic/mock data and do not require MVSEC downloads.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
python scripts/run_linear_suite.py
```

## Rerun Note

The main experiment has already completed. Only rerun it if the group needs a
new baseline, a W&B dashboard, multi-seed repeats, or a revised timestamp-aligned
event-flow pairing protocol.

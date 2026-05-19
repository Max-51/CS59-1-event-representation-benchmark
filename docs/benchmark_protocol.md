# Benchmark Protocol Notes

## Classification

- Datasets: `nmnist`, `ncaltech101`, `cifar10dvs`
- Learning-based and traditional methods are evaluated with aligned training parameters per dataset.
- Default artifact locations:
  - Learning: `artifacts/classification/learning/...`
  - Traditional: `artifacts/classification/traditional/...`

## Detection (GEN1)

- Entrypoint: `tasks/detection/scripts/run_all_gen1_methods.py`
- Shared preprocessing window index with fixed `window_us`.
- Early stopping and metric selection configured in CLI args.

## Optical Flow (MVSEC)

- Entrypoints under `tasks/optical_flow/scripts/`.
- Formal float64 + timestamp-aligned run artifacts:
  `artifacts/optical_flow/mvsec/20260516_float64_cached/`.

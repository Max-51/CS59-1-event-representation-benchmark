# Classification Task

## Entrypoints

- Learning-based: `tasks/classification/scripts/train_classification.py`
- EST end-to-end: `tasks/classification/scripts/train_est_e2e_classification.py`
- Traditional: `tasks/classification/scripts/train_traditional_classification.py`
- CIFAR10-DVS batch runner: `tasks/classification/scripts/run_cifar10dvs_classification_benchmark.sh`

## Datasets

- `nmnist`
- `ncaltech101`
- `cifar10dvs`

Alias compatibility is preserved (`minist`, `ncar101`, `cifa`, `cifar`, `cifar10-dvs`).

## Result Paths

- Learning: `artifacts/classification/learning/<dataset>/<run_tag>/...`
- Traditional: `artifacts/classification/traditional/<dataset>/<run_tag_or_method>/...`

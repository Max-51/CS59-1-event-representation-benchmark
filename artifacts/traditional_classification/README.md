# Traditional Classification Baseline Results

This folder records the traditional event-representation classification
baselines for N-MNIST and N-Caltech101.

The raw datasets and model checkpoints are not committed. The committed files
are lightweight result tables, training histories, figures, and the script used
to regenerate the summaries from local experiment archives.

## Protocol

- Task: event-based classification
- Downstream model: ResNet18
- Datasets:
  - N-MNIST with the official train/test split
  - N-Caltech101 with `data/splits/tonic_split_seed42.json`
- Traditional representations:
  - `event_frame`
  - `binary_event_image`
  - `timestamp_image`
  - `time_surface`
  - `voxel_grid`
- Training budget: up to 100 epochs with validation early stopping

## Files

- `tables/traditional_classification_results.csv`: final metrics for each
  dataset and representation.
- `tables/traditional_classification_history.csv`: per-epoch train/validation
  curves.
- `tables/learning_based_classification_template.csv`: requested schema for
  adding learning-based baseline results.
- `figures/*.png`: quick visual summaries for traditional baselines.
- `build_traditional_classification_report.py`: local helper used to parse the
  downloaded experiment archives and regenerate the CSV/figure files.

## Regenerate Tables and Figures

The helper expects the lightweight experiment archives downloaded from the
training machine:

- `nmnist_traditional_records.tar.gz`
- `ncaltech101_traditional_records.tar.gz`

Example:

```bash
python artifacts/traditional_classification/build_traditional_classification_report.py \
  --input-dir /path/to/downloaded/archives \
  --output-dir artifacts/traditional_classification
```

The plotting helper uses `pandas` and `matplotlib`. They are only needed for
regenerating these result summaries, not for running the benchmark itself.

## Current Results

| Dataset | Method | Best Epoch | Best Val Acc | Test Acc | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| N-MNIST | event_frame | 29 | 0.9850 | 0.9828 | 2365.40 |
| N-MNIST | binary_event_image | 16 | 0.9565 | 0.9506 | 1219.36 |
| N-MNIST | timestamp_image | 6 | 0.9848 | 0.9805 | 786.48 |
| N-MNIST | time_surface | 26 | 0.9898 | 0.9863 | 1859.29 |
| N-MNIST | voxel_grid | 12 | 0.9920 | 0.9908 | 1266.66 |
| N-Caltech101 | event_frame | 13 | 0.4835 | 0.4966 | 1020.28 |
| N-Caltech101 | binary_event_image | 20 | 0.4577 | 0.4868 | 956.50 |
| N-Caltech101 | timestamp_image | 13 | 0.5007 | 0.5293 | 793.65 |
| N-Caltech101 | time_surface | 13 | 0.4921 | 0.5178 | 838.09 |
| N-Caltech101 | voxel_grid | 11 | 0.5007 | 0.4966 | 974.24 |

## Notes

- Checkpoints are intentionally excluded from Git because they are large and
  are not needed for reading the benchmark summary.
- Learning-based results should be added with the schema in
  `tables/learning_based_classification_template.csv` so that traditional and
  learning-based representations can be compared under the same fields.

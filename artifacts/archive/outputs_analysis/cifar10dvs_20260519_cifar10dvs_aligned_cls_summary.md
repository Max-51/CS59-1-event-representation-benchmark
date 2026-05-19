# CIFAR10-DVS Summary (20260519_cifar10dvs_aligned_cls)

- best_learning: est (57.45%)
- best_traditional: event_frame (41.95%)
- gap_learning_minus_traditional: 15.5 pct points

| group | method | best_test_accuracy_pct | epochs_actual | early_stop_triggered |
|---|---:|---:|---:|---:|
| learning | est | 57.45 | 43 | True |
| learning | event_pretraining | 55.75 | 38 | True |
| learning | matrix_lstm | 55.2 | 30 | True |
| learning | get | 54.1 | 18 | True |
| learning | omnievent | 53.95 | 19 | True |
| learning | evrepsl | 52.4 | 18 | True |
| learning | ergo | 45.9 | 22 | True |
| traditional | event_frame | 41.95 | 16 | True |
| traditional | time_surface | 41.1 | 18 | True |
| traditional | timestamp_image | 40.85 | 51 | True |
| traditional | voxel_grid | 39.35 | 14 | True |
| traditional | binary_event_image | 36.8 | 29 | True |

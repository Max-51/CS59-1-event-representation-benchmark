# Traditional vs Learning-based Comparison, 2026-05-16 Float64 Update

This artifact refreshes the cross-task comparison after the MVSEC optical-flow rerun with float64 event timestamps and timestamp-aligned event/flow windows.

## What Changed

- MVSEC now uses `optical-flow/results_float64_cached_20260516/summary_all/mvsec_e100_earlystop_summary.csv`.
- All 11 local MVSEC methods are included: 6 learning-based and 5 traditional.
- The old May 8 MVSEC conclusion is superseded. With corrected float64/timestamp alignment, EST is the best local MVSEC method.
- Classification tables are unchanged from the previously aligned N-MNIST and N-Caltech101 runs.

## Key MVSEC Results

| Group | Best method | AEE | Outlier % |
| --- | --- | ---: | ---: |
| Learning-based | EST | 2.0429 | 23.36 |
| Traditional | Voxel Grid | 2.0759 | 23.85 |
| Overall | EST | 2.0429 | 23.36 |

## Files

- `classification_full_aligned_summary.csv`: existing aligned classification comparison.
- `mvsec_comparison_summary.csv`: latest float64 MVSEC optical-flow comparison.
- `traditional_vs_learning_float64_report_cn.tex`: Chinese report source.
- `traditional_vs_learning_float64_report_cn.pdf`: compiled report.
- `*_float64.png/pdf`: regenerated MVSEC figure.
- `nmnist_*` and `ncaltech101_*`: carried forward for the full cross-task report.

# MVSEC Optical Flow E100 Early-Stop Summary

Protocol: train on `outdoor_day1 + outdoor_day2`, evaluate on `indoor_flying1/2/3`, event input from extracted left-camera events, full generated GT flow frames, max 100 epochs, early-stop patience 10, block-random validation from outdoor train set, timestamp-aligned event/flow windows.

Lower is better for AEE and outlier percentage.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ergo | 2.1456 | 23.80 | 76.20 | 67 | 57 | 1.8120 | 149 | 871 |
| est | 2.0429 | 23.36 | 76.64 | 75 | 65 | 1.7453 | 149 | 871 |
| event_pretraining | 2.5401 | 30.67 | 69.33 | 40 | 30 | 1.8607 | 149 | 871 |
| evrepsl | 2.3817 | 27.90 | 72.10 | 91 | 81 | 1.7712 | 149 | 871 |
| get | 2.0452 | 23.30 | 76.70 | 43 | 33 | 1.8448 | 149 | 871 |
| matrixlstm | 2.1857 | 25.20 | 74.80 | 95 | 85 | 1.7706 | 149 | 871 |
| OmniEvent* | 0.9900 | 3.24 | 96.76 | paper | paper | paper | paper | paper |

*OmniEvent is a paper-reported reference row. Its displayed values are simple averages from the OmniEvent paper's MVSEC `indoor_flying1/2/3` results, not a local run in this repository. Source: arXiv:2508.01842, Table 2.

## Reporting Notes

- This is a unified downstream optical-flow benchmark / adapted reproduction, not an exact reimplementation of each paper's original optical-flow decoder.
- Local methods are compared under the same EVFlowNet-like decoder and the same MVSEC train/eval protocol, so the comparison is fair inside this benchmark.
- `OmniEvent*` is included only as reported-only context and should not be ranked as a local result from this pipeline.
- Local CSV curves and SVG figures are generated from the run JSON and curve CSV files.

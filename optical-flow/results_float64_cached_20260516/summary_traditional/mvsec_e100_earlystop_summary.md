# MVSEC Optical Flow E100 Early-Stop Summary

Protocol: train on `outdoor_day1 + outdoor_day2`, evaluate on `indoor_flying1/2/3`, event input from extracted left-camera events, full generated GT flow frames, max 100 epochs, early-stop patience 10, block-random validation from outdoor train set, timestamp-aligned event/flow windows.

Lower is better for AEE and outlier percentage.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| event_frame | 2.2085 | 25.81 | 74.19 | 48 | 38 | 1.8975 | 149 | 871 |
| binary_event_image | 2.1625 | 25.27 | 74.73 | 64 | 54 | 1.8693 | 149 | 871 |
| timestamp_image | 2.1122 | 24.83 | 75.17 | 34 | 24 | 2.1892 | 149 | 871 |
| time_surface | 2.2628 | 26.57 | 73.43 | 46 | 36 | 1.9056 | 149 | 871 |
| voxel_grid | 2.0759 | 23.85 | 76.15 | 87 | 77 | 1.7799 | 149 | 871 |
## Reporting Notes

- This is a unified downstream optical-flow benchmark / adapted reproduction, not an exact reimplementation of each paper's original optical-flow decoder.
- Local methods are compared under the same EVFlowNet-like decoder and the same MVSEC train/eval protocol, so the comparison is fair inside this benchmark.
- Local CSV curves and SVG figures are generated from the run JSON and curve CSV files.

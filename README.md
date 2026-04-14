# Event Representation Benchmark (CS59)

## Overview
This project focuses on benchmarking learning-based event representations for event camera data.

Currently included methods:
- EST
- ERGO
- GET
- Matrix-LSTM
- EvRepSL

Target tasks:
- Classification
- Optical Flow
- Object Detection

## Included Papers

| Method | Paper | Venue | Year | Code |
|--------|------|------|------|------|
| EST | End-to-End Learning of Representations for Asynchronous Event-Based Data | ICCV | 2019 | https://github.com/uzh-rpg/rpg_event_representation_learning |
| ERGO | From Chaos Comes Order: Ordering Event Representations for Object Recognition and Detection | ICCV | 2023 | https://github.com/uzh-rpg/event_representation_study |
| GET | Group Event Transformer for Event-Based Vision | ICCV | 2023 | https://github.com/Peterande/GET-Group-Event-Transformer |
| Matrix-LSTM | A Differentiable Recurrent Surface for Asynchronous Event-Based Data | ECCV | 2020 | https://github.com/marcocannici/matrixlstm |
| EvRepSL | Event-stream Representation via Self-Supervised Learning for Event-Based Vision | TIP | 2024 | https://github.com/VincentQQu/EvRepSL |

## Project Scope
We focus on learning-based representations due to their flexibility, adaptability, and potential for end-to-end optimization.

## Repository Structure
- `docs/`: documentation and paper summaries
- `metadata/`: structured information about papers and datasets
- `configs/`: experiment configurations
- `src/`: core benchmark framework
- `scripts/`: training, evaluation, and preprocessing scripts
- `data/`: dataset organization
- `results/`: experiment outputs

## Current Status
- Repository structure initialized
- Paper analysis in progress
- Benchmark implementation upcoming

# Event Representation Benchmark (CS59)

## Overview
This project focuses on benchmarking **learning-based event representations** for event camera data.

## Currently Included Methods
- EST  
- ERGO  
- GET  
- Matrix-LSTM  
- EvRepSL  
- OmniEvent  
- Event Pre-training  

## Target Tasks
- Classification  
- Optical Flow  
- Object Detection  

## Included Papers

| Method | Paper | Venue | Year | Code |
|--------|------|-------|------|------|
| EST | [End-to-End Learning of Representations for Asynchronous Event-Based Data](https://arxiv.org/abs/1904.08245) | ICCV | 2019 | https://github.com/uzh-rpg/rpg_event_representation_learning |
| ERGO | [From Chaos Comes Order: Ordering Event Representations for Object Recognition and Detection](https://arxiv.org/abs/2310.02642) | ICCV | 2023 | https://github.com/uzh-rpg/event_representation_study |
| GET | [Group Event Transformer for Event-Based Vision](https://arxiv.org/abs/2304.13455) | ICCV | 2023 | https://github.com/Peterande/GET-Group-Event-Transformer |
| Matrix-LSTM | [A Differentiable Recurrent Surface for Asynchronous Event-Based Data](https://arxiv.org/abs/2001.03455) | ECCV | 2020 | https://github.com/marcocannici/matrixlstm |
| EvRepSL | [Event-stream Representation via Self-supervised Learning for Event-Based Vision](https://arxiv.org/abs/2412.07080) | TIP | 2024 | https://github.com/VincentQQu/EvRepSL |
| OmniEvent | [OmniEvent: Unified Event Representation Learning](https://arxiv.org/abs/2508.01842) | AAAI | 2026 | - |
| Event Pre-training | [Event Camera Data Pre-training](https://arxiv.org/abs/2301.01928) | ICCV | 2023 | https://github.com/Yan98/Event-Camera-Data-Pre-training |

## Project Scope
We focus on learning-based representations due to their flexibility, adaptability, and potential for end-to-end optimization.

## Repository Structure

- **data/**
  - dataset organization

- **docs/**
  - documentation and paper summaries

- **metadata/papers/**
  - structured metadata for included papers

- **src/**
  - `datasets/` : dataset interfaces
  - `representations/` : representation wrappers and registry
  - `tasks/` : task interfaces

- **third_party/**
  - external reference implementations (Git submodules)

- **Root Files**
  - `.gitmodules` : submodule configuration
  - `environment.yml` : environment setup (to be completed)
  - `requirements.txt` : dependency list (placeholder)
  - `run_benchmark.py` : benchmark entry script
  - `test_registry.py` : registry testing script
  - `test_evrepsl_local.py` : local EvRepSL testing script

## Current Status
- Repository structure initialized  
- Paper analysis in progress  
- Benchmark implementation upcoming  

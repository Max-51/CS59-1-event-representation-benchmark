# GET Notes

## Paper
Group Event Transformer for Event-Based Vision  
Paper: https://arxiv.org/abs/2310.02642  
Code: https://github.com/Peterande/GET-Group-Event-Transformer

## Core Idea
Represent events as grouped tokens and process with transformer-based architecture.

## Representation
Group Event Tokens

## Tasks
- Classification
- Object Detection

## Datasets
- CIFAR10-DVS
- N-MNIST
- Gen1

## Strengths
- Strong global modeling
- Captures spatial + temporal + polarity info

## Limitations
- High computational cost
- Strong dependency on transformer backbone

## Reproduction Notes
- Hard to unify with CNN-based methods
- May require separate evaluation group

## TODO
- Decide fair comparison strategy
- Measure efficiency impact
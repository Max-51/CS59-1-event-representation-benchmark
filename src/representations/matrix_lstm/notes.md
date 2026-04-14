# Matrix-LSTM Notes

## Paper
A Differentiable Recurrent Surface for Asynchronous Event-Based Data  
Paper: https://arxiv.org/abs/2001.03455  
Code: https://github.com/marcocannici/matrixlstm

## Core Idea
Use recurrent neural networks to build continuous-time event representations.

## Representation
Recurrent Surface Representation

## Tasks
- Classification
- Optical Flow

## Datasets
- N-Caltech101
- MVSEC

## Strengths
- Strong temporal modeling
- Works well for dynamic tasks

## Limitations
- Slow (sequential processing)
- Hard to parallelize

## Reproduction Notes
- May be expensive to run
- Need efficient batching strategy

## TODO
- Optimize runtime
- Compare with EST temporal bins
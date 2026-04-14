# ERGO Notes

## Paper
From Chaos Comes Order: Ordering Event Representations for Object Recognition and Detection  
Paper: https://arxiv.org/abs/2304.13455  
Code: https://github.com/uzh-rpg/event_representation_study

## Core Idea
Optimize event representation by learning ordering and structure using Gromov-Wasserstein distance.

## Representation
Ordered Event Representation

## Tasks
- Classification
- Object Detection

## Datasets
- N-Caltech101
- Gen1

## Strengths
- Improves temporal consistency
- Better structure for detection tasks

## Limitations
- High computational cost (GWD)
- Not purely end-to-end

## Reproduction Notes
- Need to understand ordering pipeline
- May not integrate directly with unified backbone

## TODO
- Simplify representation for benchmark
- Compare ordered vs unordered versions
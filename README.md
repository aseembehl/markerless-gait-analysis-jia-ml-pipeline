# Markerless Gait Analysis for Children and Adolescents with Juvenile Idiopathic Arthritis Using a Machine Learning Pipeline

This repository contains code for the method described in the paper, "Markerless Gait Analysis for Children and Adolescents with Juvenile Idiopathic Arthritis Using a Machine Learning Pipeline."

## Overview

The project implements a markerless gait analysis pipeline for sagittal-plane gait videos. The goal is to estimate anatomical landmark positions, derive joint kinematics, and reconstruct gait cycle trajectories from video data.

## Method

The pipeline consists of the following stages:

1. A Mask R-CNN-based detector generates a bounding box around the body in each frame of a sagittal-plane gait video.
2. Within the detected bounding box, a TCFormer model identifies the positions of predefined anatomical landmarks.
3. Joint angles are computed from the detected landmarks on a frame-by-frame basis.
4. The resulting angle sequences are assembled into continuous gait cycle trajectories.

## Getting Started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/aseembehl/markerless-gait-analysis-jia-ml-pipeline.git
cd markerless-gait-analysis-jia-ml-pipeline
pip install -e '.[video,models]'

jia-gait run path/to/video.mp4 -o results/child01
```

The output folder contains the gait-cycle plots for hip, knee and ankle per walking direction (`plots/`), a quality report (`quality.txt`) and the landmarks, joint angles and gait cycles as data files. See [docs/usage.md](docs/usage.md) for installation options, all commands, the output files and configuration.

## Paper

**Title:** Markerless Gait Analysis for Children and Adolescents with Juvenile Idiopathic Arthritis Using a Machine Learning Pipeline

## Repository Purpose

This repository is intended to host the code used to implement the machine learning pipeline described in the paper, including preprocessing, landmark detection, joint angle computation, and gait computation.

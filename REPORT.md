# snn-gesture-recognition-genx320 — Report

- [Overview](#overview)
- [Project Progression](#project-progression)
  - [1. Initial](#1-initial)
  - [2. Split by Time](#2-split-by-time)
  - [3. Ceiling Ref](#3-ceiling-ref)
  - [4. Event Count](#4-event-count)
  - [5. Recording](#5-recording)
  - [6. Event10k](#6-event10k)
  - [7. Finetuning](#7-finetuning)
  - [8. Latency](#8-latency)
  - [9. Environment Pi](#9-environment-pi)
  - [99. Accuracies / Domain Gap](#99-accuracies--domain-gap)
- [Key Findings / Results](#key-findings--results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Notes](#notes)

## Overview

This project trains and deploys a spiking neural network (SNN) for event-based hand gesture recognition, using a Prophesee GenX320 event camera and a Raspberry Pi 5. 

The model is built with [SpikingJelly](https://github.com/fangwei123456/spikingjelly), base-trained on the DVS128Gesture dataset, then fine-tuned on custom recordings collected with the GenX320.

This report walks through the project chronologically, stage by stage (matching the folders in `project-history/`), then summarizes the key results, known limitations, and open directions for future work.

## Project Progression
### 1. Initial
*See [`project-history/1-Initial`](../project-history/1-Initial) for code.*

### 2. Split by Time
*See [`project-history/2-Split_by_Time`](../project-history/2-Split_by_Time) for code.*

### 3. Ceiling Ref
*See [`project-history/3-Ceiling_Ref`](../project-history/3-Ceiling_Ref) for code.*

### 4. Event Count
*See [`project-history/4-EventCount`](../project-history/4-EventCount) for code.*

### 5. Recording
*See [`project-history/5-Recording`](../project-history/5-Recording) for code.*

### 6. Event10k
*See [`project-history/6-Event10k`](../project-history/6-Event10k) for code.*

### 7. Finetuning
*See [`project-history/7-Finetuning`](../project-history/7-Finetuning) for code.*

### 8. Latency
*See [`project-history/8-Latency`](../project-history/8-Latency) for code.*

### 9. Environment Pi
*See [`project-history/9-EnvironmentPi`](../project-history/9-EnvironmentPi) for code.*

### 99. Accuracies / Domain Gap
*See [`project-history/99-Accuracies_DomainGap`](../project-history/99-Accuracies_DomainGap) for code.*

## Key Findings / Results

## Limitations

## Future Work

## Notes


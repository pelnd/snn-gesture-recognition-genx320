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
### Preparation

Before building a model, getting familiar with the framework and setting up the environment was crucial. The first stage of the project consisted of a literature review on SNNs and event-based data processing, followed by setting up the Raspberry Pi–Prophesee GenX320 connection via the Metavision SDK and preparing a virtual environment. The Raspberry Pi used was a Raspberry Pi 5 Model B, running Python 3.11.2 and was set up using Prophesee's custom Linux image (based on Raspberry Pi OS Bookworm), which comes with the RPi sensor driver and OpenEB precompiled. SpikingJelly was chosen as the SNN framework, with PyTorch (2.13.0) and NumPy (1.26.4) installed at the versions recommended by SpikingJelly's documentation. OpenCV (4.5.5.64) was added for monitoring/visualization.

### 1. Initial Model ([1-test-model](project-history/1-test-model))
This is a tiny model along with the first working end-to-end pipeline, meant to establish training, evaluation, and live inference before scaling up to the actual training. 

#### Training
SpikingJelly's DVSGestureNet on DVS128Gesture was trained with reduced settings to keep it fast: 32 channels, 8 timesteps, batch size 2, 2 epochs. An event-count based windowing approach was preferred for robustness between DVS128 (used for base training) and GenX320 (the actual deployment camera), since fixed-time windows would capture very different numbers of events on each sensor. 

#### Evaluation
For this initial model two evaluation approaches were used: first a static evaluation against ready-made frames from the DVS128Gesture test set, and later a streaming evaluation that replays raw events through the same count-based windowing used for live inference. Only the streaming evaluation was used for the following stages.

**Static evaluation** (full DVS128Gesture test set): 63.5% overall accuracy (183/288). Some classes did well (right hand wave 100%, left hand wave 96%), others poorly (right arm clockwise 8%, air drums 33%), which is expected for a 2-epoch test run.

**Streaming (count-based windowing) evaluation** reproduced the exact same accuracy and confusion matrix as the static evaluation (183/288, identical per-cell), confirming the streaming windowing matches the offline run exactly. It also showed that frame construction (the event-to-frame loop) took ~704ms on average, far more than the ~145ms net inference, a sign that event-accumulation, not compute, would be the real-time bottleneck later in the project.

#### Live Inference Pipeline
A live inference pipeline was built to run on the Raspberry Pi: collecting event data from the GenX320 sensor, binning it into frames, and using the trained model weights to classify gestures.

The first version (`pipeline_basic.py`) was a bare-minimum loop: open the camera, accumulate events into count-based frames, run inference, print the predicted class. This version performed poorly for a couple of reasons. The center-crop technique it used was flawed — discarding everything outside a fixed 128×128 window of the sensor's 320×320 output. The event-count mismatch between sensors was also huge: EVENTS_PER_FRAME was first set to 11,000, based on an average from DVS128 training clips, but on the GenX320 sensor that barely supplied any information, causing the model to collapse to predicting a single class.

`pipeline_final.py` was built as an improved version. Event count was raised to 30,000, which performed noticeably better, at least producing varied predictions, even though accuracy remained low as expected. Center-crop was abandoned in favor of downscaling the full sensor frame instead. Diagnostics such as occ (occupancy, the fraction of pixels with any events), tot (average total events per frame), and per-window timing, along with a live cv2 visualization showing predicted class names, were also added.
Camera bias was also introduced as a controllable variable, but the value initially used (-80) was later found to fail on this hardware, silently falling back to a prevailing default bias instead. The value was later fixed at 25/28, not because it improved accuracy, but because that happened to be the setting accidentally used to record some of the training clips, so keeping it kept the live pipeline consistent with the data.


In summary, this stage established the full pipeline end-to-end — training, evaluation, and live inference — and surfaced the core problem that would shape the rest of the project: DVS128 and GenX320 differ enormously in event density, so windowing choices tuned for one sensor don't transfer to the other. Resolving that mismatch became the focus of the following stages.


### 2. Training Experiments ([2-training-tests](project-history/2-training-tests))

Building on the initial model from stage 1, this stage tests different windowing strategies and training variations before deciding on a model to carry forward. The tests include two windowing approaches: event-count-based (frames built from a fixed or per-recording-adaptive number of events) and time-based (frames built from fixed real-time durations). Event-count windowing was also tested with variations to compare accuracy.



#### [Event-count base](project-history/02-training-tests/01-eventcount-base)
This run continues directly from the initial model in stage 1, using the same event-count windowing (split_by='number') but scaling up the settings: 64 channels, T=8, batch size 16, 64 epochs.   
It reached 90.62% accuracy (261/288) on the DVS128Gesture test set. Air drums and air guitar were the weakest classes, most often confused with each other and with "other gestures." This left room for improvement, so a few variations were explored next to see if accuracy could be pushed higher.

#### [Weight decay](project-history/02-training-tests/02-eventcount-with-decaying-weight)
Same base config, but with weight decay (1e-4) added to the optimizer, a small penalty that shrinks the weights slightly on every step, to see if it would help avoid overfitting.  
It didn't. Accuracy dropped to 84.38% (243/288), worse across most classes than the base run. Weight decay was dropped from later runs.

#### [Jitter augmentation](project-history/02-training-tests/03-eventcount-jitter)
Same base config, with a random per-sample pixel shift (±4px, same shift applied across all T frames) added as augmentation.  
Accuracy rose to 92.71% (267/288), which was an improvement from previous runs, but still short of satisfying.

#### [Fixed-count sliding window (10k)](project-history/02-training-tests/04-eventcount-sliding-10k)
Event-count windowing was revisited as a solution to the density mismatch. As expected, it did better than time-windowed on live deployment, but that still was not enough to resolve the density mismatch issue.  

The problem stemmed from something more fundamental: SpikingJelly's split_by='number' divided each recording into frames based on that recording's own total event count, so frame density still varied recording to recording during training, while live prediction always read a fixed absolute number of events per frame. To better match training with live inference, a new approach was tested: building frames from a fixed count of events instead during training, the same mechanism the live GenX320 pipeline uses.  

This approach required deciding on a fixed event count before training. Measuring events-per-frame per gesture class showed density varies about 3.8x across classes (right arm counter-clockwise highest, hand clap lowest). 10,000 events was chosen because it was close to the lowest classes' 10th-percentile density, keeping every class represented even after the fixed cutoff, at the cost of hand clap losing more samples than the rest.

This approach reached 94.57% accuracy (1,255/1,327 windows) on the test dataset, the best of the event-count runs. It is worth noting this approach supplied more windows per clip, possibly inflating the accuracy. Still, it's a solid number, and more importantly, its windowing already matched how the live pipeline builds frames, for a better match with deployment overall.  

Two things stayed unresolved here: hand clap's slightly lower accuracy in this run traced directly to it losing the most samples to the fixed-count cutoff, being the lowest-density class. Air guitar's confusion with "other gestures," on the other hand, showed up across every windowing method tried and didn't trace to density or sample count, treated as a limitation rather than something to keep chasing.





### 5. Recording ([5-Recording](project-history/5-Recording))

### 7. Finetuning ([7-Finetuning](project-history/7-Finetuning))

### 8. Latency ([8-Latency](project-history/8-Latency))

### 9. Environment Pi ([9-EnvironmentPi](project-history/9-EnvironmentPi))

### 99. Accuracies / Domain Gap ([99-Accuracies_DomainGap](project-history/99-Accuracies_DomainGap))

## Key Findings / Results

## Limitations

## Future Work

## Notes


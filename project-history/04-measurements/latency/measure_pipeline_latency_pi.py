"""
End-to-end LIVE pipeline latency: event capture + windowing + inference, on the Pi.

Reuses app3.py's exact camera/windowing logic (same DISPLAY_N_EVENTS split for
accumulation granularity, same exact-EVENTS_PER_FRAME slicing with carryover) but
strips the display/idle-timeout code and adds timing instead. Measures REAL
prediction-to-prediction latency during live use -- as opposed to
measure_latency_pi.py, which times only the forward pass on pre-recorded clips
(no camera, no event-accumulation wait).

Keep gesturing continuously while this runs -- idle gaps get folded into the
latency numbers as genuine (very large) outliers, same as they would in real use.
"""

import csv
import time

import numpy as np
import torch

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from metavision_core.event_io import EventsIterator

DEVICE = 'cpu'
T = 8
CHANNELS = 64
NUM_CLASSES = 11
H, W = 128, 128
MODEL_PATH = '../../03-finetuning/finetuning-raw/weights-and-logs/checkpoint_latest.pth'
CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise',
    'right arm counter clockwise', 'left arm clockwise',
    'left arm counter clockwise', 'arm rolls',
    'air drums', 'air guitar',
]

EVENTS_PER_FRAME = 50000        # must match fine-tuning windowing exactly
DISPLAY_N_EVENTS = 5000         # accumulation granularity only, same as app3.py -- no display here
DISPLAY_MAX_WAIT_US = 100_000

NATIVE_H, NATIVE_W = 320, 320

N_PREDICTIONS = 16              # how many predictions to collect before stopping -- edit as needed
OUT_CSV = './pipeline_latency_pi.csv'

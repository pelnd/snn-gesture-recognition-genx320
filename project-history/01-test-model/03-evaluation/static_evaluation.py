"""
Evaluates a trained DVSGestureNet checkpoint on the full DVS128Gesture
test set. Reports overall and per-class accuracy, a confusion matrix,
model size/parameter count, and per-sample inference timing (CPU).
"""

import os
import time

import torch
import numpy as np

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = '../../../data/DVS128Gesture'
MODEL_PATH = '../02-weights-and-logs/checkpoint_best.pth'
DEVICE = 'cpu'  
T = 8
CHANNELS = 32
NUM_CLASSES = 11

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise', 
    'right arm counter clockwise', 'left arm clockwise', 
    'left arm counter clockwise', 'arm roll',
    'air drums', 'air guitar', 
]


def main():
    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    functional.set_step_mode(net, 'm')
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

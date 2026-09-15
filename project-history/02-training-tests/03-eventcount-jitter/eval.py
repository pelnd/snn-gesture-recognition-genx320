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

DATA_DIR = './data/DVS128Gesture'
RUN_NAME = 'count_jitter'
MODEL_PATH = './weights-and-logs/checkpoint_best.pth'
DEVICE = 'cpu'
T = 8
CHANNELS = 64
NUM_CLASSES = 11

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise', 
    'right arm counter clockwise', 'left arm clockwise', 
    'left arm counter clockwise', 'arm roll',
    'air drums', 'air guitar', 
]


def main():
    print('=' * 60)
    print('RUN CONFIG')
    print('=' * 60)
    print(f'Run name:  {RUN_NAME}')
    print(f'Channels:  {CHANNELS}')
    print(f'T:         {T}')
    print(f'Windowing: count-based (split_by="number")')

    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    functional.set_step_mode(net, 'm')
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    net.to(DEVICE)
    net.eval()

    num_params = sum(p.numel() for p in net.parameters())        # model complexity
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # model size

    print('=' * 60)
    print('MODEL COMPLEXITY')
    print('=' * 60)
    print(f'Parameters:  {num_params:,}')
    print(f'Model size:  {model_size_mb:.2f} MB (on disk)')

    test_set = DVS128Gesture(DATA_DIR, train=False, data_type='frame', frames_number=T, split_by='number')
    print(f'test samples: {len(test_set)}')

    # confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    inference_times = []

    with torch.no_grad():
        for frame, label in test_set:

            frame = torch.from_numpy(frame).unsqueeze(1).to(DEVICE)  #[T, 1, C, H, W]

            start = time.perf_counter()
            out_firing_rate = net(frame).mean(0)  # shape [1, NUM_CLASSES]
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            pred = out_firing_rate.argmax(1).item()
            confusion[label][pred] += 1

            functional.reset_net(net)

    total_correct = np.trace(confusion)
    total_samples = confusion.sum()
    overall_acc = total_correct / total_samples

    print()
    print('=' * 60)
    print('ACCURACY')
    print('=' * 60)
    print(f'Overall accuracy: {overall_acc:.4f}  ({total_correct}/{total_samples})')
    print()
    print('Per-class accuracy:')
    for i, name in enumerate(CLASS_NAMES):
        class_total = confusion[i].sum()
        if class_total == 0:
            print(f'  {name:30s}: no test samples')
            continue
        class_correct = confusion[i][i]
        print(f'  {name:30s}: {class_correct / class_total:.4f}  '
              f'({class_correct}/{class_total})')

    print()
    print('=' * 60)
    print('CONFUSION MATRIX (rows=true, columns=predicted)')
    print('=' * 60)
    header = '        ' + ''.join(f'{i:>5}' for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = f'true {i:>2} |' + ''.join(f'{confusion[i][j]:>5}' for j in range(NUM_CLASSES))
        print(row)

    inference_times = np.array(inference_times)
    print()
    print('=' * 60)
    print(f'INFERENCE TIME (single sample, {DEVICE})')
    print('=' * 60)
    print(f'Mean:   {inference_times.mean() * 1000:.2f} ms')
    print(f'Median: {np.median(inference_times) * 1000:.2f} ms')
    print(f'Min:    {inference_times.min() * 1000:.2f} ms')
    print(f'Max:    {inference_times.max() * 1000:.2f} ms')


if __name__ == '__main__':
    main()

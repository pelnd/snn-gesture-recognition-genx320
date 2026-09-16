"""
Evaluates a trained DVSGestureNet checkpoint on the DVS128Gesture test set,
using the same time-based windowing as this run's training (125ms frames,
split_by='time', sliced into non-overlapping T=16 windows via
SlidingWindowDataset). Reports overall and per-class accuracy, a confusion
matrix, model size/parameter count, and per-sample inference timing (CPU).
"""

import os
import time
 
import torch
import numpy as np
 
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = './data/DVS128Gesture'
RUN_NAME = 'time_windowing'
MODEL_PATH = './weights-and-logs/checkpoint_best.pth'
DEVICE = 'cpu'  
T = 16
DURATION = 125_000
CHANNELS = 128
NUM_CLASSES = 11

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise', 
    'right arm counter clockwise', 'left arm clockwise', 
    'left arm counter clockwise', 'arm rolls',
    'air drums', 'air guitar', 
]

class SlidingWindowDataset(torch.utils.data.Dataset):
    # Slices Dataset into T-frame windows: same as training, so this
    # eval reproduces training's test_acc exactly before breaking it down
    # per class.

    def __init__(self, base_dataset, T=8, stride=None):
        self.base = base_dataset
        self.T = T
        self.stride = stride if stride is not None else T
        self.index = []  # (recording_idx, start_frame)
        for i in range(len(base_dataset)):
            frames, _ = base_dataset[i]        # [N, C, H, W], N variable
            n = frames.shape[0]
            for start in range(0, n - T + 1, self.stride):
                self.index.append((i, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec_idx, start = self.index[idx]
        frames, label = self.base[rec_idx]
        return frames[start:start + self.T], label


def main():

    print('=' * 60)
    print('RUN CONFIG')
    print('=' * 60)
    print(f'Run name:  {RUN_NAME}')
    print(f'Channels:  {CHANNELS}')
    print(f'T:         {T}')
    print(f'Duration:  {DURATION} us')


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

    test_base = DVS128Gesture(DATA_DIR, train=False, data_type='frame', split_by='time', duration=DURATION)
    test_set = SlidingWindowDataset(test_base, T=T)

    print(f'test samples: {len(test_set)}  (from {len(test_base)} recordings)')  

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

    # printing accuracies
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

    # printing conf matrix
    print()
    print('=' * 60)
    print('CONFUSION MATRIX (rows=true, columns=predicted)')
    print('=' * 60)
    header = '        ' + ''.join(f'{i:>5}' for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = f'true {i:>2} |' + ''.join(f'{confusion[i][j]:>5}' for j in range(NUM_CLASSES))
        print(row)

    # inference time
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

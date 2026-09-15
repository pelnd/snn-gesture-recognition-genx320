"""
Evaluates a trained DVSGestureNet checkpoint on the DVS128Gesture test set,
using the same fixed event-count sliding windows as this run's training
(EventCountWindowDataset, N_EVENTS=10,000). Reports overall and per-class
accuracy, a confusion matrix, model size/parameter count, and per-sample
inference timing (CPU).
"""

import os
import time

import torch
import numpy as np

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = './data/DVS128Gesture'
RUN_NAME = 'eventcount_10k'
MODEL_PATH = './weights-and-logs/checkpoint_best.pth'
DEVICE = 'cpu'  
T = 8
N_EVENTS = 10_000   # must match training's fixed events-per-frame count
CHANNELS = 64
NUM_CLASSES = 11

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise', 
    'right arm counter clockwise', 'left arm clockwise', 
    'left arm counter clockwise', 'arm rolls',
    'air drums', 'air guitar', 
]

def events_to_frame(events, H=128, W=128):
    x = events['x'].astype(np.int64)
    y = events['y'].astype(np.int64)
    p = events['p'].astype(np.int64)
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y, x), 1.0)
    return frame

class EventCountWindowDataset(torch.utils.data.Dataset):
    # Slices each recording's raw events into fixed-n_events frames, then
    # groups into non-overlapping T-frame windows -- fixed absolute event
    # count per frame, same mechanism as training and the live GenX320 script.

    def __init__(self, base_dataset, T=8, n_events=4000):
        self.T = T
        self.n_events = n_events
        self.samples = []  # list of ([T, 2, H, W] array, label)

        for i in range(len(base_dataset)):
            events, label = base_dataset[i]
            n_total = len(events['t'])
            n_frames_available = n_total // n_events
            n_windows = n_frames_available // T

            for w in range(n_windows):
                frames = []
                for t in range(T):
                    start = (w * T + t) * n_events
                    end = start + n_events
                    chunk = {k: events[k][start:end] for k in ('x', 'y', 'p')}
                    frames.append(events_to_frame(chunk))
                self.samples.append((np.stack(frames), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():

    print('=' * 60)
    print('RUN CONFIG')
    print('=' * 60)
    print(f'Run name:  {RUN_NAME}')
    print(f'Channels:  {CHANNELS}')
    print(f'T:         {T}')
    print(f'Windowing: fixed event-count (N_EVENTS={N_EVENTS})')

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

    test_base = DVS128Gesture(DATA_DIR, train=False, data_type='event')
    print('building test windows...')
    test_set = EventCountWindowDataset(test_base, T=T, n_events=N_EVENTS)

    print(f'test samples: {len(test_set)}  (from {len(test_base)} recordings)')

    # confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    inference_times = []

    with torch.no_grad():
        for frame, label in test_set:

            frame = torch.from_numpy(frame).unsqueeze(1).to(DEVICE).float()  #[T, 1, C, H, W]

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

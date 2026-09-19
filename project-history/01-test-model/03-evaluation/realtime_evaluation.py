"""
Evaluates the streaming (count-based windowing) pipeline on the full
DVS128Gesture test set. 

Events are accumulated into T frames of
events_per_frame = len(events) // T, matching training's split_by='number'
windowing. 

Reports overall and per-class accuracy and a confusion matrix,
same as the static eval script, plus two separate latency numbers: frame
construction time and net inference time.
"""

import os
import time

import numpy as np
import torch

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


def events_to_frame(events, H=128, W=128):
    # events: list of (t, x, y, p) tuples -> [2, H, W] frame
    frame = np.zeros((2, H, W), dtype=np.float32)
    for t, x, y, p in events:
        frame[int(p), int(y), int(x)] += 1
    return frame


def stream_predict(net, events, events_per_frame, device=DEVICE):
    """
    Runs one sample's events through count-based windowing and inference,
    timing frame construction and net inference separately.
    Returns (prediction, frame_time, inference_time); prediction is None
    if the sample didn't have enough events to fill T frames.
    """
    frame_buffer = []
    current_events = []
    frame_time_total = 0.0
    inference_time = None
    pred = None

    for t, x, y, p in zip(events['t'], events['x'], events['y'], events['p']):
        current_events.append((int(t), int(x), int(y), int(p)))

        if len(current_events) >= events_per_frame:
            start = time.perf_counter()
            frame = events_to_frame(current_events)
            frame_time_total += time.perf_counter() - start

            frame_buffer.append(frame)
            current_events = []

            if len(frame_buffer) == T:
                x_in = np.stack(frame_buffer)[:, None]  # [T, 1, 2, H, W]
                x_in = torch.from_numpy(x_in).float().to(device)

                start = time.perf_counter()
                with torch.no_grad():
                    out_firing_rate = net(x_in).mean(0)
                inference_time = time.perf_counter() - start

                pred = out_firing_rate.argmax(1).item()
                functional.reset_net(net)
                break  # this sample's math only ever fills one T-buffer, see note above

    return pred, frame_time_total, inference_time


def main():
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

    num_params = sum(p.numel() for p in net.parameters())
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    print('=' * 60)
    print('MODEL COMPLEXITY')
    print('=' * 60)
    print(f'Parameters:  {num_params:,}')
    print(f'Model size:  {model_size_mb:.2f} MB (on disk)')

    test_set = DVS128Gesture(DATA_DIR, train=False, data_type='event')

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    inference_times = []
    frame_construction_times = []
    skipped = 0

    for i in range(len(test_set)):
        events, label = test_set[i]
        events_per_frame = len(events['t']) // T

        if events_per_frame == 0:
            # sample has fewer than T events total, can't form even one frame
            skipped += 1
            continue

        pred, frame_time, inf_time = stream_predict(net, events, events_per_frame)

        if pred is None:
            skipped += 1
            continue

        confusion[label][pred] += 1
        inference_times.append(inf_time)
        frame_construction_times.append(frame_time)

    total_correct = np.trace(confusion)
    total_samples = confusion.sum()
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    print()
    print('=' * 60)
    print('ACCURACY (streaming pipeline)')
    print('=' * 60)
    print(f'Overall accuracy: {overall_acc:.4f}  ({total_correct}/{total_samples})')
    if skipped:
        print(f'Skipped {skipped} sample(s) with too few events to form a full window')
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
    frame_construction_times = np.array(frame_construction_times)

    print()
    print('=' * 60)
    print(f'LATENCY (single sample, {DEVICE})')
    print('=' * 60)
    print('Net forward pass (same measurement as offline eval):')
    print(f'  Mean:   {inference_times.mean() * 1000:.2f} ms')
    print(f'  Median: {np.median(inference_times) * 1000:.2f} ms')
    print(f'  Min:    {inference_times.min() * 1000:.2f} ms')
    print(f'  Max:    {inference_times.max() * 1000:.2f} ms')
    print()
    print('Frame construction (events_to_frame loop, extra cost the offline eval doesn\'t pay):')
    print(f'  Mean:   {frame_construction_times.mean() * 1000:.2f} ms')
    print(f'  Median: {np.median(frame_construction_times) * 1000:.2f} ms')
    print(f'  Min:    {frame_construction_times.min() * 1000:.2f} ms')
    print(f'  Max:    {frame_construction_times.max() * 1000:.2f} ms')
    print()
    total_mean = inference_times.mean() + frame_construction_times.mean()
    print(f'Combined mean (frame construction + inference): {total_mean * 1000:.2f} ms per prediction')


if __name__ == '__main__':
    main()



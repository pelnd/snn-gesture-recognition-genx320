"""
Evaluates the deduped fine-tuned model (finetuning-deduped/train.py,
checkpoint_latest.pth) on the held-out test_clips.txt set -- clips
excluded from fine-tuning.

Same combined dedup + windowing as the dedup fine-tuning run
(N_EVENTS=38,000, T=8), applied only to held-out clips. Reports overall
accuracy, per-class accuracy, and a confusion matrix (rows=true,
cols=predicted).
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

TEST_CLIPS_FILE = '../test_clips.txt'
CHECKPOINT_PATH = './weights-and-logs/checkpoint_latest.pth'

DEVICE = 'cpu'
T = 8
N_EVENTS = 38_000
REFRACTORY_US = 1000  # same value used in the dedup fine-tuning run
BATCH_SIZE = 16
CHANNELS = 64
NUM_CLASSES = 11
H, W = 128, 128
NATIVE_H, NATIVE_W = 320, 320

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm rolls', 'air drums', 'air guitar',
]


def combined_dedup(events):
    # same logic as finetune_genx320_dedup.py -- spatial bin to 128x128, then
    # suppress same-pixel repeats within REFRACTORY_US
    x_bin = (events['x'].astype(np.int64) * W) // NATIVE_W
    y_bin = (events['y'].astype(np.int64) * H) // NATIVE_H
    p = events['p'].astype(np.int64)
    t = events['t'].astype(np.int64)

    pixel_key = (x_bin * H + y_bin) * 2 + p
    order = np.lexsort((t, pixel_key))
    pixel_key_sorted = pixel_key[order]
    t_sorted = t[order]
    x_sorted = x_bin[order]
    y_sorted = y_bin[order]
    p_sorted = p[order]

    same_pixel = pixel_key_sorted[1:] == pixel_key_sorted[:-1]
    dt = t_sorted[1:] - t_sorted[:-1]
    suppressed = same_pixel & (dt < REFRACTORY_US)
    keep_mask = np.ones(len(t_sorted), dtype=bool)
    keep_mask[1:] = ~suppressed

    chrono_order = np.argsort(t_sorted[keep_mask])
    return {
        't': t_sorted[keep_mask][chrono_order],
        'x_bin': x_sorted[keep_mask][chrono_order],
        'y_bin': y_sorted[keep_mask][chrono_order],
        'p': p_sorted[keep_mask][chrono_order],
    }


def frame_from_binned(x_bin, y_bin, p):
    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y_bin, x_bin), 1.0)
    return frame


def load_test_clips(test_clips_file):
    clips = []
    with open(test_clips_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append((row['clip_path'], int(row['class_idx'])))
    return clips


class TestWindowDataset(torch.utils.data.Dataset):
    def __init__(self, test_clips, T=8, n_events=38_000):
        self.samples = []

        for clip_path, class_idx in test_clips:
            raw_events = np.load(clip_path)
            deduped = combined_dedup(raw_events)
            n_total = len(deduped['t'])
            n_frames_available = n_total // n_events
            n_windows = n_frames_available // T

            if n_windows == 0:
                print(f'warning: test clip has 0 usable windows after dedup, skipping: {clip_path}')
                continue

            for w_idx in range(n_windows):
                frames = []
                for t in range(T):
                    start = (w_idx * T + t) * n_events
                    end = start + n_events
                    frames.append(frame_from_binned(
                        deduped['x_bin'][start:end],
                        deduped['y_bin'][start:end],
                        deduped['p'][start:end],
                    ))
                self.samples.append((np.stack(frames), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    functional.set_step_mode(net, 'm')
    net.to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    print(f'loaded checkpoint from {CHECKPOINT_PATH} (epoch={checkpoint.get("epoch")})')

    test_clips = load_test_clips(TEST_CLIPS_FILE)
    print(f'{len(test_clips)} held-out test clips loaded')

    test_set = TestWindowDataset(test_clips, T=T, n_events=N_EVENTS)
    print(f'{len(test_set)} test windows built (after dedup)')

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    per_class_correct = {i: 0 for i in range(NUM_CLASSES)}
    per_class_total = {i: 0 for i in range(NUM_CLASSES)}
    total_correct, total = 0, 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # rows=true, cols=predicted

    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(DEVICE).transpose(0, 1).float()
            label = label.to(DEVICE)

            out_firing_rate = net(frame).mean(0)
            pred = out_firing_rate.argmax(1)

            for p, l in zip(pred.tolist(), label.tolist()):
                per_class_total[l] += 1
                confusion[l, p] += 1
                if p == l:
                    per_class_correct[l] += 1
            total += label.numel()
            total_correct += (pred == label).sum().item()

            functional.reset_net(net)

    print(f'\noverall test accuracy: {total_correct}/{total} = {total_correct/total:.4f}\n')
    print(f'{"class":<28} {"correct":>8} {"total":>8} {"accuracy":>10}')
    for idx in range(NUM_CLASSES):
        c, t = per_class_correct[idx], per_class_total[idx]
        acc_str = f'{c/t:.4f}' if t > 0 else 'n/a'
        print(f'{CLASS_NAMES[idx]:<28} {c:>8} {t:>8} {acc_str:>10}')

    short_names = [n[:4] for n in CLASS_NAMES]
    print('\nconfusion matrix (rows=true, cols=predicted)')
    header = f'{"":<28}' + ''.join(f'{name:>6}' for name in short_names)
    print(header)
    for idx in range(NUM_CLASSES):
        row = f'{CLASS_NAMES[idx]:<28}' + ''.join(f'{confusion[idx, j]:>6}' for j in range(NUM_CLASSES))
        print(row)


if __name__ == '__main__':
    main()

"""
Evaluates the fine-tuned model (finetuning-raw/train.py, checkpoint_latest.pth)
on the held-out test_clips.txt set -- clips excluded from fine-tuning.

Same fixed-count windowing and binning as fine-tuning (N_EVENTS=50,000, T=8),
applied only to held-out clips. Reports overall accuracy, per-class accuracy,
and a confusion matrix (rows=true, cols=predicted).

Supports excluding specific clips via EXCLUDE_CLIPS, for computing a separate
"excluding known-ambiguous clip(s)" number for the report -- leave it empty
for the normal full-set evaluation.
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning'
TEST_CLIPS_FILE = '../test_clips.txt'
CHECKPOINT_PATH = './weights-and-logs/checkpoint_latest.pth'

# Clips to leave out of this run, for reporting purposes only -- leave empty
# for the normal 89.7% evaluation. Fill in only when computing a separate
# "excluding known-ambiguous clip" number for the report.
EXCLUDE_CLIPS = {
    'C_class01_right_hand_wave/clip_002.npy'
}

DEVICE = 'cpu'  # switch to 'cuda' if running on the GPU machine
T = 8
N_EVENTS = 50_000
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


def events_to_frame(x, y, p, h=H, w=W):
    x_bin = (x.astype(np.int64) * w) // NATIVE_W
    y_bin = (y.astype(np.int64) * h) // NATIVE_H
    p = p.astype(np.int64)
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
    frame = np.zeros((2, h, w), dtype=np.float32)
    np.add.at(frame, (p, y_bin, x_bin), 1.0)
    return frame


def load_test_clips(test_clips_file, recordings_dir):
    # returns list of (clip_path, class_idx) -- clip_path rebuilt from
    # test_clips.txt's path (relative to recordings_dir) to a loadable one
    clips = []
    n_excluded = 0
    with open(test_clips_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['clip_path'] in EXCLUDE_CLIPS:
                n_excluded += 1
                continue
            clip_path = os.path.normpath(os.path.join(recordings_dir, row['clip_path']))
            clips.append((clip_path, int(row['class_idx'])))
    if EXCLUDE_CLIPS:
        print(f'excluded {n_excluded}/{n_excluded + len(clips)} listed clips via EXCLUDE_CLIPS')
    return clips


class TestWindowDataset(torch.utils.data.Dataset):
    def __init__(self, test_clips, T=8, n_events=50_000):
        self.samples = []  # (frames [T,2,H,W], label, clip_path) -- keep clip_path for per-clip breakdown

        for clip_path, class_idx in test_clips:
            events = np.load(clip_path)
            n_total = len(events)
            n_frames_available = n_total // n_events
            n_windows = n_frames_available // T

            if n_windows == 0:
                print(f'warning: test clip has 0 usable windows, skipping: {clip_path}')
                continue

            for w_idx in range(n_windows):
                frames = []
                for t in range(T):
                    start = (w_idx * T + t) * n_events
                    end = start + n_events
                    chunk = events[start:end]
                    frames.append(events_to_frame(chunk['x'], chunk['y'], chunk['p']))
                self.samples.append((np.stack(frames), class_idx, clip_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label, clip_path = self.samples[idx]
        return frame, label


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

    test_clips = load_test_clips(TEST_CLIPS_FILE, RECORDINGS_DIR)
    print(f'{len(test_clips)} held-out test clips loaded')

    test_set = TestWindowDataset(test_clips, T=T, n_events=N_EVENTS)
    print(f'{len(test_set)} test windows built')

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

    # confusion matrix -- rows = true class, columns = predicted class
    short_names = [n[:4] for n in CLASS_NAMES]  # abbreviate to keep columns narrow
    print('\nconfusion matrix (rows=true, cols=predicted)')
    header = f'{"":<28}' + ''.join(f'{name:>6}' for name in short_names)
    print(header)
    for idx in range(NUM_CLASSES):
        row = f'{CLASS_NAMES[idx]:<28}' + ''.join(f'{confusion[idx, j]:>6}' for j in range(NUM_CLASSES))
        print(row)


if __name__ == '__main__':
    main()

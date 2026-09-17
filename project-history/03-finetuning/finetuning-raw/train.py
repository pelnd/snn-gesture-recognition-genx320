"""
Fine-tunes SpikingJelly's DVSGestureNet from the pretrained
checkpoint_best_10k.pth (base_train_eventcount_10k) on raw GenX320
recordings. Trains only, no test loop, so only checkpoint_latest.pth 
and a training log (CSV) are saved.

Same fixed-count windowing as the base training run, applied here to raw
GenX320 clips (n_events=50,000, since GenX320 events are denser than
DVS128Gesture's). Clips listed in TEST_CLIPS_FILE are excluded, held out
for a raw-vs-deduped comparison later.

Adapted from SpikingJelly's classify_dvsg.py example
(https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/classify_dvsg.py),
licensed under the Qǐzhì Open Source License 1.0 (see third-party-licenses/).
"""

import os
import csv
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning' 
TEST_CLIPS_FILE = '../test_clips.txt'  # excluded from fine-tuning, held out for raw-vs-dedup comparison
BASE_CHECKPOINT = '../checkpoint_best_10k.pth'

DEVICE = 'cuda'
T = 8
N_EVENTS = 50_000
BATCH_SIZE = 16
CHANNELS = 64
EPOCHS = 16
LR = 1e-4
NUM_CLASSES = 11
RUN_NAME = 'finetune_genx320_raw_50k'  # set manually per run, keeps this separate from base_train_eventcount_10k
CHECKPOINT_DIR = './weights-and-logs'
PRINT_EVERY = 20

NUM_WORKERS = 0
H, W = 128, 128
NATIVE_H, NATIVE_W = 320, 320

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm rolls', 'air drums', 'air guitar',
]


def load_excluded_paths(test_clips_file):
    excluded = set()
    if not os.path.exists(test_clips_file):
        print(f'warning: {test_clips_file} not found, no clips excluded')
        return excluded
    with open(test_clips_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            clip_path = parts[2]
            excluded.add(os.path.normpath(clip_path))
    return excluded


def events_to_frame(x, y, p, h=H, w=W):
    # bin 320x320 native coords down to 128x128 -- same formula as recorder
    # and live inference scripts, must stay identical across all three
    x_bin = (x.astype(np.int64) * w) // NATIVE_W
    y_bin = (y.astype(np.int64) * h) // NATIVE_H
    p = p.astype(np.int64)
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
    frame = np.zeros((2, h, w), dtype=np.float32)
    np.add.at(frame, (p, y_bin, x_bin), 1.0)
    return frame


class GenX320WindowDataset(torch.utils.data.Dataset):
    # same fixed-event-count, non-overlapping T-frame windowing as
    # EventCountWindowDataset (DVS128 base training), applied to raw
    # GenX320 .npy clips instead of DVS128Gesture recordings.

    def __init__(self, recordings_dir, excluded_paths, T=8, n_events=50_000):
        self.T = T
        self.n_events = n_events
        self.samples = []

        class_folders = sorted(glob.glob(os.path.join(recordings_dir, '*_class*_*')))
        for folder in class_folders:
            name = os.path.basename(folder)
            try:
                class_idx = int(name.split('_class')[1].split('_')[0])
            except (IndexError, ValueError):
                continue

            clips = sorted(glob.glob(os.path.join(folder, 'clip_*.npy')))
            for clip_path in clips:
                if os.path.normpath(clip_path) in excluded_paths:
                    continue

                events = np.load(clip_path)
                n_total = len(events)
                n_frames_available = n_total // n_events
                n_windows = n_frames_available // T

                for w_idx in range(n_windows):
                    frames = []
                    for t in range(T):
                        start = (w_idx * T + t) * n_events
                        end = start + n_events
                        chunk = events[start:end]
                        frames.append(events_to_frame(chunk['x'], chunk['y'], chunk['p']))
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

    checkpoint = torch.load(BASE_CHECKPOINT, map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    print(f'loaded pretrained weights from {BASE_CHECKPOINT} '
          f'(base checkpoint epoch={checkpoint.get("epoch")}, max_test_acc={checkpoint.get("max_test_acc")})')

    excluded_paths = load_excluded_paths(TEST_CLIPS_FILE)
    print(f'excluding {len(excluded_paths)} held-out test clips from fine-tuning')

    print('building fine-tuning windows...')
    train_set = GenX320WindowDataset(RECORDINGS_DIR, excluded_paths, T=T, n_events=N_EVENTS)
    print(f'fine-tuning samples: {len(train_set)}')

    class_counts = {}
    for _, label in train_set.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    print('samples per class:', {CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())})

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.csv')
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'train_loss', 'train_acc'])

    for epoch in range(EPOCHS):
        functional.reset_net(net)
        net.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (frame, label) in enumerate(train_loader):
            frame = frame.to(DEVICE).transpose(0, 1).float()
            label = label.to(DEVICE)
            label_onehot = F.one_hot(label, NUM_CLASSES).float()

            optimizer.zero_grad()
            out_firing_rate = net(frame).mean(0)
            loss = F.mse_loss(out_firing_rate, label_onehot)
            loss.backward()
            optimizer.step()

            train_total += label.numel()
            train_loss += loss.item() * label.numel()
            train_correct += (out_firing_rate.argmax(1) == label).sum().item()

            functional.reset_net(net)

            if batch_idx % PRINT_EVERY == 0:
                print(f'  epoch {epoch}, batch {batch_idx}, running loss={loss.item():.4f}')

        train_loss /= train_total
        train_acc = train_correct / train_total

        print(f'epoch={epoch}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}')

        log_writer.writerow([epoch, train_loss, train_acc])
        log_file.flush()

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_acc': train_acc,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth'))

    log_file.close()


if __name__ == '__main__':
    main()

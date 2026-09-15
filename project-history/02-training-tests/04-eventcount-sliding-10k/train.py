"""
Trains SpikingJelly's DVSGestureNet on the DVS128Gesture dataset.
Runs a training loop with periodic test-set evaluation, saving checkpoints
(latest + best by test accuracy), a training curve log (CSV), and the run's
hyperparameters (JSON) after each epoch.

Sliding fixed-count variant: instead of SpikingJelly's built-in per-recording
windowing (frame count adapts to each recording's length), this builds
frames from raw events by slicing a fixed absolute count (n_events=10,000)
per frame, then groups non-overlapping runs of T frames into windows. Same
windowing mechanism as the live GenX320 pipeline. n_events was tuned against
per-class p10 event density. No augmentation.

Adapted from SpikingJelly's classify_dvsg.py example
(https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/classify_dvsg.py),
licensed under the Qǐzhì Open Source License 1.0 (see third-party-licenses/).
"""

import os
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = './data/DVS128Gesture'
DEVICE = 'cuda'
T = 8
N_EVENTS = 10_000   # fixed events per frame, tuned against per-class p10 density
BATCH_SIZE = 16
CHANNELS = 64
EPOCHS = 64
LR = 1e-3
NUM_CLASSES = 11
RUN_NAME = 'eventcount_10k'
CHECKPOINT_DIR = './weights-and-logs'
RESUME_PATH = None    # set to './weights-and-logs/checkpoint_latest.pth' to resume
PRINT_EVERY = 100

NUM_WORKERS = 0

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm roll', 'air drums', 'air guitar', 'other gestures',
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
    # groups into non-overlapping T-frame windows, fixed absolute event
    # count per frame, same mechanism as the live GenX320 script.

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
    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )

    functional.set_step_mode(net, 'm')
    net.to(DEVICE)

    train_base = DVS128Gesture(DATA_DIR, train=True, data_type='event')
    test_base = DVS128Gesture(DATA_DIR, train=False, data_type='event')

    print('building train windows...')
    train_set = EventCountWindowDataset(train_base, T=T, n_events=N_EVENTS)
    print('building test windows...')
    test_set = EventCountWindowDataset(test_base, T=T, n_events=N_EVENTS)

    print(f'train samples: {len(train_set)}  (from {len(train_base)} recordings)')
    print(f'test samples: {len(test_set)}  (from {len(test_base)} recordings)')

    train_class_counts = {}
    for _, label in train_set.samples:
        train_class_counts[label] = train_class_counts.get(label, 0) + 1
    print('train samples per class:',
          {CLASS_NAMES[k]: v for k, v in sorted(train_class_counts.items())})

    test_class_counts = {}
    for _, label in test_set.samples:
        test_class_counts[label] = test_class_counts.get(label, 0) + 1
    print('test samples per class:',
          {CLASS_NAMES[k]: v for k, v in sorted(test_class_counts.items())})

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    max_test_acc = -1

    with open(os.path.join(CHECKPOINT_DIR, 'run_config.json'), 'w') as f:
        json.dump({'T': T, 'n_events': N_EVENTS, 'channels': CHANNELS, 'batch_size': BATCH_SIZE,
                   'lr': LR, 'epochs': EPOCHS, 'run_name': RUN_NAME}, f, indent=2)

    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    start_epoch = 0
    if RESUME_PATH is not None:
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(f'resumed from {RESUME_PATH}, starting at epoch {start_epoch}')

    for epoch in range(start_epoch, EPOCHS):

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

        scheduler.step()

        net.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.to(DEVICE).transpose(0, 1).float()
                label = label.to(DEVICE)
                label_onehot = F.one_hot(label, NUM_CLASSES).float()

                out_firing_rate = net(frame).mean(0)
                loss = F.mse_loss(out_firing_rate, label_onehot)

                test_total += label.numel()
                test_loss += loss.item() * label.numel()
                test_correct += (out_firing_rate.argmax(1) == label).sum().item()
                functional.reset_net(net)

        test_loss /= test_total
        test_acc = test_correct / test_total

        print(f'epoch={epoch}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
              f'test_loss={test_loss:.4f}  test_acc={test_acc:.4f}')

        log_writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        log_file.flush()

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc,
        }

        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth'))

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint['max_test_acc'] = max_test_acc
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pth'))
            print(f'  -> new best test_acc={max_test_acc:.4f}, saved checkpoint_best.pth')

    log_file.close()


if __name__ == '__main__':
    main()

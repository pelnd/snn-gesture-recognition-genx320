"""
Domain-gap baseline evaluation, covers all 7 runs (base model x 5, fine-tuned
model x 2) by changing the 4 CONFIG values below and re-running.

Built on top of eval_heldout.py, same windowing/binning, same checkpoint
loading convention, same EXCLUDE_CLIPS mechanism.

Each run appends one row to RESULTS_CSV so all 7 results end up in one file.
"""

import os
import csv
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

# ============================================================
# CONFIG -- set these 4 values per run
# ============================================================

MODEL = 'finetuned'        # 'base' or 'finetuned'
CLIP_SET = 'heldout'       # 'all' (all clips, base model only) or 'heldout' (33 held-out clips)
N_EVENTS = 50_000      # 50_000 or 10_000
EXCLUDE_CLIPS = {
    # fill in for the "heldout 50k excl. ambiguous" run only:
    # 'C_class01_right_hand_wave/clip_002.npy'
}

# ============================================================
# ============================================================

TEST_CLIPS_FILE = '../../03-finetuning/test_clips.txt'
RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning'

CHECKPOINT_PATHS = {
    'base': '../../03-finetuning/checkpoint_best_10k.pth',
    'finetuned': '../../03-finetuning/finetuning-raw/weights-and-logs/checkpoint_latest.pth',
}

RESULTS_CSV = './domain_gap_results.csv'

DEVICE = 'cpu'
T = 8
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


def load_heldout_clips(test_clips_file):
    # returns list of (clip_path, class_idx), same format/behaviour as eval_heldout.py
    clips = []
    n_excluded = 0
    with open(test_clips_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['clip_path'] in EXCLUDE_CLIPS:
                n_excluded += 1
                continue
            clips.append((row['clip_path'], int(row['class_idx'])))
    if EXCLUDE_CLIPS:
        print(f'excluded {n_excluded}/{n_excluded + len(clips)} listed clips via EXCLUDE_CLIPS')
    return clips


def load_all_clips(recordings_dir):
    # returns list of (clip_path, class_idx) for every clip under recordings_dir,
    # parsed the same way as the latency scripts (folder name "*_classNN_*")
    clips = []
    class_folders = sorted(glob.glob(os.path.join(recordings_dir, '*_class*_*')))
    assert len(class_folders) > 0, f'no class folders found under {recordings_dir} -- check the path'

    for folder in class_folders:
        name = os.path.basename(folder)
        try:
            class_idx = int(name.split('_class')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        for clip_path in sorted(glob.glob(os.path.join(folder, 'clip_*.npy'))):
            if clip_path in EXCLUDE_CLIPS:
                continue
            clips.append((clip_path, class_idx))
    return clips


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, clips, T, n_events):
        self.samples = []
        for clip_path, class_idx in clips:
            events = np.load(clip_path)
            n_windows = (len(events) // n_events) // T
            if n_windows == 0:
                continue
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
        frame, label = self.samples[idx]
        return frame, label


def main():
    assert MODEL in CHECKPOINT_PATHS, f'MODEL must be one of {list(CHECKPOINT_PATHS)}'
    assert CLIP_SET in ('all', 'heldout'), "CLIP_SET must be 'all' or 'heldout'"
    if CLIP_SET == 'all' and MODEL == 'finetuned':
        print('warning: running the fine-tuned model on "all" clips includes its own '
              'training data -- this number will be contaminated, not a real generalization '
              'figure. Proceeding anyway since you asked for it.')

    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    functional.set_step_mode(net, 'm')
    net.to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATHS[MODEL], map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    print(f'model={MODEL}  clip_set={CLIP_SET}  N_EVENTS={N_EVENTS}')
    print(f'loaded {CHECKPOINT_PATHS[MODEL]} (epoch={checkpoint.get("epoch")})')

    clips = load_all_clips(RECORDINGS_DIR) if CLIP_SET == 'all' else load_heldout_clips(TEST_CLIPS_FILE)
    print(f'{len(clips)} clips loaded')

    dataset = WindowDataset(clips, T=T, n_events=N_EVENTS)
    print(f'{len(dataset)} windows built')
    if len(dataset) == 0:
        print('no windows built -- check RECORDINGS_DIR / TEST_CLIPS_FILE / N_EVENTS above')
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    per_class_correct = {i: 0 for i in range(NUM_CLASSES)}
    per_class_total = {i: 0 for i in range(NUM_CLASSES)}
    total_correct, total = 0, 0

    with torch.no_grad():
        for frame, label in loader:
            frame = frame.to(DEVICE).transpose(0, 1).float()
            label = label.to(DEVICE)
            pred = net(frame).mean(0).argmax(1)
            for p, l in zip(pred.tolist(), label.tolist()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1
            total += label.numel()
            total_correct += (pred == label).sum().item()
            functional.reset_net(net)

    acc = total_correct / total
    print(f'\noverall accuracy: {total_correct}/{total} = {acc:.4f}\n')
    print(f'{"class":<28} {"correct":>8} {"total":>8} {"accuracy":>10}')
    for idx in range(NUM_CLASSES):
        c, t = per_class_correct[idx], per_class_total[idx]
        acc_str = f'{c/t:.4f}' if t > 0 else 'n/a'
        print(f'{CLASS_NAMES[idx]:<28} {c:>8} {t:>8} {acc_str:>10}')

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'clip_set', 'n_events', 'n_excluded_clips',
                              'correct', 'total', 'accuracy'])
        writer.writerow([MODEL, CLIP_SET, N_EVENTS, len(EXCLUDE_CLIPS),
                          total_correct, total, f'{acc:.4f}'])
    print(f'\nappended result row to {RESULTS_CSV}')


if __name__ == '__main__':
    main()

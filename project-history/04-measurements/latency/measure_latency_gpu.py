"""
Pure inference-latency benchmark for the fine-tuned GenX320 SNN -- LAB PC (GPU).
Same architecture / checkpoint-loading / windowing conventions as
finetune_raw.py and app3.py.

Run with RUN_ON = 'heldout' first (sanity check + latency, 33 clips),
then set RUN_ON = 'all' and run again (latency only, ~277 clips, no
leakage concern since we're not scoring accuracy on that pass).
"""

import os
import glob
import time

import numpy as np
import torch

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

# config

DEVICE = 'cuda'

RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning'
CHECKPOINT_PATH = '../../03-finetuning/finetuning-raw/weights-and-logs/checkpoint_latest.pth'
TEST_CLIPS_FILE = '../../03-finetuning/test_clips.txt'

RUN_ON = 'heldout'   # 'heldout' (sanity check + latency, 33 clips) or 'all' (latency only, ~277 clips)

# fixed -- must match the fine-tuning setup, don't change

T = 8
N_EVENTS = 50_000
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


def load_heldout_relpaths(test_clips_file):
    # matches by the "<class_folder>/clip_XXX.npy" stored in test_clips.txt
    relpaths = set()
    if not os.path.exists(test_clips_file):
        print(f'warning: {test_clips_file} not found, treating all clips as non-held-out')
        return relpaths
    with open(test_clips_file, 'r') as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(',')
            relpaths.add(parts[2])
    return relpaths


def events_to_frame(x, y, p, h=H, w=W):
    x_bin = (x.astype(np.int64) * w) // NATIVE_W
    y_bin = (y.astype(np.int64) * h) // NATIVE_H
    p = p.astype(np.int64)
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
    frame = np.zeros((2, h, w), dtype=np.float32)
    np.add.at(frame, (p, y_bin, x_bin), 1.0)
    return frame


def build_windows(recordings_dir, heldout_relpaths, only_heldout):
    assert os.path.isdir(recordings_dir), f'RECORDINGS_DIR not found: {recordings_dir}'

    samples = []  # (frames [T,2,H,W], label, clip_path)
    skipped_clips = 0
    used_clips = 0

    class_folders = sorted(glob.glob(os.path.join(recordings_dir, '*_class*_*')))
    assert len(class_folders) > 0, (
        f'no class folders found under {recordings_dir} -- check the path (typo?)'
    )

    for folder in class_folders:
        name = os.path.basename(folder)
        try:
            class_idx = int(name.split('_class')[1].split('_')[0])
        except (IndexError, ValueError):
            continue

        clips = sorted(glob.glob(os.path.join(folder, 'clip_*.npy')))
        for clip_path in clips:
            rel = os.path.basename(folder) + '/' + os.path.basename(clip_path)
            is_heldout = rel in heldout_relpaths

            if only_heldout and not is_heldout:
                continue

            events = np.load(clip_path)
            n_total = len(events)
            n_frames_available = n_total // N_EVENTS
            n_windows = n_frames_available // T

            if n_windows == 0:
                skipped_clips += 1
                continue
            used_clips += 1

            for w_idx in range(n_windows):
                frames = []
                for t in range(T):
                    start = (w_idx * T + t) * N_EVENTS
                    end = start + N_EVENTS
                    chunk = events[start:end]
                    frames.append(events_to_frame(chunk['x'], chunk['y'], chunk['p']))
                samples.append((np.stack(frames), class_idx, clip_path))

    print(f'clips used: {used_clips}, clips skipped (not enough events for one window): {skipped_clips}')
    return samples


def load_checkpoint(net, path, device):
    assert os.path.exists(path), f'CHECKPOINT_PATH not found: {path}'
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'net' in obj:
        net.load_state_dict(obj['net'])
        print(f'loaded {path} (dict-wrapped checkpoint, epoch={obj.get("epoch")}, '
              f'train_acc={obj.get("train_acc")})')
    else:
        net.load_state_dict(obj)
        print(f'loaded {path} (raw state_dict, no dict wrapper)')


def main():
    assert torch.cuda.is_available(), 'DEVICE=cuda but torch.cuda.is_available() is False'
    print(f'device={DEVICE}  gpu={torch.cuda.get_device_name(0)}  run_on={RUN_ON}')

    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    functional.set_step_mode(net, 'm')
    net.to(DEVICE)
    net.eval()
    load_checkpoint(net, CHECKPOINT_PATH, DEVICE)

    heldout_relpaths = load_heldout_relpaths(TEST_CLIPS_FILE)
    print(f'{len(heldout_relpaths)} held-out clips listed in {TEST_CLIPS_FILE}')

    samples = build_windows(RECORDINGS_DIR, heldout_relpaths, only_heldout=(RUN_ON == 'heldout'))
    print(f'{len(samples)} windows to measure ({RUN_ON})')

    latencies_ms = []
    correct = 0

    for idx, (frames, label, clip_path) in enumerate(samples):
        x_in = frames[:, None]  # [T, 1, 2, H, W]
        x_in = torch.from_numpy(x_in).float().to(DEVICE)

        with torch.no_grad():
            torch.cuda.synchronize()  # flush any pending async work before starting the clock
            t0 = time.perf_counter()
            out_firing_rate = net(x_in).mean(0)
            pred = out_firing_rate.argmax(1).item()  # .item() forces a sync, so t1 is accurate
            t1 = time.perf_counter()

        functional.reset_net(net)  # not timed -- clears state for next window

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        if pred == label:
            correct += 1

        if idx % 50 == 0:
            print(f'  window {idx}/{len(samples)}  latency={latency_ms:.2f}ms')

    lat = np.array(latencies_ms)

    print()
    print(f'=== {RUN_ON} results, device={DEVICE} ===')
    print(f'windows measured: {len(lat)}')
    print(f'latency: median={np.median(lat):.2f}ms  mean={lat.mean():.2f}ms  '
          f'std={lat.std():.2f}ms  min={lat.min():.2f}ms  max={lat.max():.2f}ms')

    if RUN_ON == 'heldout':
        acc = correct / len(samples) if samples else 0.0
        print(f'sanity-check accuracy: {correct}/{len(samples)} = {acc*100:.1f}%  '
              f'(compare against 89.7% / 140/156 from the GPU eval)')


if __name__ == '__main__':
    main()

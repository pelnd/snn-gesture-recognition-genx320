"""
Reports event density (events/sec) across the recorded GenX320 fine-tuning
clips.

Per-class: percentiles of events/sec (p10-p90) and mean clip duration, used
to compare density against DVS128Gesture and check the ranking of gestures
by density holds across both sensors.

Per-subject: overall and per-class mean events/sec, used to check how much
subjects differ in recording style/pace.
"""

import os
import glob
import numpy as np

RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning'

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm rolls', 'air drums', 'air guitar',
]


def events_per_sec(clip_path):
    events = np.load(clip_path)
    if len(events) < 2:
        return None
    duration_s = (events['t'].max() - events['t'].min()) / 1e6
    if duration_s <= 0:
        return None
    return len(events) / duration_s, duration_s, len(events)


def main():
    class_folders = sorted(glob.glob(os.path.join(RECORDINGS_DIR, '*_class*_*')))

    per_class = {i: [] for i in range(len(CLASS_NAMES))}

    for folder in class_folders:
        name = os.path.basename(folder)
        # folder = {subject}_class{NN}_{name}
        try:
            class_idx = int(name.split('_class')[1].split('_')[0])
        except (IndexError, ValueError):
            print(f'skipping unrecognized folder: {name}')
            continue

        subject = name.split('_class')[0]

        clips = sorted(glob.glob(os.path.join(folder, 'clip_*.npy')))
        for clip_path in clips:
            result = events_per_sec(clip_path)
            if result is None:
                print(f'skipping empty/degenerate clip: {clip_path}')
                continue
            rate, duration_s, n_events = result
            per_class[class_idx].append((rate, duration_s, n_events, subject))

    # section 1: per-class density with percentiles
    print('=== per-class event density (events/sec) ===')
    print(f'{"class":<28} {"clips":>6} {"p10":>9} {"p25":>9} {"median":>9} {"p75":>9} {"p90":>9} {"mean dur(s)":>12}')
    for idx in range(len(CLASS_NAMES)):
        entries = per_class[idx]
        if not entries:
            print(f'{CLASS_NAMES[idx]:<28} {"NO CLIPS FOUND":>6}')
            continue
        rates = np.array([e[0] for e in entries])
        durs = np.array([e[1] for e in entries])
        p10, p25, p50, p75, p90 = np.percentile(rates, [10, 25, 50, 75, 90])
        print(f'{CLASS_NAMES[idx]:<28} {len(entries):>6} {p10:>9.0f} {p25:>9.0f} {p50:>9.0f} {p75:>9.0f} {p90:>9.0f} {durs.mean():>12.2f}')

    # section 2: per-subject comparison, overall and per-class
    print('\n=== per-subject clip counts ===')
    subject_totals = {}
    for entries in per_class.values():
        for rate, duration_s, n_events, subject in entries:
            subject_totals.setdefault(subject, 0)
            subject_totals[subject] += 1
    for s, n in sorted(subject_totals.items()):
        print(f'  {s}: {n} clips total')

    print('\n=== per-subject event density, overall (events/sec) ===')
    all_subjects = sorted(subject_totals.keys())
    overall_by_subject = {s: [] for s in all_subjects}
    for entries in per_class.values():
        for rate, duration_s, n_events, subject in entries:
            overall_by_subject[subject].append(rate)
    for s in all_subjects:
        rates = np.array(overall_by_subject[s])
        print(f'  {s}: mean={rates.mean():.0f}  median={np.median(rates):.0f}  n={len(rates)}')

    print('\n=== per-subject event density, per class (mean events/sec) ===')
    header = f'{"class":<28}' + ''.join(f'{s:>14}' for s in all_subjects)
    print(header)
    for idx in range(len(CLASS_NAMES)):
        entries = per_class[idx]
        row = f'{CLASS_NAMES[idx]:<28}'
        for s in all_subjects:
            rates = np.array([e[0] for e in entries if e[3] == s])
            cell = f'{rates.mean():.0f}' if len(rates) > 0 else 'no clips'
            row += f'{cell:>14}'
        print(row)


if __name__ == '__main__':
    main()

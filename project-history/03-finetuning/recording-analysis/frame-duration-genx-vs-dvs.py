import numpy as np
import os
import glob
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DVS_DATA_DIR = '../../../data/DVS128Gesture'
GENX_RECORDINGS_DIR = '../../../data/genx320-data-for-finetuning'

DVS_N_EVENTS = 10_000
GENX_CANDIDATES = [37_000, 50_000]

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm rolls', 'air drums', 'air guitar',
]


def duration_for_n_events(t_array, n_events):
    if len(t_array) < n_events:
        return None
    return (t_array[n_events - 1] - t_array[0]) / 1e6


def get_dvs_frame_durations():
    per_class = {i: [] for i in range(len(CLASS_NAMES))}
    for train_flag in (True, False):
        dataset = DVS128Gesture(DVS_DATA_DIR, train=train_flag, data_type='event')
        for idx in range(len(dataset)):
            events, label = dataset[idx]
            d = duration_for_n_events(events['t'], DVS_N_EVENTS)
            if d is not None:
                per_class[label].append(d)
    return per_class


def get_genx_frame_durations(n_events):
    class_folders = sorted(glob.glob(os.path.join(GENX_RECORDINGS_DIR, '*_class*_*')))
    per_class = {i: [] for i in range(len(CLASS_NAMES))}
    for folder in class_folders:
        name = os.path.basename(folder)
        try:
            class_idx = int(name.split('_class')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        clips = sorted(glob.glob(os.path.join(folder, 'clip_*.npy')))
        for clip_path in clips:
            events = np.load(clip_path)
            d = duration_for_n_events(events['t'], n_events)
            if d is not None:
                per_class[class_idx].append(d)
    return per_class


def main():
    print(f'Loading DVS128 (N_EVENTS={DVS_N_EVENTS})...')
    dvs_durs = get_dvs_frame_durations()

    genx_durs = {}
    for n_events in GENX_CANDIDATES:
        print(f'Loading GenX320 (N_EVENTS={n_events})...')
        genx_durs[n_events] = get_genx_frame_durations(n_events)

    print()
    print(f'{"class":<28} {"DVS p25":>9} {"DVS med":>9} {"DVS p75":>9} | '
          f'{"37k p25":>9} {"37k med":>9} {"37k p75":>9} | '
          f'{"50k p25":>9} {"50k med":>9} {"50k p75":>9}')

    for idx in range(len(CLASS_NAMES)):
        row = f'{CLASS_NAMES[idx]:<28}'

        dvs_vals = np.array(dvs_durs[idx])
        if len(dvs_vals) > 0:
            p25, p50, p75 = np.percentile(dvs_vals, [25, 50, 75])
            row += f' {p25:>9.3f} {p50:>9.3f} {p75:>9.3f}'
        else:
            row += f' {"n/a":>9} {"n/a":>9} {"n/a":>9}'

        for n_events in GENX_CANDIDATES:
            vals = np.array(genx_durs[n_events][idx])
            row += ' |'
            if len(vals) > 0:
                p25, p50, p75 = np.percentile(vals, [25, 50, 75])
                row += f' {p25:>9.3f} {p50:>9.3f} {p75:>9.3f}'
            else:
                row += f' {"n/a":>9} {"n/a":>9} {"n/a":>9}'

        print(row)


if __name__ == '__main__':
    main()

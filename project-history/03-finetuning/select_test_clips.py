"""
Picks held-out test clips for the raw-vs-deduped comparison.

For each class, looks at clips with at least one usable window, then picks
N_TEST_PER_CLASS of them, preferring clips with spare windows over ones
right at the edge, and alternating across subjects (real subject, after
merging the re-recorded C/D folders into A/B) so the test set isn't skewed
toward one person's style.

Writes the picks to OUTPUT_FILE (test_clips.txt), with clip_path relative
to RECORDINGS_DIR. These clips must be left out of both the raw and
deduped fine-tuning runs.
"""

import os
import glob
import numpy as np

RECORDINGS_DIR = '../../data/genx320-data-for-finetuning'
OUTPUT_FILE = './test_clips.txt'
N_EVENTS = 50_000
T = 8
N_TEST_PER_CLASS = 3

# folder prefixes C and D are actually subject A and B (re-recording batch), map them
SUBJECT_MAP = {'A': 'A', 'B': 'B', 'C': 'A', 'D': 'B'}

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm rolls', 'air drums', 'air guitar',
]


def n_windows_for_clip(clip_path, n_events, t_frames):
    events = np.load(clip_path)
    return len(events) // (n_events * t_frames)


def select_test_clips(usable_clips):
    # usable_clips: list of (clip_path, n_windows, subject)
    # group by real subject, sort each group by n_windows descending (prefer clips
    # with windows to spare over ones right at the 1-window edge)
    groups = {}
    for clip_path, n_win, subject in usable_clips:
        groups.setdefault(subject, []).append((clip_path, n_win))
    for subject in groups:
        groups[subject].sort(key=lambda x: -x[1])

    # round-robin pick across subjects, starting with whichever has more usable
    # clips, so the test set mixes subjects rather than draining one
    subjects_by_size = sorted(groups.keys(), key=lambda s: -len(groups[s]))
    picks = []
    pointers = {s: 0 for s in subjects_by_size}
    while len(picks) < N_TEST_PER_CLASS:
        progressed = False
        for s in subjects_by_size:
            if len(picks) >= N_TEST_PER_CLASS:
                break
            if pointers[s] < len(groups[s]):
                clip_path, n_win = groups[s][pointers[s]]
                picks.append((clip_path, n_win, s))
                pointers[s] += 1
                progressed = True
        if not progressed:
            break  # ran out of clips across all subjects
    return picks


def main():
    class_folders = sorted(glob.glob(os.path.join(RECORDINGS_DIR, '*_class*_*')))
    per_class = {i: [] for i in range(len(CLASS_NAMES))}

    for folder in class_folders:
        name = os.path.basename(folder)
        try:
            class_idx = int(name.split('_class')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        raw_subject = name.split('_class')[0]
        subject = SUBJECT_MAP.get(raw_subject, raw_subject)

        clips = sorted(glob.glob(os.path.join(folder, 'clip_*.npy')))
        for clip_path in clips:
            n_win = n_windows_for_clip(clip_path, N_EVENTS, T)
            if n_win >= 1:
                per_class[class_idx].append((clip_path, n_win, subject))

    all_selected = []
    print(f'Test set selection (N_EVENTS={N_EVENTS}, T={T}, {N_TEST_PER_CLASS}/class)\n')
    print(f'{"class":<28} {"usable clips":>12} {"selected":>9} {"subjects":>10}')
    for idx in range(len(CLASS_NAMES)):
        usable = per_class[idx]
        if len(usable) < N_TEST_PER_CLASS:
            print(f'{CLASS_NAMES[idx]:<28} {len(usable):>12} {"NOT ENOUGH":>9}')
            continue
        selected = select_test_clips(usable)
        subj_counts = {}
        for clip_path, n_win, subject in selected:
            subj_counts[subject] = subj_counts.get(subject, 0) + 1
            all_selected.append((idx, CLASS_NAMES[idx], clip_path, n_win, subject))
        subj_str = ', '.join(f'{s}:{c}' for s, c in sorted(subj_counts.items()))
        print(f'{CLASS_NAMES[idx]:<28} {len(usable):>12} {len(selected):>9} {subj_str:>10}')

     with open(OUTPUT_FILE, 'w') as f:
        f.write('class_idx,class_name,clip_path,n_windows,subject\n')
        for class_idx, class_name, clip_path, n_win, subject in all_selected:
            rel_path = os.path.relpath(clip_path, RECORDINGS_DIR).replace(os.sep, '/')
            f.write(f'{class_idx},{class_name},{rel_path},{n_win},{subject}\n')

    print(f'\n{len(all_selected)} test clips written to {OUTPUT_FILE}')
    print('Exclude these exact paths from BOTH the raw and deduped fine-tuning runs.')


if __name__ == '__main__':
    main()

"""
Live recording tool for building the GenX320 fine-tuning dataset.

Streams events from the GenX320 sensor via Metavision SDK, shows a live
occupancy display for the current gesture class, and saves raw (t, x, y, p)
events for each recorded clip as clip_NNN.npy under OUT_DIR/<subject>_class<i>_<name>/.

Each clip starts with a COUNTDOWN_SECONDS delay before recording begins, so the
"reaching for the keyboard to start" motion is never captured. The last
TRIM_TAIL_US of raw events is cut on save, for the same reason at the other
end ("reaching for the keyboard to stop").

Controls: [space] start/stop a clip, [ ]/[ [ ] cycle gesture class (disabled
while recording, to avoid mislabeling a clip mid-recording), q quit.
"""

import os
import glob
import time
import numpy as np
import cv2

from metavision_core.event_io import EventsIterator


H, W = 128, 128
NATIVE_H, NATIVE_W = 320, 320
OUT_DIR = '../../data/genx320-data-for-finetuning'

CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures', 'left hand wave',
    'right arm clockwise', 'right arm counter clockwise',
    'left arm clockwise', 'left arm counter clockwise',
    'arm roll', 'air drums', 'air guitar',
]

# tuning bias values
BIAS_DIFF_ON = 25
BIAS_DIFF_OFF = 28

# small window just for live display/occupancy feedback, NOT the training window
# raw (t,x,y,p) events are saved for every event regardless of this
DISPLAY_N_EVENTS = 5000
DISPLAY_MAX_WAIT_US = 100_000

COUNTDOWN_SECONDS = 2.0

# trim this much off the END of every saved clip, to cut out the
# "reaching for the keyboard to stop recording" motion
TRIM_TAIL_US = 2_000_000


def events_to_display_frame(events):
    x = (events['x'].astype(np.int64) * W) // NATIVE_W
    y = (events['y'].astype(np.int64) * H) // NATIVE_H
    p = events['p'].astype(np.int64)
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, p = x[in_bounds], y[in_bounds], p[in_bounds]
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y, x), 1.0)
    return frame


def next_clip_index(folder):
    existing = glob.glob(os.path.join(folder, 'clip_*.npy'))
    if not existing:
        return 1
    nums = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing]
    return max(nums) + 1


def save_clip(events_list, folder, clip_idx):
    os.makedirs(folder, exist_ok=True)
    all_events = np.concatenate(events_list)

    # drop the tail end of the clip to cut out the "reaching for the
    # keyboard to hit stop" motion, keep only events up to
    # (last_timestamp - TRIM_TAIL_US)
    if all_events.size > 0:
        cutoff = all_events['t'].max() - TRIM_TAIL_US
        all_events = all_events[all_events['t'] <= cutoff]

    out = np.zeros(len(all_events), dtype=[('t', '<i8'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])
    out['t'] = all_events['t']
    out['x'] = all_events['x']
    out['y'] = all_events['y']
    p = all_events['p']
    out['p'] = (p > 0).astype(np.uint8) if p.min() < 0 else p.astype(np.uint8)
    path = os.path.join(folder, f'clip_{clip_idx:03d}.npy')
    np.save(path, out)
    return path, len(out)


def main():
    subject = input('subject id (e.g. subjectA): ').strip()

    mv_iterator = EventsIterator("", mode="mixed", n_events=DISPLAY_N_EVENTS, delta_t=DISPLAY_MAX_WAIT_US)
    device = mv_iterator.reader.device
    biases = device.get_i_ll_biases()
    biases.set("bias_diff_on", BIAS_DIFF_ON)
    biases.set("bias_diff_off", BIAS_DIFF_OFF)
    height, width = mv_iterator.get_size()
    assert (height, width) == (NATIVE_H, NATIVE_W), (
        f'expected {NATIVE_H}x{NATIVE_W}, got {height}x{width}'
    )

    class_idx = 0
    recording = False
    countdown_start = None
    clip_buffer = []
    cv2.namedWindow("Recorder", cv2.WINDOW_NORMAL)

    print("controls: [space]=start/stop clip  [ ] = prev/next class  q = quit")

    for events in mv_iterator:
        if events.size == 0:
            continue

        if countdown_start is not None and time.time() - countdown_start >= COUNTDOWN_SECONDS:
            recording = True
            clip_buffer = []
            countdown_start = None

        if recording:
            clip_buffer.append(events)

        frame = events_to_display_frame(events)
        occ = (frame > 0).mean()

        disp = np.zeros((H, W, 3), dtype=np.uint8)
        disp[:, :, 1] = np.clip(frame[1] * 20, 0, 255)
        disp[:, :, 2] = np.clip(frame[0] * 20, 0, 255)
        disp = cv2.resize(disp, (512, 512))

        folder_name = f"{subject}_class{class_idx:02d}_{CLASS_NAMES[class_idx].replace(' ', '_')}"
        folder_path = os.path.join(OUT_DIR, folder_name)

        if countdown_start is not None:
            remaining = max(0.0, COUNTDOWN_SECONDS - (time.time() - countdown_start))
            status = f"GET READY: {remaining:.1f}s"
            color = (0, 255, 255)
        elif recording:
            status = "RECORDING"
            color = (0, 0, 255)
        else:
            status = "idle"
            color = (255, 255, 255)
        cv2.putText(disp, f"{status}  class {class_idx}: {CLASS_NAMES[class_idx]}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(disp, f"subject: {subject}  occ: {occ:.3f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if recording:
            cv2.putText(disp, f"events buffered: {sum(e.size for e in clip_buffer)}",
                        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Recorder", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if not recording and countdown_start is None:
                countdown_start = time.time()
            elif recording:
                recording = False
                if clip_buffer and sum(e.size for e in clip_buffer) > 0:
                    idx = next_clip_index(folder_path)
                    path, n = save_clip(clip_buffer, folder_path, idx)
                    print(f'saved {path}  ({n} events)')
                else:
                    print('empty clip, not saved')
                clip_buffer = []
        elif key == ord(']'):
            if not recording:
                class_idx = (class_idx + 1) % len(CLASS_NAMES)
        elif key == ord('['):
            if not recording:
                class_idx = (class_idx - 1) % len(CLASS_NAMES)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

"""
Live inference pipeline for the GenX320 event camera on Raspberry Pi,
using time-based windowing to match the 05-time-windowing /
06-ceiling-ref training approach: events are grabbed in fixed 125ms
chunks (matching training's DURATION), then randomly subsampled down to
a fixed event count per frame before being fed to the model, since a
125ms window from the real camera holds far more events than a 125ms
window from DVS128Gesture recordings.

Displays a live event visualization with the predicted gesture class
overlaid. Requires the Metavision SDK (Prophesee) for camera access.
"""

import numpy as np
import torch
import cv2

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

from metavision_core.event_io import EventsIterator


DEVICE = 'cpu'
T = 8
CHANNELS = 64
NUM_CLASSES = 11
H, W = 128, 128
MODEL_PATH = './weights-and-logs/checkpoint_best.pth'
CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise',
    'right arm counter clockwise', 'left arm clockwise',
    'left arm counter clockwise', 'arm rolls',
    'air drums', 'air guitar',
]

EVENTS_PER_FRAME = 30_000
NATIVE_H, NATIVE_W = 320, 320


def events_to_frame_cropped(events, target_events=EVENTS_PER_FRAME, rng=None):
    # ('x','y','p','t'). Randomly subsamples down to target_events if the
    # window has more, then bins into a [2, H, W] frame.
    if rng is None:
        rng = np.random.default_rng()

    if events.size > target_events:
        idx = rng.choice(events.size, size=target_events, replace=False)
        events = events[idx]

    x = (events['x'].astype(np.int64) * W) // NATIVE_W
    y = (events['y'].astype(np.int64) * H) // NATIVE_H
    p = events['p'].astype(np.int64)

    # normalize polarity to {0, 1} in case this sensor/SDK version uses {-1, 1}
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)

    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y, x), 1.0)
    return frame


def genx320_camera():
    # Yields raw events in fixed 125ms chunks, matching training's
    # time-based windowing.
    mv_iterator = EventsIterator("", mode="delta_t", delta_t=125_000)
    device = mv_iterator.reader.device
    biases = device.get_i_ll_biases()
    biases.set("bias_diff_on", 25)
    biases.set("bias_diff_off", 28)

    height, width = mv_iterator.get_size()
    print(f'camera opened, reported resolution: {width}x{height}')

    assert (height, width) == (NATIVE_H, NATIVE_W), (
        f'expected {NATIVE_H}x{NATIVE_W}, got {height}x{width} -- '
        f'crop offsets would be wrong, fix NATIVE_H/NATIVE_W before continuing'
    )

    for events in mv_iterator:
        if events.size == 0:
            continue
        yield events


def main():
    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    functional.set_step_mode(net, 'm')
    net.to(DEVICE)
    net.eval()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    print(f'loaded weights from {MODEL_PATH}')

    frame_buffer = []
    pred = -1
    cv2.namedWindow("SNN Prediction", cv2.WINDOW_NORMAL)
    spans = []

    for events in genx320_camera():
        frame = events_to_frame_cropped(events, target_events=EVENTS_PER_FRAME)
        spans.append((events['t'][-1] - events['t'][0]) / 1000.0)
        frame_buffer.append(frame)

        if len(frame_buffer) == T:

            occ = np.mean([(f > 0).mean() for f in frame_buffer])
            tot = np.mean([f.sum() for f in frame_buffer])
            print(f'e/f: {tot:.0f}  non-zero: {occ:.4f}')

            x_in = np.stack(frame_buffer)[:, None]           # [T, 1, 2, H, W]
            x_in = torch.from_numpy(x_in).float().to(DEVICE)

            with torch.no_grad():
                out_firing_rate = net(x_in).mean(0)
                pred = out_firing_rate.argmax(1).item()

            functional.reset_net(net)

            print(f'predicted class: {pred}  '
                  f'(raw firing rates: {out_firing_rate[0].tolist()})')

            print(f'frame spans: {[f"{s:.0f}" for s in spans]}')
            spans = []
            frame_buffer = []

        # create event visualization
        event_display = np.zeros((H, W, 3), dtype=np.uint8)

        # positive polarity -> green
        event_display[:, :, 1] = np.clip(frame[1] * 20, 0, 255)

        # negative polarity -> red
        event_display[:, :, 2] = np.clip(frame[0] * 20, 0, 255)

        # enlarge for display
        event_display = cv2.resize(event_display, (512, 512))

        # add prediction text
        cv2.putText(
            event_display,
            f"Class: {CLASS_NAMES[pred] if pred >= 0 else '...'}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )

        cv2.imshow("SNN Prediction", event_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

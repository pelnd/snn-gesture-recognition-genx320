"""
Runs live gesture prediction on GenX320 camera events. Accumulates events
into count-based frames until T frames are collected, then runs inference
and prints the predicted class. No visualization — minimal/initial version
of the pipeline.

Requires the Prophesee Metavision SDK (installed separately, not via pip)
and a connected GenX320 camera.
"""

import numpy as np
import torch
 
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net

from metavision_core.event_io import EventsIterator


DEVICE = 'cpu'
T = 8
CHANNELS = 32
NUM_CLASSES = 11
H, W = 128, 128                     
MODEL_PATH = '../02-weights-and-logs/checkpoint_best.pth'


# temporary events_per_frame
EVENTS_PER_FRAME = 11000
MAX_WINDOW_WAIT_US = 500_000  # 500ms

NATIVE_H, NATIVE_W = 320, 320
Y_OFFSET = (NATIVE_H - H) // 2
X_OFFSET = (NATIVE_W - W) // 2



def events_to_frame_cropped(events):
    # events: structured array ('x','y','p','t')
    # crops to a centered 128x128 window and bins into a [2, H, W] frame

    x = events['x'].astype(np.int64) - X_OFFSET
    y = events['y'].astype(np.int64) - Y_OFFSET
    p = events['p'].astype(np.int64)
 
    # keep only events that fall inside the cropped region
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, p = x[in_bounds], y[in_bounds], p[in_bounds]
 
    # normalize polarity to {0, 1} in case this sensor/SDK version uses {-1, 1}
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)
 
    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y, x), 1.0)
    return frame

def genx320_camera(events_per_frame=EVENTS_PER_FRAME):
    # opens the connected GenX320 camera in fixed-count mode ("n_events")
    # and yields events_per_frame events at a time, skipping empty batches
  
    mv_iterator = EventsIterator("", mode="n_events", n_events=events_per_frame)
    height, width = mv_iterator.get_size()
    print(f'camera opened, reported resolution: {width}x{height}')
 
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

    for events in genx320_camera():
        frame = events_to_frame_cropped(events)
        frame_buffer.append(frame)
 
        if len(frame_buffer) == T:
            x_in = np.stack(frame_buffer)[:, None]           # [T, 1, 2, H, W]
            x_in = torch.from_numpy(x_in).float().to(DEVICE)
 
            with torch.no_grad():
                out_firing_rate = net(x_in).mean(0)
                pred = out_firing_rate.argmax(1).item()
 
            functional.reset_net(net)
 
            print(f'predicted class: {pred}  '
                  f'(raw firing rates: {out_firing_rate[0].tolist()})')
 
            frame_buffer = []  
 
if __name__ == '__main__':
    main()

# Data

This project uses two datasets: DVS128Gesture (public benchmark, used for base training) and a custom GenX320 recordings dataset (used for fine-tuning).

## DVS128Gesture

Used in `project-history/01-test-model/` and `project-history/02-training-tests/` for training and evaluating the base model, before fine-tuning on GenX320 data.

Not included in this repo. Download it manually from IBM's official release (https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) and extract it to `data/DVS128Gesture/`, the training and evaluation scripts expect it there. Alternatively you can change the paths in the scripts.

## GenX320 fine-tuning recordings

Recorded specifically for this project (see `REPORT.md` for the full story on recording setup, event density, and windowing choices) and used to fine-tune the base model in `project-history/03-finetuning/`.

The full dataset (~4GB) is not included directly in this repo. Hosting location: TBD.

### `example-recordings/`

A small sample of clips from the fine-tuning dataset, included directly in this repo so the data format can be inspected without downloading the full set.

Each file is a NumPy structured array (`.npy`) with fields:

- `x`, `y` -- pixel coordinates, native GenX320 resolution (320x320)
- `p` -- event polarity
- `t` -- timestamp

Load with:
```python
import numpy as np
events = np.load('example-recordings/A_class00_hand_clap_clip_012.npy')
events['x'], events['y'], events['p'], events['t']
```

Filenames follow `<subject>_<class>_clip_<number>.npy`. In the full dataset these same clips live nested in per-subject-per-class folders (e.g. `A_class00_hand_clap/clip_012.npy`); they've been flattened here just for these examples.

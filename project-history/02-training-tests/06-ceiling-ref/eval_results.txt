============================================================
RUN CONFIG
============================================================
Run name:  ceiling_ref
Channels:  128
T:         16
Duration:  125000 us
============================================================
MODEL COMPLEXITY
============================================================
Parameters:  1,698,926
Model size:  19.48 MB (on disk)
The directory [D:\VsCode Projects\Staj 2026\spiking jelly\DVS  Gesture dataset\duration_125000] already exists.
test samples: 855  (from 288 recordings)

============================================================
ACCURACY
============================================================
Overall accuracy: 0.9591  (820/855)

Per-class accuracy:
  hand clap                     : 0.9412  (48/51)
  right hand wave               : 1.0000  (68/68)
  other gestures                : 0.9792  (47/48)
  left hand wave                : 1.0000  (62/62)
  right arm clockwise           : 1.0000  (92/92)
  right arm counter clockwise   : 1.0000  (80/80)
  left arm clockwise            : 1.0000  (84/84)
  left arm counter clockwise    : 1.0000  (79/79)
  arm rolls                     : 1.0000  (146/146)
  air drums                     : 0.7973  (59/74)
  air guitar                    : 0.7746  (55/71)

============================================================
CONFUSION MATRIX (rows=true, columns=predicted)
============================================================
            0    1    2    3    4    5    6    7    8    9   10
true  0 |   48    0    0    0    0    0    0    0    0    1    2
true  1 |    0   68    0    0    0    0    0    0    0    0    0
true  2 |    0    0   47    0    0    1    0    0    0    0    0
true  3 |    0    0    0   62    0    0    0    0    0    0    0
true  4 |    0    0    0    0   92    0    0    0    0    0    0
true  5 |    0    0    0    0    0   80    0    0    0    0    0
true  6 |    0    0    0    0    0    0   84    0    0    0    0
true  7 |    0    0    0    0    0    0    0   79    0    0    0
true  8 |    0    0    0    0    0    0    0    0  146    0    0
true  9 |   14    0    1    0    0    0    0    0    0   59    0
true 10 |    0    0   14    0    0    0    0    0    0    2   55

============================================================
INFERENCE TIME (single sample, cpu)
============================================================
Mean:   215.14 ms
Median: 211.90 ms
Min:    188.44 ms
Max:    483.52 ms

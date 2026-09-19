# Release

This folder contains the final trained model and the code to run it live.

## Contents
- `app.py`: live inference script: reads events from a GenX320 camera, classifies gestures in real time, and shows the prediction in a window.
- `checkpoint_latest.pth`: the fine-tuned model weights `app.py` loads by default.
- `requirements.txt`: Python dependencies.
- `demo.gif`: a short demo of the model in action 

## Requirements
- Hardware: Raspberry Pi (or any machine) with a Prophesee GenX320 event camera connected.
- Software: Prophesee's Metavision SDK (for camera access) plus the dependencies listed in `requirements.txt`.

## Running it

A window opens showing the event stream and the currently predicted gesture class. Press `q` to quit.

## How it works
Events are accumulated into windows of 50,000 events each (`EVENTS_PER_FRAME`), 8 windows per prediction (`T`). If no gesture is detected for 2 seconds (`IDLE_TIMEOUT_S`), the partial window is discarded and accumulation restarts, so a pause never gets stitched into a prediction the model wasn't trained on.

Camera bias is set to `25/28`, this is not a tuned-for-accuracy value, but the setting confirmed to match what the fine-tuning data was actually recorded under (see `REPORT.md` for the full story on this).

## Recognized gestures
hand clap, right hand wave, other gestures, left hand wave,
right arm clockwise, right arm counter clockwise,
left arm clockwise, left arm counter clockwise,
arm rolls, air drums, air guitar

For training and finetuning codes, see `project-history/`.

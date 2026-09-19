# Project History

Past experiments and progress, in chronological order. This is where the project's earlier stages live, the code that led to the final `release/` version. The reasoning and results behind each stage are in `REPORT.md` at the repo root; this is just a map of what's in each folder.

## 01-test-model

The first end-to-end pipeline, built on DVS128Gesture:

- `01-training/train.py` : trains the first model.
- `02-weights-and-logs/` : its checkpoints.
- `03-evaluation/` : `static_evaluation.py` and `realtime_evaluation.py`, each with a matching `..._results.txt`.
- `04-pipeline/` : `pipeline_basic.py` and `pipeline_final.py`, the first live inference scripts.

## 02-training-tests

Six training/windowing variants compared on DVS128Gesture to pick the base model that would later be fine-tuned:   
`01-eventcount-base`  
`02-eventcount-with-decaying-weight`  
`03-eventcount-jitter`  
`04-eventcount-sliding-10k`  
`05-time-windowing`  
`06-ceiling-ref`  
Each folder follows the same layout: `train.py`, `eval.py`, `eval_results.txt`, and a `weights-and-logs/` folder (checkpoints, training log/curves, run config). `confusion_matrices.png` compares all six at a glance.

## 03-finetuning

Fine-tuning the base model on real GenX320 recordings.

- `record-new-data.py` : records the GenX320 dataset.
- `recording-analysis/` : scripts and results comparing event density and frame duration between GenX320 and DVS128Gesture.
- `select_test_clips.py` / `test_clips.txt` : picks and lists the held-out test clips.
- `checkpoint_best_10k.pth` : the base model checkpoint being fine-tuned.
- `finetuning-raw/` and `finetuning-deduped/` : two fine-tuning runs, one on the raw recordings and one on a deduplicated version; each has `train.py`, `eval.py`, `eval_results.txt`, and `weights-and-logs/`. `finetuning-raw/` additionally has `pipeline.py` and an `eval_with_exluded.py` variant that excludes ambiguous clips.

## 04-measurements

Evaluating the fine-tuned model.

- `domain-gap/` : `eval.py` runs the domain-gap baseline; `base/` and `finetuned/` hold the per-run clip lists and results for each model/clip-set/window-size combination, collected in `domain_gap_results.csv`.
- `latency/` : `measure_latency_gpu.py`, `measure_latency_laptop.py`, `measure_latency_rpi.py` benchmark inference-only latency on each machine; `measure_pipeline_latency_pi.py` benchmarks the full pipeline on the Pi, logged to `pipeline_latency_pi.csv`. Raw results in `results/`.

## Running

Each script is run from its own containing folder as the working directory (e.g. `cd 03-finetuning && python finetune_raw.py`), since paths inside them are relative to that folder.

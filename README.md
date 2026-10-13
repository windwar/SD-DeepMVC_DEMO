# SD-DeepMVC_DEMO

Reproduction demo for **Structure-guided and Density-aware Deep Multi-view Clustering**.

## Environment

Python 3.8.20, PyTorch 1.8.1+cu111, and an NVIDIA CUDA GPU.

```sh
python -m pip install -r requirements.txt
```

## Run

From the repository directory:

```sh
python run_all.py --dry-run
python run_all.py
```

Results are saved to `results/runs.csv` and `results/summary.csv`. Repeating the
command skips completed runs. To use another output directory:

```sh
python run_all.py --output ./new_results
```

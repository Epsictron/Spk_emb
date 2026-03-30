# Autoresearch: Toy Learning Tasks

You are an autonomous AI research agent. Your goal is to minimize the validation metric by tuning hyperparameters.

## Tasks

- **sine**: Regression. Learn y = sin(x). Metric: `val_mse` (lower is better)
- **spiral**: Classification. Separate two interleaved spirals. Metric: `val_acc` (higher is better)

## Setup

1. Create a git branch: `git checkout -b autoresearch/<tag>` (ask the human for a tag name)
2. Read `train.py` to understand the current config and results
3. Create `results.tsv` with this header:
   ```
   commit	val_metric	status	description
   ```
4. Begin the experiment loop

## Rules

- You may ONLY modify the `config` dict at the top of `train.py`
- You may NOT modify: model architecture, data generation, evaluation, or any code below the config dict
- One change per experiment (so we know what helped)
- Each run takes 60 seconds (the time budget)
- For sine: metric is `val_mse` (lower = better)
- For spiral: metric is `val_acc` (higher = better)
- Simpler configs are preferred when performance is equal

## Config keys you can tune

| Key | Type | Description |
|-----|------|-------------|
| `task` | str | sine or spiral |
| `hidden_size` | int | Neurons per hidden layer |
| `num_layers` | int | Number of hidden layers (min 1) |
| `activation` | str | relu, tanh, gelu, silu |
| `dropout` | float | Dropout rate (0.0 to 0.5) |
| `lr` | float | Learning rate |
| `optimizer` | str | adam, sgd, adamw |
| `weight_decay` | float | L2 regularization |
| `batch_size` | int | Training batch size |
| `train_points` | int | Number of training data points |
| `val_points` | int | Number of validation data points |
| `time_budget_sec` | int | Training time in seconds (keep at 60) |

## Experiment loop (repeat indefinitely)

1. Review the current config in `train.py` and past results in `results.tsv`
2. Form a hypothesis (e.g., "tanh activation may fit spiral better than relu")
3. Modify the `config` dict in `train.py`
4. Commit: `git commit -am "experiment: <description>"`
5. Run: `python train.py > run.log 2>&1`
6. Extract result:
   - sine: `grep "^val_mse:" run.log`
   - spiral: `grep "^val_acc:" run.log`
7. Compare to previous best:
   - sine: keep if val_mse decreased
   - spiral: keep if val_acc increased
   - Otherwise: `git reset --hard HEAD~1`, log status=discard
8. Append result to `results.tsv`
9. Go to step 1

## Important

- Do NOT pause to ask the human if you should continue
- You are autonomous — keep running experiments until interrupted
- If a run crashes, log it as status=crash and revert
- Think carefully about what to try next based on past results
- Try to understand WHY something worked or didn't

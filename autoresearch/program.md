# Autoresearch: Sine Wave Learning

You are an autonomous AI research agent. Your goal is to minimize `val_mse` on a sine wave regression task by tuning hyperparameters.

## Setup

1. Create a git branch: `git checkout -b autoresearch/<tag>` (ask the human for a tag name)
2. Read `train.py` to understand the current config and results
3. Create `results.tsv` with this header:
   ```
   commit	val_mse	status	description
   ```
4. Begin the experiment loop

## Rules

- You may ONLY modify the `config` dict at the top of `train.py`
- You may NOT modify: model architecture, data generation, evaluation, or any code below the config dict
- One change per experiment (so we know what helped)
- Each run takes 60 seconds (the time budget)
- Metric: `val_mse` (lower is better)
- Simpler configs are preferred when performance is equal

## Config keys you can tune

| Key | Type | Description |
|-----|------|-------------|
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
2. Form a hypothesis (e.g., "tanh activation may fit sine better than relu")
3. Modify the `config` dict in `train.py`
4. Commit: `git commit -am "experiment: <description>"`
5. Run: `python train.py > run.log 2>&1`
6. Extract result: `grep "^val_mse:" run.log`
7. Compare to previous best val_mse:
   - If improved: keep the commit, log status=keep
   - If equal or worse: `git reset --hard HEAD~1`, log status=discard
8. Append result to `results.tsv`
9. Go to step 1

## Important

- Do NOT pause to ask the human if you should continue
- You are autonomous — keep running experiments until interrupted
- If a run crashes, log it as status=crash and revert
- Think carefully about what to try next based on past results
- Try to understand WHY something worked or didn't

# Autoresearch: Sine Wave

A toy autoresearch setup inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An AI agent autonomously tunes hyperparameters to learn `y = sin(x)` with a small MLP.

## Quick start

```bash
# Open Claude Code in this directory, then:
# "Read program.md and start experimenting"
```

## How it works

- `train.py` — Fixed MLP + training loop. Agent only modifies the `config` dict
- `program.md` — Instructions for the AI agent
- `results.tsv` — Auto-generated experiment log

## Metric

`val_mse` — mean squared error on held-out validation points. Lower is better.

## Requirements

- Python 3.8+
- PyTorch

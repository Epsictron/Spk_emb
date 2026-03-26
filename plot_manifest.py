"""
Generate data distribution plots from manifest.

Usage:
    python plot_manifest.py --manifest data/manifest.json --output data_distribution.png
"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def plot(manifest, output):
    durations = [e["duration"] for e in manifest]
    speakers = [e["speaker_id"] for e in manifest]
    datasets = [e["dataset_name"] for e in manifest]
    languages = [e["language"] for e in manifest]
    genders = [e.get("gender", "unknown").lower() for e in manifest]

    spk_counts = Counter(speakers)
    ds_counts = Counter(datasets)
    lang_counts = Counter(languages)
    gender_counts = Counter(genders)

    # Duration per speaker
    spk_dur = {}
    for e in manifest:
        spk_dur[e["speaker_id"]] = spk_dur.get(e["speaker_id"], 0) + e["duration"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Data Distribution — {len(manifest)} utterances, {len(spk_counts)} speakers", fontsize=16, fontweight="bold")

    # 1. Duration histogram
    ax = axes[0, 0]
    ax.hist(durations, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Utterance Duration Distribution")
    ax.axvline(sum(durations) / len(durations), color="red", linestyle="--", label=f"Mean: {sum(durations)/len(durations):.1f}s")
    ax.legend()

    # 2. Gender distribution
    ax = axes[0, 1]
    labels, values = zip(*sorted(gender_counts.items()))
    colors = {"male": "#4C72B0", "female": "#DD8452", "unknown": "#AAAAAA"}
    bar_colors = [colors.get(l, "#CCCCCC") for l in labels]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, str(v), ha="center", fontweight="bold")
    ax.set_title("Gender Distribution")
    ax.set_ylabel("Utterances")

    # 3. Samples per speaker (top 50 + histogram)
    ax = axes[1, 0]
    counts = sorted(spk_counts.values())
    ax.hist(counts, bins=min(50, len(counts)), color="#55A868", edgecolor="white")
    ax.set_xlabel("Utterances per Speaker")
    ax.set_ylabel("Number of Speakers")
    ax.set_title("Utterances per Speaker Distribution")
    ax.axvline(sum(counts) / len(counts), color="red", linestyle="--", label=f"Mean: {sum(counts)/len(counts):.1f}")
    ax.legend()

    # 4. Total duration per speaker (histogram)
    ax = axes[1, 1]
    spk_durs = sorted(spk_dur.values())
    ax.hist(spk_durs, bins=min(50, len(spk_durs)), color="#C44E52", edgecolor="white")
    ax.set_xlabel("Total Duration per Speaker (s)")
    ax.set_ylabel("Number of Speakers")
    ax.set_title("Duration per Speaker Distribution")
    total_hrs = sum(durations) / 3600
    ax.axvline(sum(spk_durs) / len(spk_durs), color="navy", linestyle="--", label=f"Mean: {sum(spk_durs)/len(spk_durs):.1f}s\nTotal: {total_hrs:.1f}h")
    ax.legend()

    # 5. Dataset distribution
    ax = axes[2, 0]
    ds_labels, ds_vals = zip(*sorted(ds_counts.items(), key=lambda x: -x[1]))
    bars = ax.barh(ds_labels, ds_vals, color="#8172B3", edgecolor="white")
    for bar, v in zip(bars, ds_vals):
        ax.text(bar.get_width() + max(ds_vals) * 0.01, bar.get_y() + bar.get_height() / 2, str(v), va="center")
    ax.set_xlabel("Utterances")
    ax.set_title("Dataset Distribution")

    # 6. Language distribution
    ax = axes[2, 1]
    lang_labels, lang_vals = zip(*sorted(lang_counts.items(), key=lambda x: -x[1]))
    bars = ax.barh(lang_labels, lang_vals, color="#CCB974", edgecolor="white")
    for bar, v in zip(bars, lang_vals):
        ax.text(bar.get_width() + max(lang_vals) * 0.01, bar.get_y() + bar.get_height() / 2, str(v), va="center")
    ax.set_xlabel("Utterances")
    ax.set_title("Language Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--output", default="data_distribution.png")
    args = parser.parse_args()
    plot(load(args.manifest), args.output)

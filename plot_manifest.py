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


def fmt_duration(secs):
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {int(secs % 60)}s"


def plot(manifest, output):
    durations = [e["duration"] for e in manifest]
    speakers = [e["speaker_id"] for e in manifest]
    datasets = [e["dataset_name"] for e in manifest]
    languages = [e["language"] for e in manifest]
    genders = [e.get("gender", "").lower() for e in manifest]
    has_gender = any(g in ("male", "female") for g in genders)

    spk_counts = Counter(speakers)
    ds_counts = Counter(datasets)
    lang_counts = Counter(languages)

    # Per-speaker stats
    spk_dur = {}
    for e in manifest:
        spk_dur[e["speaker_id"]] = spk_dur.get(e["speaker_id"], 0) + e["duration"]

    # Per-dataset duration
    ds_dur = {}
    ds_spks = {}
    for e in manifest:
        ds_dur[e["dataset_name"]] = ds_dur.get(e["dataset_name"], 0) + e["duration"]
        ds_spks.setdefault(e["dataset_name"], set()).add(e["speaker_id"])

    # Gender counts (speakers, not utterances)
    if has_gender:
        spk_gender = {}
        for e in manifest:
            g = e.get("gender", "").lower()
            if e["speaker_id"] not in spk_gender:
                spk_gender[e["speaker_id"]] = g if g in ("male", "female") else "unknown"
        gender_spk_counts = Counter(spk_gender.values())
        gender_utt_counts = Counter(genders if has_gender else [])

    total_dur = sum(durations)
    total_spks = len(spk_counts)
    total_utts = len(manifest)

    # --- Build summary text ---
    summary_lines = [
        ("Total Utterances", f"{total_utts:,}"),
        ("Total Speakers", f"{total_spks:,}"),
        ("Total Duration", fmt_duration(total_dur)),
        ("Mean Utt Duration", f"{total_dur / total_utts:.2f}s"),
        ("Min Utt Duration", f"{min(durations):.2f}s"),
        ("Max Utt Duration", f"{max(durations):.2f}s"),
        ("Median Utt Duration", f"{sorted(durations)[len(durations)//2]:.2f}s"),
        ("Avg Utts/Speaker", f"{total_utts / total_spks:.1f}"),
        ("Avg Duration/Speaker", fmt_duration(total_dur / total_spks)),
        ("Datasets", f"{len(ds_counts)}"),
        ("Languages", f"{len(lang_counts)}"),
    ]
    if has_gender:
        summary_lines.append(("Male Speakers", f"{gender_spk_counts.get('male', 0):,}"))
        summary_lines.append(("Female Speakers", f"{gender_spk_counts.get('female', 0):,}"))
        if gender_spk_counts.get("unknown", 0) > 0:
            summary_lines.append(("Unknown Gender", f"{gender_spk_counts.get('unknown', 0):,}"))

    # Per-dataset summary
    ds_summary = []
    for ds_name in sorted(ds_counts.keys()):
        ds_summary.append((ds_name, f"{ds_counts[ds_name]:,} utts | {len(ds_spks[ds_name]):,} spks | {fmt_duration(ds_dur[ds_name])}"))

    # --- Plot ---
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Data Distribution", fontsize=18, fontweight="bold", y=0.98)

    # 0. Summary table (top-left)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    table_data = [[k, v] for k, v in summary_lines]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F0F0")
        cell.set_edgecolor("#DDDDDD")
    ax.set_title("Summary", fontsize=13, fontweight="bold", pad=10)

    # 1. Per-dataset summary table (top-right)
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ds_table_data = [[k, v] for k, v in ds_summary]
    table2 = ax.table(cellText=ds_table_data, colLabels=["Dataset", "Details"],
                      loc="center", cellLoc="left")
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 1.4)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#8172B3")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F0F0")
        cell.set_edgecolor("#DDDDDD")
    ax.set_title("Per-Dataset Breakdown", fontsize=13, fontweight="bold", pad=10)

    # 2. Duration histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(durations, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Utterance Duration Distribution")
    mean_dur = total_dur / total_utts
    ax.axvline(mean_dur, color="red", linestyle="--", label=f"Mean: {mean_dur:.1f}s")
    ax.legend()

    # 3. Gender or Top speakers
    ax = fig.add_subplot(gs[1, 1])
    if has_gender:
        labels, values = zip(*sorted(gender_spk_counts.items()))
        colors = {"male": "#4C72B0", "female": "#DD8452", "unknown": "#AAAAAA"}
        bar_colors = [colors.get(l, "#CCCCCC") for l in labels]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    str(v), ha="center", fontweight="bold")
        ax.set_title("Gender Distribution (Speakers)")
        ax.set_ylabel("Speakers")
    else:
        top_n = min(20, len(spk_counts))
        top_spks = spk_counts.most_common(top_n)
        spk_labels, spk_vals = zip(*reversed(top_spks))
        ax.barh(spk_labels, spk_vals, color="#DD8452", edgecolor="white")
        ax.set_xlabel("Utterances")
        ax.set_title(f"Top {top_n} Speakers by Utterance Count")

    # 4. Utterances per speaker histogram
    ax = fig.add_subplot(gs[2, 0])
    counts = sorted(spk_counts.values())
    ax.hist(counts, bins=min(50, len(counts)), color="#55A868", edgecolor="white")
    ax.set_xlabel("Utterances per Speaker")
    ax.set_ylabel("Number of Speakers")
    ax.set_title("Utterances per Speaker Distribution")
    ax.axvline(sum(counts) / len(counts), color="red", linestyle="--",
               label=f"Mean: {sum(counts)/len(counts):.1f}")
    ax.legend()

    # 5. Duration per speaker histogram
    ax = fig.add_subplot(gs[2, 1])
    spk_durs = sorted(spk_dur.values())
    ax.hist(spk_durs, bins=min(50, len(spk_durs)), color="#C44E52", edgecolor="white")
    ax.set_xlabel("Total Duration per Speaker (s)")
    ax.set_ylabel("Number of Speakers")
    ax.set_title("Duration per Speaker Distribution")
    ax.axvline(sum(spk_durs) / len(spk_durs), color="navy", linestyle="--",
               label=f"Mean: {sum(spk_durs)/len(spk_durs):.1f}s")
    ax.legend()

    # 6. Dataset duration bar chart
    ax = fig.add_subplot(gs[3, 0])
    ds_names = sorted(ds_dur.keys(), key=lambda x: -ds_dur[x])
    ds_dur_hrs = [ds_dur[d] / 3600 for d in ds_names]
    bars = ax.barh(ds_names, ds_dur_hrs, color="#8172B3", edgecolor="white")
    for bar, v in zip(bars, ds_dur_hrs):
        ax.text(bar.get_width() + max(ds_dur_hrs) * 0.01,
                bar.get_y() + bar.get_height() / 2, f"{v:.1f}h", va="center")
    ax.set_xlabel("Duration (hours)")
    ax.set_title("Dataset Duration")

    # 7. Language distribution
    ax = fig.add_subplot(gs[3, 1])
    lang_labels, lang_vals = zip(*sorted(lang_counts.items(), key=lambda x: -x[1]))
    bars = ax.barh(lang_labels, lang_vals, color="#CCB974", edgecolor="white")
    for bar, v in zip(bars, lang_vals):
        ax.text(bar.get_width() + max(lang_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2, str(v), va="center")
    ax.set_xlabel("Utterances")
    ax.set_title("Language Distribution")

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")

    # Also print summary to console
    print(f"\n{'='*45}")
    for k, v in summary_lines:
        print(f"  {k:.<30} {v}")
    print(f"{'='*45}")
    for k, v in ds_summary:
        print(f"  {k:.<20} {v}")
    print(f"{'='*45}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--output", default="data_distribution.png")
    args = parser.parse_args()
    plot(load(args.manifest), args.output)

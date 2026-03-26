"""
Generate manifest.json from dataset folders.

Expected folder structure:
    <dataset_path>/<speaker_id>/<audio_files>

Usage:
    python create_manifest.py
"""
import json
import os
import soundfile as sf

DATASETS = [
    {
        "name": "librispeech",
        "language": "english",
        "path": "/path/to/librispeech",
    },
    # Add more datasets here
]

OUTPUT = "data/manifest.json"


def build_manifest(datasets):
    manifest = []
    for ds in datasets:
        root = ds["path"]
        if not os.path.isdir(root):
            print(f"[SKIP] {root} not found")
            continue

        for spk_id in sorted(os.listdir(root)):
            spk_dir = os.path.join(root, spk_id)
            if not os.path.isdir(spk_dir):
                continue

            for fname in sorted(os.listdir(spk_dir)):
                if not fname.endswith((".wav", ".flac", ".mp3", ".ogg")):
                    continue
                fpath = os.path.join(spk_dir, fname)
                try:
                    info = sf.info(fpath)
                    duration = info.duration
                except Exception as e:
                    print(f"[WARN] {fpath}: {e}")
                    continue

                manifest.append({
                    "audio_file_path": fpath,
                    "speaker_id": spk_id,
                    "duration": round(duration, 3),
                    "dataset_name": ds["name"],
                    "language": ds["language"],
                })

        print(f"[OK] {ds['name']}: {sum(1 for m in manifest if m['dataset_name'] == ds['name'])} files")

    return manifest


if __name__ == "__main__":
    manifest = build_manifest(DATASETS)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal: {len(manifest)} entries -> {OUTPUT}")

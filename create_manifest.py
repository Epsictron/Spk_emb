"""
Generate manifest.json from dataset folders.

Expected folder structure:
    <dataset_path>/<speaker_id>/<audio_files>

Usage:
    python create_manifest.py
"""
import json
import os
from multiprocessing import Pool, cpu_count
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
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def process_file(args):
    fpath, spk_id, ds_name, ds_lang = args
    try:
        info = sf.info(fpath)
        return {
            "audio_file_path": fpath,
            "speaker_id": spk_id,
            "duration": round(info.duration, 3),
            "dataset_name": ds_name,
            "language": ds_lang,
        }
    except Exception as e:
        print(f"[WARN] {fpath}: {e}")
        return None


def build_manifest(datasets):
    # Collect all file tasks first
    tasks = []
    for ds in datasets:
        root = ds["path"]
        if not os.path.isdir(root):
            print(f"[SKIP] {root} not found")
            continue
        for spk_id in os.listdir(root):
            spk_dir = os.path.join(root, spk_id)
            if not os.path.isdir(spk_dir):
                continue
            for fname in os.listdir(spk_dir):
                if fname.endswith(AUDIO_EXTS):
                    tasks.append((os.path.join(spk_dir, fname), spk_id, ds["name"], ds["language"]))

    print(f"Found {len(tasks)} files, processing with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, tasks, chunksize=256)

    manifest = [r for r in results if r is not None]

    for ds in datasets:
        count = sum(1 for m in manifest if m["dataset_name"] == ds["name"])
        print(f"[OK] {ds['name']}: {count} files")

    return manifest


if __name__ == "__main__":
    manifest = build_manifest(DATASETS)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal: {len(manifest)} entries -> {OUTPUT}")

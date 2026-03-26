"""
Generate manifest.json from dataset folders.

Expected folder structure:
    <dataset_path>/<speaker_id>/<audio_files>

Gender file formats supported (per dataset):
    JSON:  {"spk001": "male", "spk002": "female"}
    CSV/TSV: speaker_id,gender  (or speaker_id\\tgender)

Usage:
    python create_manifest.py
"""
import json
import os
import csv
from multiprocessing import Pool, cpu_count
import soundfile as sf

DATASETS = [
    {
        "name": "librispeech",
        "language": "english",
        "path": "/path/to/librispeech",
        "gender_file": "/path/to/librispeech_gender.json",
    },
    # Add more datasets here
]

OUTPUT = "data/manifest.json"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def load_gender_map(gender_file):
    """Load speaker_id -> gender mapping from JSON or CSV/TSV."""
    if not gender_file or not os.path.isfile(gender_file):
        return {}

    if gender_file.endswith(".json"):
        with open(gender_file) as f:
            return json.load(f)

    # CSV or TSV
    gender_map = {}
    with open(gender_file) as f:
        delimiter = "\t" if gender_file.endswith(".tsv") else ","
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                spk_id, gender = row[0].strip(), row[1].strip().lower()
                if spk_id and gender and spk_id != "speaker_id":  # skip header
                    gender_map[spk_id] = gender
    return gender_map


def process_file(args):
    fpath, spk_id, ds_name, ds_lang, gender = args
    try:
        info = sf.info(fpath)
        entry = {
            "audio_file_path": fpath,
            "speaker_id": spk_id,
            "duration": round(info.duration, 3),
            "dataset_name": ds_name,
            "language": ds_lang,
        }
        if gender:
            entry["gender"] = gender
        return entry
    except Exception as e:
        print(f"[WARN] {fpath}: {e}")
        return None


def build_manifest(datasets):
    tasks = []
    for ds in datasets:
        root = ds["path"]
        if not os.path.isdir(root):
            print(f"[SKIP] {root} not found")
            continue

        gender_map = load_gender_map(ds.get("gender_file"))
        if gender_map:
            print(f"[OK] Loaded gender info for {len(gender_map)} speakers from {ds['gender_file']}")

        for spk_id in os.listdir(root):
            spk_dir = os.path.join(root, spk_id)
            if not os.path.isdir(spk_dir):
                continue
            gender = gender_map.get(spk_id, "")
            for fname in os.listdir(spk_dir):
                if fname.endswith(AUDIO_EXTS):
                    tasks.append((os.path.join(spk_dir, fname), spk_id, ds["name"], ds["language"], gender))

    print(f"Found {len(tasks)} files, processing with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, tasks, chunksize=256)

    manifest = [r for r in results if r is not None]

    for ds in datasets:
        count = sum(1 for m in manifest if m["dataset_name"] == ds["name"])
        with_gender = sum(1 for m in manifest if m["dataset_name"] == ds["name"] and m.get("gender"))
        print(f"[OK] {ds['name']}: {count} files ({with_gender} with gender)")

    return manifest


if __name__ == "__main__":
    manifest = build_manifest(DATASETS)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal: {len(manifest)} entries -> {OUTPUT}")

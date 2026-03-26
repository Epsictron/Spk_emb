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
        "gender_file": "/path/to/librispeech_gender.json",  # required
    },
    # Add more datasets here. gender_file is mandatory for each dataset.
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


def validate_gender_map(gender_map, gender_file):
    """Validate gender values and report issues."""
    valid_genders = {"male", "female", "m", "f"}
    issues = []
    for spk_id, gender in gender_map.items():
        if gender.lower() not in valid_genders:
            issues.append(f"  Speaker '{spk_id}' has invalid gender: '{gender}'")
    if issues:
        print(f"[WARN] Gender file {gender_file} has {len(issues)} invalid entries:")
        for issue in issues[:10]:
            print(issue)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    # Normalize m/f to male/female
    normalized = {}
    for spk_id, gender in gender_map.items():
        g = gender.lower()
        if g == "m":
            g = "male"
        elif g == "f":
            g = "female"
        if g in ("male", "female"):
            normalized[spk_id] = g
    return normalized


def build_manifest(datasets):
    tasks = []
    for ds in datasets:
        root = ds["path"]
        if not os.path.isdir(root):
            print(f"[SKIP] {root} not found")
            continue

        gender_file = ds.get("gender_file")
        if not gender_file:
            print(f"[SKIP] {ds['name']}: gender_file is required but not provided")
            continue
        gender_map = load_gender_map(gender_file)
        if not gender_map:
            print(f"[SKIP] {ds['name']}: gender_file '{gender_file}' is empty or not found")
            continue
        gender_map = validate_gender_map(gender_map, gender_file)
        print(f"[OK] Loaded gender info for {len(gender_map)} speakers from {gender_file}")

        # Check which speakers in folders are missing from gender file
        all_spk_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        missing = [s for s in all_spk_dirs if s not in gender_map]
        if missing:
            print(f"[WARN] {ds['name']}: Removing {len(missing)}/{len(all_spk_dirs)} speakers (no gender info)")
            print(f"  Removed speakers: {missing}")

        for spk_id in all_spk_dirs:
            spk_dir = os.path.join(root, spk_id)
            gender = gender_map.get(spk_id, "")
            if not gender:
                continue
            for dirpath, _, filenames in os.walk(spk_dir):
                for fname in filenames:
                    if fname.endswith(AUDIO_EXTS):
                        tasks.append((os.path.join(dirpath, fname), spk_id, ds["name"], ds["language"], gender))

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

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = "ILSVRC/imagenet-1k"

LABELS = [
    ("A_mammal", "mammal", 1306),
    ("B_vehicle", "vehicle", 574),
    ("C_other_animal", "other_animal", 1146),
    ("D_object", "object", 1162),
]

EXCLUDE_SYNSETS = [
    "plant.n.02", "plant_part.n.01", "fungus.n.01", "fruit.n.01",
    "vegetable.n.01", "food.n.01", "food.n.02",
    "arthropod.n.01", "gastropod.n.01",
]

EXTRA_EXCLUDE = ["flowerpot", "greenhouse", "lawn mower", "plow", "thresher",
                 "vase", "hay", "wheelbarrow", "pot,"]


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def load_wnids_and_names(token):
    wnids = names = None

    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, "classes.py", repo_type="dataset", token=token)
        pairs = re.findall(r'"(n\d{8})"\s*:\s*"([^"]*)"', Path(p).read_text())
        if len(pairs) == 1000:
            wnids = [w for w, _ in pairs]
            names = [n for _, n in pairs]
            print("class list: dataset repo classes.py")
    except Exception as e:
        print(f"  (classes.py not usable: {type(e).__name__})")

    if wnids is None:
        from timm.data import ImageNetInfo
        info = ImageNetInfo("imagenet-1k")
        wnids = [info.index_to_label_name(i) for i in range(1000)]
        names = [info.index_to_description(i) for i in range(1000)]
        print("class list: timm ImageNetInfo")

    assert len(wnids) == 1000 and wnids == sorted(wnids), \
        "class list is not 1000 sorted wnids — label ids would be misaligned"
    assert wnids[0] == "n01440764" and wnids[-1] == "n15075141", "unexpected wnid order"
    return wnids, [n.lower() for n in names]


def group_classes(wnids, names):
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synset("mammal.n.01")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

    excl = {wn.synset(s) for s in EXCLUDE_SYNSETS}
    mammal, animal = wn.synset("mammal.n.01"), wn.synset("animal.n.01")
    vehicle, artifact = wn.synset("vehicle.n.01"), wn.synset("artifact.n.01")

    group = {}
    for i, (w, name) in enumerate(zip(wnids, names)):
        s = wn.synset_from_pos_and_offset("n", int(w[1:]))
        anc = set(s.closure(lambda x: x.hypernyms())) | \
              set(s.closure(lambda x: x.instance_hypernyms())) | {s}
        if anc & excl or any(k in name for k in EXTRA_EXCLUDE):
            group[i] = None
        elif mammal in anc:
            group[i] = "mammal"
        elif vehicle in anc:
            group[i] = "vehicle"
        elif animal in anc:
            group[i] = "other_animal"
        elif artifact in anc:
            group[i] = "object"
        else:
            group[i] = None
    return group


def report_groups(group, names, path):
    out = {}
    for _, key, need in LABELS:
        idx = [i for i, g in group.items() if g == key]
        out[key] = {"n_classes": len(idx), "max_images": 50 * len(idx),
                    "needed": need, "classes": [names[i] for i in idx]}
    out["excluded"] = [names[i] for i, g in group.items() if g is None]
    Path(path).write_text(json.dumps(out, indent=1))
    print("\ngroup               classes  max images  needed")
    for _, key, need in LABELS:
        o = out[key]
        print(f"  {key:<16}{o['n_classes']:>7}{o['max_images']:>12}{need:>8}")
        if o["max_images"] < need:
            sys.exit(f"!! group '{key}' cannot supply {need} images")
    print(f"  (excluded / unused: {len(out['excluded'])} classes)")
    print(f"full class lists written to {path} — skim it before training.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/control_set")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (keep fixed)")
    ap.add_argument("--classes-only", action="store_true",
                    help="only build and print the class grouping; download nothing")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = get_token()
    if not token and not args.classes_only:
        sys.exit("No HF_TOKEN. Add it as a Colab secret (Notebook access ON).")

    wnids, names = load_wnids_and_names(token)
    group = group_classes(wnids, names)
    report_groups(group, names, out / "control_classes.json")
    if args.classes_only:
        return

    from huggingface_hub import HfApi, hf_hub_download

    shards = sorted(f for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
                    if "validation-" in f and f.endswith(".parquet"))
    print(f"{len(shards)} validation shards available")

    quota = {key: need for _, key, need in LABELS}
    dirname = {key: d for d, key, _ in LABELS}
    for d in dirname.values():
        (out / d).mkdir(exist_ok=True)
    taken = {key: 0 for key in quota}
    rows = []

    for si, shard in enumerate(shards):
        if all(taken[k] >= quota[k] for k in quota):
            break
        p = hf_hub_download(REPO, shard, repo_type="dataset", token=token)
        df = pd.read_parquet(p)
        rng = np.random.default_rng(args.seed + si)
        grp = df["label"].map(group)
        print(f"{shard}: {len(df)} images")
        for key in quota:
            need = quota[key] - taken[key]
            if need <= 0:
                continue
            cand = np.flatnonzero((grp == key).to_numpy())
            rng.shuffle(cand)
            got = 0
            for r in cand:
                if got >= need:
                    break
                img = df["image"].iloc[r]
                data = img["bytes"]
                try:
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    assert min(im.size) >= 64
                except Exception:
                    continue
                fn = f"{dirname[key]}_{taken[key]:05d}.jpg"
                (out / dirname[key] / fn).write_bytes(data)
                lab = int(df["label"].iloc[r])
                rows.append({"file": f"{dirname[key]}/{fn}", "label": dirname[key],
                             "shard": shard, "row": int(r), "imagenet_label": lab,
                             "wnid": wnids[lab], "class_name": names[lab]})
                taken[key] += 1
                got += 1
            print(f"   {dirname[key]:<16}+{got:>4}   total {taken[key]}/{quota[key]}")
        del df

    short = {k: quota[k] - taken[k] for k in quota if taken[k] < quota[k]}
    if short:
        sys.exit(f"!! ran out of shards, still short: {short}")

    pd.DataFrame(rows).to_csv(out / "control_manifest.csv", index=False)
    total = sum(taken.values())
    print(f"\nDONE: {total} images in {out}")
    for d, key, need in LABELS:
        n = len(list((out / d).glob("*.jpg")))
        print(f"   {d:<16}{n:>6}  (target {need})")
    assert total == sum(n for *_, n in LABELS)
    print("Next: point pretrain_cls.py --data at this folder for the control arm.")


if __name__ == "__main__":
    main()

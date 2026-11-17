import sys
from pathlib import Path

import pandas as pd


DRIVE = Path("/content/drive/MyDrive")

IMAGE_LIST = DRIVE / "GLS_image_list.xlsx"
IMAGE_LIST_SHEET = "Table2"
ID_MAP = DRIVE / "image_id_to_original_filename_HC.xlsx"
ID_MAP_SHEET = "image_id_to_original_filena (2)"

DATASET_ROOT = DRIVE / "yolo_seg_800sq"
DATASET_SPLITS = ["train", "val", "test"]
DATASET_PATTERN = "{split}/images"

OUT_CSV = DRIVE / "logo_groups.csv"

MANUAL = {
    "1516": "cedara_2019",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def assign_group(folder: str) -> str:
    f = str(folder)
    if "Baynesfield" in f:
        return "baynesfield_2009"
    if "2012-03" in f:
        return "cedara_2012"
    if "Cedara" in f:
        return "cedara_2019"
    if "Delmas" in f or "Richard Kotze" in f or "RK photos" in f:
        return "delmas"
    return "UNRESOLVED"


def main():
    for p in (IMAGE_LIST, ID_MAP):
        if not p.exists():
            sys.exit(f"Missing spreadsheet: {p}")

    idmap = pd.read_excel(ID_MAP, sheet_name=ID_MAP_SHEET)
    idmap["img_path"] = idmap["img_path"].astype(str)
    idmap["source_folder"] = idmap["img_path"].str.split("/").str[3]
    idmap["original_filename"] = idmap["img_path"].str.split("/").str[-1]

    lst = pd.read_excel(IMAGE_LIST, sheet_name=IMAGE_LIST_SHEET)
    lst["stem"] = lst["file_name"].astype(str).str.split(".").str[0]
    lst["id"] = pd.to_numeric(lst["stem"], errors="coerce")

    unmatched_ids = lst["id"].isna().sum()
    if unmatched_ids:
        print(f"!! {unmatched_ids} row(s) in the image list have a "
              f"non-numeric filename and cannot be joined")

    m = lst.merge(idmap[["id", "img_path", "source_folder", "original_filename"]],
                  on="id", how="left")

    missing = m["source_folder"].isna().sum()
    print(f"image list rows: {len(lst)}    joined to a source folder: "
          f"{len(m) - missing}")
    if missing:
        print(f"!! {missing} row(s) did not join — these cannot be grouped:")
        for s in m[m["source_folder"].isna()]["file_name"].head(10):
            print(f"     {s}")

    m["group"] = m["source_folder"].apply(assign_group)

    applied = []
    for stem, grp in MANUAL.items():
        hit = m["stem"] == str(stem)
        if hit.any():
            before = m.loc[hit, "group"].iloc[0]
            m.loc[hit, "group"] = grp
            m.loc[hit, "group_source"] = "manual"
            applied.append(f"{stem}: {before} -> {grp}")
    m["group_source"] = m.get("group_source", pd.Series(index=m.index, dtype=object))
    m["group_source"] = m["group_source"].fillna("folder")

    if applied:
        print("\nmanual assignments:")
        for a in applied:
            print(f"  {a}")

    still = m[m["group"] == "UNRESOLVED"]
    if len(still):
        print(f"\n!! {len(still)} image(s) still UNRESOLVED — add them to MANUAL:")
        for _, r in still.iterrows():
            print(f"     {r['file_name']}   {r['img_path']}")

    dataset_stems = set()
    for s in DATASET_SPLITS:
        d = DATASET_ROOT / DATASET_PATTERN.format(split=s)
        if d.is_dir():
            dataset_stems |= {p.stem.lower() for p in d.iterdir()
                              if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                              and not p.name.startswith(".")}

    if not dataset_stems:
        print(f"\n!! Could not read the dataset at {DATASET_ROOT}. "
              f"Falling back to label_status.")
        keep = m[m["label_status"].isin(["LABELED", "REVIEWED"])].copy()
    else:
        print(f"\ndataset on disk: {len(dataset_stems)} images")
        keep = m[m["stem"].str.lower().isin(dataset_stems)].copy()
        not_in_sheet = dataset_stems - set(keep["stem"].str.lower())
        if not_in_sheet:
            print(f"!! {len(not_in_sheet)} image(s) on disk are NOT in the "
                  f"spreadsheet and will have no group:")
            for s in sorted(not_in_sheet)[:10]:
                print(f"     {s}")

    print(f"manifest rows: {len(keep)}")

    print("\n" + "=" * 64)
    print("GROUPS")
    print("=" * 64)
    vc = keep["group"].value_counts()
    tot = len(keep)
    hdr = f"{'group':<20}{'images':>8}{'%':>8}{'lesions':>10}{'mean/img':>10}{'max':>7}"
    print(hdr)
    print("-" * len(hdr))
    for g in sorted(vc.index):
        sub = keep[keep["group"] == g]
        n = len(sub)
        les = sub["gls_instance_count"].sum()
        print(f"{g:<20}{n:>8}{100*n/tot:>7.1f}%{les:>10,}"
              f"{les/n:>10.1f}{sub['gls_instance_count'].max():>7}")
    print("-" * len(hdr))
    print(f"{'total':<20}{tot:>8}{100:>7.1f}%"
          f"{keep['gls_instance_count'].sum():>10,}")

    print("\nSource folders feeding each group:")
    for g in sorted(vc.index):
        sub = keep[keep["group"] == g]
        for f, n in sub["source_folder"].value_counts().items():
            print(f"  {g:<20} {n:>4}  {f}")

    print("\n" + "=" * 64)
    print("LOGO FOLDS  (val = 15% of the training pool)")
    print("=" * 64)
    print(f"{'hold out':<20}{'test':>7}{'pool':>7}{'train':>7}{'val':>6}")
    print("-" * 47)
    for g in sorted(vc.index):
        n = vc[g]
        pool = tot - n
        val = round(pool * 0.15)
        print(f"{g:<20}{n:>7}{pool:>7}{pool-val:>7}{val:>6}")

    out = keep[["stem", "file_name", "group", "group_source", "source_folder",
                "original_filename", "gls_instance_count", "label_status"]].copy()
    out = out.sort_values(["group", "stem"])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(out)} rows)")
    print("Check the group counts above before building the folds.")


if __name__ == "__main__":
    main()

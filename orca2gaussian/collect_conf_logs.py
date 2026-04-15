#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _unique_path(path: Path) -> Path:
    """Return a non-colliding path by appending _N when needed."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def collect_logs(input_root: Path, output_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Move conf_*.log files into a single folder using subfolder_conf.log naming."""
    moved = 0
    skipped = 0

    conf_dirs = sorted(p for p in input_root.rglob("conf_*") if p.is_dir())
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for conf_dir in conf_dirs:
        src_log = conf_dir / f"{conf_dir.name}.log"
        if not src_log.exists():
            skipped += 1
            continue

        sub_folder = conf_dir.parent.name
        target_name = f"{sub_folder}_{conf_dir.name}.log"
        dst_log = _unique_path(output_dir / target_name)

        if dry_run:
            print(f"[DRY RUN] {src_log} -> {dst_log}")
        else:
            src_log.rename(dst_log)
            print(f"Moved: {src_log} -> {dst_log}")
        moved += 1

    return moved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move every conf_XX/conf_XX.log under input_root into output_dir "
            "and rename to sub_folder_conf_XX.log"
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        help="Root folder containing many subfolders with conf_* directories",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Single destination folder for collected logs",
    )
    parser.add_argument(
        "--input_root",
        dest="input_root_opt",
        type=Path,
        help="Root folder containing many subfolders with conf_* directories",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_opt",
        type=Path,
        help="Single destination folder for collected logs",
    )
    parser.add_argument(
        "--output_root",
        dest="output_dir_opt",
        type=Path,
        help="Alias of --output_dir for compatibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview move operations without changing files",
    )
    args = parser.parse_args()

    input_root_raw = args.input_root_opt or args.input_root
    output_dir_raw = args.output_dir_opt or args.output_dir

    if input_root_raw is None or output_dir_raw is None:
        parser.error(
            "Provide input and output paths either as positional arguments or with --input_root/--output_dir."
        )

    input_root = input_root_raw.resolve()
    output_dir = output_dir_raw.resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")

    moved, skipped = collect_logs(input_root, output_dir, dry_run=args.dry_run)
    print(f"Done. moved={moved}, skipped_missing_log={skipped}, output={output_dir}")


if __name__ == "__main__":
    main()

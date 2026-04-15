from pathlib import Path
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description="Convert ORCA conf_* results to Gaussian log using orca_to_gaussian_log.py")
parser.add_argument(
    "--orca-folder",
    type=str,
    required=True,
    help="Path to ORCA root folder containing conf_* subfolders. Can be absolute or relative.",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing files if present.",
)
args = parser.parse_args()  

# Set your ORCA root folder here (contains many conf_* subfolders)
orca_folder = Path(args.orca_folder).expanduser().resolve()
script_path = Path("orca2gaussian/orca_to_gaussian_log.py").resolve()

if not script_path.exists():
    raise FileNotFoundError(f"Script not found: {script_path}")
if not orca_folder.exists():
    raise FileNotFoundError(f"Folder not found: {orca_folder}")

conf_dirs = sorted([p for p in orca_folder.rglob("conf_*") if p.is_dir()])

if not conf_dirs:
    print(f"No conf_* folders found in: {orca_folder.resolve()}")
else:
    for conf_dir in conf_dirs:
        log_file = conf_dir / f"{conf_dir.name}.log"
        if log_file.exists() and not args.overwrite:
            print(f"Log file already exists, skipping: {log_file}")
            continue
        try:
            subprocess.run(
                [sys.executable, str(script_path), str(conf_dir), "-o", str(log_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"Generated: {log_file}")
        except subprocess.CalledProcessError as exc:
            print(f"Failed for {conf_dir}: {exc.stderr.strip() or exc}")
"""Record ORCA completion status into the molecule's <inchikey>/metadata.json.

Called from the tail of slurm-calcParam.sh, from within a conf_N directory,
right after the orca command:

    $(which orca) calcParam.inp >> calcParam.out
    python ../../mark_orca_done.py --exit-code $?

A conformer is only marked "completed" when ORCA both exited 0 *and* wrote its
"****ORCA TERMINATED NORMALLY****" banner -- ORCA can exit 0 after a Compound
sub-job has quietly died, so the exit code alone is not enough.

Safe to run concurrently with the other conformers' completion jobs for the
same molecule (metadata_utils.update_metadata locks the shared file).
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_utils

TERMINATION_BANNER = "ORCA TERMINATED NORMALLY"
RUN_TIME_RE = re.compile(r'^TOTAL RUN TIME:\s*(.+?)\s*$', re.MULTILINE)


def read_tail(path, nbytes=8192):
    """Last nbytes of a (potentially very large) ORCA output, as text."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exit_code_positional", nargs="?", type=int, default=None,
                        help=argparse.SUPPRESS)  # back-compat with `mark_orca_done.py $?`
    parser.add_argument("--exit-code", type=int, default=None,
                        help="Exit status of the orca command.")
    parser.add_argument("--status", default=None,
                        help="Force a status (e.g. 'timeout') instead of deriving "
                             "it from the exit code and the ORCA output.")
    parser.add_argument("--output", default="calcParam.out",
                        help="ORCA output file to inspect. Default: calcParam.out")
    args = parser.parse_args()

    exit_code = args.exit_code if args.exit_code is not None else args.exit_code_positional
    conformer = os.path.basename(os.getcwd())

    terminated_normally = None
    run_time = None
    if os.path.exists(args.output):
        tail = read_tail(args.output)
        terminated_normally = TERMINATION_BANNER in tail
        m = RUN_TIME_RE.search(tail)
        if m:
            run_time = m.group(1)

    if args.status:
        status = args.status
    elif exit_code == 0 and terminated_normally:
        status = "completed"
    else:
        status = "failed"

    metadata_utils.update_metadata("..", {
        "dft": {"conformers": {conformer: {
            "status": status,
            "exit_code": exit_code,
            "orca_terminated_normally": terminated_normally,
            "orca_run_time": run_time,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }}},
    })
    print(f"{conformer}: status={status} exit_code={exit_code} "
          f"terminated_normally={terminated_normally}")


if __name__ == "__main__":
    main()

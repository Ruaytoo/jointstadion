import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp1_recovery.py",
    "exp2_estimators.py",
    "exp3_paper_baseline.py",
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n===== {script} =====", flush=True)
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)])
        if rc != 0:
            print(f"{script} FAILED (rc={rc})", file=sys.stderr)
            sys.exit(rc)
    print("\nall experiments completed")

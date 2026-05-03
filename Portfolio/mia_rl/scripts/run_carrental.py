import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    from mia_rl.experiments.carrental import run_carrental

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "carrental"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_carrental(output_dir)

if __name__ == "__main__":
    main()
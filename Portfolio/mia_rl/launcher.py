from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

SCRIPTS_DIR = ROOT / "scripts"


# def clear():

#     import os

#     os.system("cls" if os.name == "nt" else "clear")


def main():

    while True:

        #clear()

        scripts = sorted(
            SCRIPTS_DIR.glob("run_*.py")
        )

        if not scripts:
            print("No experiment scripts found.")
            return

        print("\n===================================")
        print(" RL Experiment Launcher")
        print("===================================\n")

        for idx, script in enumerate(scripts, start=1):

            name = (
                script.stem
                .replace("run_", "")
                .replace("_", " ")
                .title()
            )

            print(f"{idx}. {name}")

        print("\n0. Exit")

        try:

            choice = int(
                input("\nSelect experiment: ")
            )

            if choice == 0:
                print("\nGoodbye!\n")
                break

            selected = scripts[choice - 1]

        except (ValueError, IndexError):

            print("\nInvalid selection.")
            input("\nPress Enter to continue...")
            continue

        print(f"\nRunning: {selected.name}\n")

        subprocess.run(
            [sys.executable, str(selected)],
        )

        input("\nExperiment finished. Press Enter to return to menu...")


if __name__ == "__main__":
    main()
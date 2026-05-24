from pathlib import Path


def save_plot(
    fig,
    output_dir: Path,
    filename: str,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_dir / filename,
        dpi=150,
        bbox_inches="tight",
    )
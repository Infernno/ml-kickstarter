from pathlib import Path

from dvc.repo import Repo


def ensure_data(path: str | Path, remote: str) -> None:
    """Restore a DVC-tracked input CSV when it is missing."""
    data_path = Path(path)
    if data_path.is_file():
        return

    with Repo() as repo:
        repo.pull(targets=[str(data_path)], remote=remote)


def ensure_model(
    directory: str | Path, remote: str, required_files: tuple[str, ...]
) -> None:
    """Restore the DVC-tracked model directory if a required artifact is absent."""
    directory = Path(directory)
    if all((directory / filename).is_file() for filename in required_files):
        return
    with Repo() as repo:
        repo.pull(targets=[str(directory)], remote=remote)
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Model files are missing after DVC pull: {missing}")

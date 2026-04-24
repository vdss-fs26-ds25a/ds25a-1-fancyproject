from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile


DEFAULT_KEEP = ("design_tree.json", "output.json", "design_low_level.json")


def iter_design_dirs(root: Path):
    for path in sorted(root.glob("design_*")):
        if path.is_dir():
            yield path


def slim_from_directory(source: Path, target: Path, keep_files: tuple[str, ...]) -> int:
    count = 0
    for design_dir in iter_design_dirs(source):
        target_dir = target / design_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = False
        for filename in keep_files:
            src_file = design_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, target_dir / filename)
                copied = True
        if copied:
            count += 1
    return count


def slim_from_zip(zip_path: Path, target: Path, keep_files: tuple[str, ...]) -> int:
    extracted_designs: set[str] = set()
    keep_set = set(keep_files)
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue
            design_name = next((part for part in parts if part.startswith("design_")), None)
            if design_name is None:
                continue
            filename = parts[-1]
            if filename not in keep_set:
                continue
            target_dir = target / design_name
            target_dir.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target_dir / filename, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_designs.add(design_name)
    return len(extracted_designs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a slim AircraftVerse dataset containing only the files needed for analysis."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Path to an already-unzipped AircraftVerse folder such as data_raw/AircraftVerse_1.",
    )
    parser.add_argument(
        "--zip-files",
        type=Path,
        nargs="*",
        help="One or more AircraftVerse zip files. Files are extracted selectively without full unzip.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Destination directory for the slim dataset.",
    )
    parser.add_argument(
        "--keep-files",
        nargs="*",
        default=list(DEFAULT_KEEP),
        help="Files to preserve inside each design folder.",
    )
    args = parser.parse_args()

    if not args.source_dir and not args.zip_files:
        parser.error("Provide at least --source-dir or --zip-files.")

    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    keep_files = tuple(args.keep_files)

    total = 0
    if args.source_dir:
        total += slim_from_directory(args.source_dir, target_dir, keep_files)
    if args.zip_files:
        for zip_path in args.zip_files:
            total += slim_from_zip(zip_path, target_dir, keep_files)

    print(f"Prepared slim dataset in {target_dir}")
    print(f"Copied {total} design folders")
    print(f"Kept files: {', '.join(keep_files)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aircraft_dashboard import build_aircraft_data_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the compact processed AircraftVerse table used by the dashboard."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data_raw/aircraftverse_slim"),
        help="Slim AircraftVerse folder containing one directory per design.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/aircraft_designs.csv.gz"),
        help="Compressed processed table written for app deployment.",
    )
    args = parser.parse_args()

    frame = build_aircraft_data_from_directory(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression="gzip")

    print(f"Wrote {len(frame):,} rows to {args.output}")
    print(f"Columns: {len(frame.columns)}")


if __name__ == "__main__":
    main()

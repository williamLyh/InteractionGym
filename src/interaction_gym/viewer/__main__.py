"""Build a viewer page from saved episode records.

    uv run python -m interaction_gym.viewer runs/*.json -o viewer.html
"""

import argparse

from ..traj import load
from . import export_html


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m interaction_gym.viewer")
    p.add_argument("records", nargs="+", help="JSON files written by interaction_gym.traj.save")
    p.add_argument("-o", "--out", default="viewer.html")
    args = p.parse_args()
    records = [r for path in args.records for r in load(path)]
    print(f"{len(records)} cases -> {export_html(records, args.out)}")


if __name__ == "__main__":
    main()

"""Shared output-directory arguments."""

import os
from pathlib import Path


def add_output_argument(parser):
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("FASTMRI_OUT_ROOT", ".")),
        help="Processed data root (default: FASTMRI_OUT_ROOT or current directory)",
    )

import sys

import nrrd
import SimpleITK as sitk
import numpy as np
from lama import common
import os
import argparse
from pathlib import Path

spaced = "spaced"


def run(Target_dir, Spacing):
    volpaths = common.get_file_paths(Target_dir)
    print(volpaths)
    for volpath in volpaths:
        img = sitk.ReadImage(volpath)
        img.SetSpacing(Spacing)
        sitk.WriteImage(img, str(Target_dir / spaced / os.path.basename(volpath)), True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Init_spacing")
    parser.add_argument('-i', '--input_folder', dest='target_dir', help='Raw NRRD directory', required=True,
                        type=str)
    parser.add_argument('-s', '--spacing', dest='spacing', help='directory of LAMA target (i.e. pop avg)',
                        required=True,
                        type=str)
    args = parser.parse_args()

    spacing = [float(i) for i in args.spacing.split(",")]

    run(Path(args.target_dir), spacing)

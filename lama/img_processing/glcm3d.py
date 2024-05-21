#!/usr/bin/python

"""
Implement a 3D GLCM for use in the registration  pipeline

"""

import numpy as np
import SimpleITK as sitk
import os
from logzero import logger as logging
import argparse

try:
    from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, gldm
except ImportError:
    logging.warn('pyradiomics not installed. No glcms will be made')
    pyrad_installed = False
else:
    pyrad_installed = True

from lama import common
import yaml
from pathlib import Path

MAXINTENSITY = 255
# GLCM constants
CHUNK_SIZE = 10
GLCM_BINS = 8


# SCRIPT_DIR = dirname(realpath(__file__))
# PATH_TO_ITK_GLCM = join(SCRIPT_DIR, '../dev/texture/GLCMItk/LamaITKTexture')


def pyradiomics_glcm(vol_dir, out_dir, mask_dir, feature='Contrast'):
    """
    Create glcm and xtract features. Spit out features per chunk as a 1D numpy array
    This 1d array can be reassembled into a 3D volume using common.rebuid_subsamlped_output

    Parameters
    ----------


    """
    if not pyrad_installed:
        return

    settings = {'binWidth': 4,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None}

    vol_paths = common.get_file_paths(vol_dir)

    mask_paths = common.get_file_paths(mask_dir)

    out_dir = Path(out_dir)

    # glcm_mask = np.ones([chunksize] * 3)  # No glcm mask. Set all to 1s. Needed for pyradiomics
    # glcm_mask_img = sitk.GetImageFromArray(glcm_mask)

    for i, path in enumerate(vol_paths):
        vol = common.LoadImage(path).img
        mask = common.LoadImage(mask_paths[i]).img

        gldm_matrix = gldm.RadiomicsGLDM(inputImage=vol, mask=mask).P_gldm

        gldm_test = sitk.GetImageFromArray(gldm_matrix)

        gldm_file_name = out_dir / os.path.basename(path)

        sitk.WriteImage(gldm_test, gldm_file_name, useCompression=True)

        # result = []
        #


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create textures")
    parser.add_argument('-i', '--input_folder', dest='target_dir', help='Raw NRRD directory', required=True,
                        type=str)
    parser.add_argument('-m', '--mask_folder', dest='mask_dir', help='Raw NRRD directory', required=True,
                        type=str)

    parser.add_argument('-o', '--output', dest='out_dir', help='directory of LAMA target (i.e. pop avg)',
                        required=True,
                        type=str)
    args = parser.parse_args()

    pyradiomics_glcm(args.target_dir, args.out_dir, args.mask_dir)

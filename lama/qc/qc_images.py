"""
Make QC images of the registered volumes
"""

from os.path import splitext, basename, join
from pathlib import Path
from typing import Union, List

import SimpleITK as sitk
from logzero import logger as logging
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from PIL import Image
from scipy.ndimage import zoom

from lama import common
from lama.registration_pipeline.validate_config import LamaConfig
from lama.elastix import IGNORE_FOLDER
from lama.paths import SpecimenDataPaths

MOVING_RANGE = (0, 180)  # Rescale the moving image to these values for the cyan/red overlay


def make_qc_images(lama_specimen_dir: Path, target: Path, outdir: Path):
    """
    Generate mid-slice images for quick qc of registration process.

    Parameters
    ----------
    config
        The lama config
    lama_specimen_dir
        The registration outdir. Should contain an 'output' folder
    registerd_midslice_outdir
        Where to place the midslices
    inverted_label_overlay_outdir
        Location of inverted labels
        If None, the inverted labels will not exist
    cyan_red_dir
        Where to put target(cyan) moving(red) overlays
    target
        The target image to display in cyan

    Make qc images from:
        The final registration stage.
            What the volumes look like after registration
        The rigidly-registered images with the inverted labels overlaid
            This is a good indicator of regsitration accuracy

    """
    target = common.LoadImage(target).array
    # Make qc images for all stages of registration including any resolution images
    paths = SpecimenDataPaths(lama_specimen_dir)

    # Order output dirs by qc type
    red_cyan_dir = outdir / 'red_cyan_overlays'
    red_cyan_dir.mkdir(exist_ok=True)

    for i, (stage, img_path) in enumerate(paths.registration_imgs()):
        img = common.LoadImage(img_path).array
        overlay_cyan_red(target, img, red_cyan_dir, img_path.stem, i, stage)



    # # Get the inverted labels dir, that will map onto the first stage registration
    # if not inverted_label_overlay_outdir:
    #     inverted_label_dir = None
    # else:
    #     inverted_label_id = config['registration_stage_params'][1]['stage_id']
    #     inverted_label_dir = lama_specimen_dir / 'inverted_labels' / inverted_label_id
    #
    # generate(first_stage_dir,
    #          final_stage_dir,
    #          inverted_label_dir,
    #          registerd_midslice_outdir,
    #          inverted_label_overlay_outdir,
    #          cyan_red_dir,
    #          target)


def generate(first_stage_reg_dir: Path,
             final_stage_reg_dir: Path,
             inverted_labeldir: Path,
             out_dir_vols: Path,
             out_dir_labels: Path,
             out_dir_cyan_red: Path,
             target: Path):
    """
    Generate a mid section slice to keep an eye on the registration process.

    When overlaying the label maps, it depends on the registered volumes and inverted label maps being named identically
    """

    target = common.LoadImage(target).array
    target_slice = target[:, :, target.shape[2] // 2]

    # Make mid-slice images of the final registered volumes
    if final_stage_reg_dir:
        p = common.get_file_paths(final_stage_reg_dir, ignore_folder=IGNORE_FOLDER)
        if not p:
            logging.warn("Can't find output files for {}".format(final_stage_reg_dir))
            return

        for vol_path in p:

            # label_path = inverted_label_dir / vol_path.stem /vol_path.name

            vol_reader = common.LoadImage(vol_path)
            # label_reader = common.LoadImage(label_path)

            if not vol_reader:
                logging.error('error making qc image: {}'.format(vol_reader.error_msg))
                continue

            cast_img = sitk.Cast(sitk.RescaleIntensity(vol_reader.img), sitk.sitkUInt8)
            arr = sitk.GetArrayFromImage(cast_img)
            slice_ = arr[:, :, arr.shape[2] // 2]

            base = splitext(basename(vol_reader.img_path))[0]
            out_path = join(out_dir_vols, base + '.png')
            imsave(out_path, np.flipud(slice_))

            rc_slice = overlay_cyan_red(target_slice, slice_)
            imsave(out_dir_cyan_red / (base + '.png'), rc_slice)

    # Make a sagittal mid-section overlay of inverted labels on input (rigidly-aligned) specimen
    if first_stage_reg_dir and inverted_labeldir:
        for vol_path in common.get_file_paths(first_stage_reg_dir, ignore_folder=IGNORE_FOLDER):

            vol_reader = common.LoadImage(vol_path)

            if not vol_reader:
                logging.error(f'cannnot create qc image from {vol_path}')
                return

            label_path = inverted_labeldir / vol_path.stem / vol_path.name

            if label_path.is_file():
                label_reader = common.LoadImage(label_path)

                if not label_reader:
                    logging.error(f'cannot create qc image from label file {label_path}')
                    return

                cast_img = sitk.Cast(sitk.RescaleIntensity(vol_reader.img), sitk.sitkUInt8)
                arr = sitk.GetArrayFromImage(cast_img)
                slice_ = np.flipud(arr[:, :, arr.shape[2] // 2])
                l_arr = label_reader.array
                l_slice_ = np.flipud(l_arr[:, :, l_arr.shape[2] // 2])

                base = splitext(basename(label_reader.img_path))[0]
                out_path = join(out_dir_labels, base + '.png')
                blend_8bit(slice_, l_slice_, out_path)
            else:
                logging.info('No inverted label found. Skipping creation of inverted label-image overlay')


def blend_8bit(gray_img: np.ndarray, label_img: np.ndarray, out: Path, alpha: float=0.18):

    overlay_im = sitk.LabelOverlay(sitk.GetImageFromArray(gray_img),
                                   sitk.GetImageFromArray(label_img),
                                   alpha,
                                   0)
    sitk.WriteImage(overlay_im, out)


def overlay_cyan_red(target: np.ndarray,
                     specimen: np.ndarray,
                     out_dir: Path, name: str,
                     img_num: int,
                     stage_id: str) -> List[np.ndarray]:
    """
    Create a cyan red overlay
    Parameters
    ----------
    target
    specimen`
    img_num
        A number to prefix onto the qc image so that when browing a folder the images will be sorteed
    Returns
    -------
    0: axial
    1: coronal
    2: sagittal
    """
    ax_out = out_dir / 'axial'
    ax_out.mkdir(exist_ok=True)
    cor_out = out_dir / 'coronal'
    cor_out.mkdir(exist_ok=True)
    sag_out = out_dir / 'sagittal'
    sag_out.mkdir(exist_ok=True)

    if target.shape != specimen.shape:
        raise ValueError('target and specimen must be same shape')

    specimen = np.clip(specimen, 0, 255)
    specimen = rescale_intensity(specimen, out_range=MOVING_RANGE)

    def get_slices(img):
        slices = []
        for ax in [0,1,2]:
            slices.append(np.take(img, img.shape[ax] // 2, axis=ax))
        return slices

    def get_rgb(slice_1, slice_2):
        rgb = np.zeros([*slice_1.shape, 3], np.uint8)

        rgb[..., 0] = slice_1
        rgb[..., 1] = slice_2
        rgb[..., 2] = slice_2
        return rgb

    t = get_slices(target)
    s = get_slices(specimen)

    oris = [(ax_out, 'axial'), (cor_out, 'coronal'), (sag_out, 'sagittal')]

    # put slices in folders by orientation
    for i, (ori_out, ori_name) in enumerate(oris):
        rgb = get_rgb(s[i], t[i])
        rgb = np.flipud(rgb)
        imsave(ori_out / f'{img_num}_{stage_id}_{name}_{ori_name}.png', rgb)


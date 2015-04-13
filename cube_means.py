#!/usr/bin/env python

"""
Generate a statistics data file that can be used as input for vpv.py
"""
from simplejson import ordered_dict

import numpy as np
import SimpleITK as sitk
import argparse
import cPickle as pickle
import harwellimglib as hil
import sys
import os
import statsmodels.sandbox.stats.multicomp as multicomp
import tempfile
from scipy import stats
import math




def cube_vect_magnitude_mean(cube_of_vectors):
    """
    For a cube of deformation vectors, get the mean magnitude
    :param cube_of_vectors:
    :return: mean magnitude
    """
    vectors = []
    #Append each vector from the cube to a list
    for z in cube_of_vectors:
        for y in z:
            for vec in y:
                vectors.append(vec)
    #Get the mean vector. Then get the magnitude of it using np.linalg.norm
    return np.linalg.norm(np.mean(vectors, axis=0))


def cube_scalar_mean(cube_of_scalars):
    """
    For a cube of jacobian scalars, get the mean value
    :param cube_of_jac_scalars:
    :return:mean jacobian
    """
    return np.mean(cube_of_scalars)



def group_position_means(all_files_data):
    """
    Ggroup together mean magnitudes from each file by corresponding position.
    :param all_files_data: Data for all files at one cube position
    :return:pos_means, list
    """
    pos = [all_files_data[0][0]]  #=(x,y,z)
    all_means = []

    #add the each files cube average
    for deform_file_data in all_files_data:
        all_means.append(deform_file_data[1])

    pos.append(all_means)
    return pos


def run(WTs, mutants, analysis_type, chunksize, outfile):
    """
    :param jacobians:List of jacobian files
    :param deforms: List of defomation files
    :param outfile: Where to put the output
    :param threads: Max num of threads to use
    :param deform_file_pattern: defomation file regex finder in case these files are mixed in with others
    """

    if os.path.isdir(args.outfile):
        sys.exit("Supply an output file path not a directory")

    print('processing')

    wt_paths = hil.GetFilePaths(os.path.abspath(WTs))
    mut_paths = hil.GetFilePaths(os.path.abspath(mutants))

    if len(wt_paths) < 1:
        raise IOError("can't find volumes in {}".format(WTs))
    if len(mut_paths) < 1:
        raise IOError("can't find volumes in {}".format(mutants))

    print('### Wild types to process ###')
    print([os.path.basename(x) for x in wt_paths])
    print('### Mutants to process ###')
    print([os.path.basename(x) for x in mut_paths])

    vol_stats(wt_paths, mut_paths, analysis_type, chunksize, outfile)



def vol_stats(wts, muts, analysis_type, chunksize, outfile):

    memmapped_wts = memory_map_volumes(wts)
    memmapped_muts = memory_map_volumes(muts)

    # filename = os.path.basename(rawdata_file)
    # img = sitk.ReadImage(rawdata_file)
    # if data_type == 'intensities':
    #     print('normalizing')
    #     img = sitk.Normalize(img)
    zdim, ydim, xdim = memmapped_wts[0].shape[0:3]  # For vectors there's and extra dimension so can't just unpack

    # Create an array to store the
    pval_results = []
    positive_tscores = []

    for z in range(0, zdim - chunksize, chunksize):
        for y in range(0, ydim - chunksize, chunksize):
            for x in range(0, xdim - chunksize, chunksize):

                wt_means = get_mean_cube(memmapped_wts, z, y, x, chunksize, analysis_type)
                mut_means = get_mean_cube(memmapped_muts, z, y, x, chunksize, analysis_type)

                pval, is_positive = ttest(wt_means, mut_means)

                # pval_result_array[z:z+chunksize, y:y+chunksize, x:x+chunksize] = pval
                # tscore_is_positive[z:z+chunksize, y:y+chunksize, x:x+chunksize] = is_positive
                pval_results.append(pval)
                positive_tscores.append(is_positive)

    print("calculating FDR")
    # Make fdr-corrected array -  multipletests[1] result is the pvals_corrected aray
    fdr_results = multicomp.multipletests(pval_results)

    # Exand the array back up to the volume size - must be a more effient way of doing this
    fdr_vol = np.zeros(shape=memmapped_wts[0].shape, dtype=np.float32)

    small_array_index = 0

    for z in range(0, zdim - chunksize, chunksize):
        for y in range(0, ydim - chunksize, chunksize):
            for x in range(0, xdim - chunksize, chunksize):
                pval = pval_results[small_array_index]
                tscore_positive = positive_tscores[small_array_index]
                small_array_index += 1

                # Set low pvalues to nearer to 1. easier for doing colormaps in vpv
                if tscore_positive:
                    pval = 1 - pval
                else:
                    pval = -1 + pval

                fdr_vol[z:z+chunksize, y:y+chunksize, x:x+chunksize] = pval

    result_vol = sitk.GetImageFromArray(fdr_vol)
    sitk.WriteImage(result_vol, outfile)

def rescale_volume(vol):
    """
    For display of tstat volumes in Slicer, we should get rid of negative values
    :param vol : SimpleITK Image
    :return rescaled_vol: SimpleITK Image
    """
    arr = sitk.GetArrayFromImage(vol)
    min = arr.min()
    max_ = arr.max()

5

def get_mean_cube(arrays, z, y, x, chunksize, a_type):

    means = []
    for arr in arrays:

        if a_type == 'def':
            cube_mean = cube_vect_magnitude_mean(arr[z:z+chunksize, y:y+chunksize, x:x+chunksize])
        else:
            cube_mean = cube_scalar_mean(arr[z:z+chunksize, y:y+chunksize, x:x+chunksize])
        means.append(cube_mean)

    return means


def ttest(wt, mut):
    """
    Calculate the pvalue and the tstatistic for the wt and mut subsampled region

    Args:
       wt (list):  wildtype values
       mut (list): mutant values

    Returns:
       tuple: (pvalue(float), is_tscore_positive(bool)
    """
    tscore, pval = stats.ttest_ind(wt, mut)[0:2]

    # set the pvalue to negative if the t-statistic is negative
    if tscore < 0:
        is_tscore_positive = False
    else:
        is_tscore_positive = True

    # Set pval nan values to 1. This can happen if all values in tested arrays are 0
    if math.isnan(pval):
        pval = 1.0

    return pval, is_tscore_positive


def memory_map_volumes(vol_paths):
    """
    Create memory-mapped volumes
    """
    memmapped_vols = []
    for vp in vol_paths:
        img = sitk.ReadImage(vp)
        # if norm:
        #     img = sitk.Normalize(img)
        array = sitk.GetArrayFromImage(img)
        tempraw = tempfile.TemporaryFile(mode='wb+')
        array.tofile(tempraw)
        memmpraw = np.memmap(tempraw, dtype=np.float32, mode='r', shape=array.shape)

        memmapped_vols.append(memmpraw)

    return memmapped_vols



if __name__ == '__main__':

    # mpl = multiprocessing.log_to_stderr()
    # mpl.setLevel(multiprocessing.SUBDEBUG)

    parser = argparse.ArgumentParser("messageS")
    parser.add_argument('-c', dest='cubesize', type=int, help='Size of the sub array', required=True)
    parser.add_argument('-w', dest='wt_vols_dir', help='Folder containing WT data ', required=True)
    parser.add_argument('-m', dest='mut_vols_dir', help='Folder containing Mut data', required=True)
    parser.add_argument('-a', dest='analysis_type', help='<int, def, jac> Intensities, deformation fields, or spatial jacobians', required=True)
    parser.add_argument('-o', dest='outfile', help='', required=True)
    # parser.add_argument('-r', dest='registered_vols', help='Folder containing registered vols, for intensity difference'
    #                                                        ' analysis', default=None)
    # parser.add_argument('-o', dest='outfile', help='File to store pickle file of means/stdvs of vectors,intensities etc', required=True)
    # parser.add_argument('-t', dest='threads', type=int, help='How many threads to use', default=4)
    # parser.add_argument('-dp', dest='deform_file_pattern', help='String that is contained in the deform file names',
    #                     default='deformationField')
    # parser.add_argument('-jp', dest='jac_file_pattern', help='String that is contained in the jacobian file names',
    #                     default='spatialJacobian')

    args = parser.parse_args()

    run(args.wt_vols_dir, args.mut_vols_dir, args.analysis_type, args.cubesize, args.outfile)

# from lama.stats.permutation_stats.run_permutation_stats import run
import pytest
from pathlib import Path
from lama.scripts import lama_permutation_stats
from lama.stats.linear_model import lm_sm
from lama.stats.permutation_stats.run_permutation_stats import *
import lama.scripts.lama_permutation_stats

import random
from pathlib import Path
from datetime import date
import itertools
import pandas as pd
import numpy as np
from scipy.stats import zmap
from logzero import logger as logging
import yaml

from lama import common
from lama.stats.permutation_stats import distributions
from lama.stats.permutation_stats import p_thresholds
from lama.paths import specimen_iterator, get_specimen_dirs, LamaSpecimenData
from lama.qc.organ_vol_plots import make_plots, pvalue_dist_plots
from lama.common import write_array, read_array, init_logging, git_log, LamaDataException
from lama.stats.common import cohens_d
from lama.stats.penetrence_expressivity_plots import heatmaps_for_permutation_stats
from lama.stats.permutation_stats import run_permutation_stats

from lama.stats.permutation_stats.distributions import two_way_max_combinations, recursive_comb_maker, null_line
from lama.tests import (out_dir, wt_dir, mut_dir, treat_dir, inter_dir, label_meta, n_perm)

cfg = Path(
    "C:/Users/u5823099/Anaconda3/Lib/site-packages/lama/LAMA/lama/tests/configs/standard_stats/generate_data.toml")

stats_cfg = Path(
    "C:/Users/u5823099/Anaconda3/Lib/site-packages/lama/LAMA/lama/tests/configs/permutation_stats/perm_no_qc.yaml")


@pytest.mark.skip
def test_data_prep(two_way=True):
    """Just test if I can normalise stuff properly"""
    np.random.seed(999)
    init_logging(out_dir / 'stats.log')
    logging.info(git_log())
    logging.info(f'Running {__name__} with following commands\n{common.command_line_agrs()}')

    logging.info('Searching for staging data')
    wt_staging = get_staging_data(wt_dir)
    mut_staging = get_staging_data(mut_dir)

    logging.info('searching for organ volume data')
    wt_organ_vol = get_organ_volume_data(wt_dir)
    mut_organ_vol = get_organ_volume_data(mut_dir)
    if two_way:
        logging.info('Searching for two-way staging and organ volume data')
        treat_staging = get_staging_data(treat_dir)
        inter_staging = get_staging_data(inter_dir)
        treat_organ_vol = get_organ_volume_data(treat_dir)
        inter_organ_vol = get_organ_volume_data(inter_dir)
        two_way_data = [treat_staging, treat_organ_vol,
                        inter_staging, inter_organ_vol]

    data = prepare_data(wt_organ_vol,
                        wt_staging,
                        mut_organ_vol,
                        mut_staging,
                        label_meta=label_meta,
                        normalise_to_whole_embryo=True,
                        qc_file=None,
                        two_way=two_way,
                        two_way_data=two_way_data)

    # So pandas is a bit stupid as read/write of the file is causing type errors
    # How about we just look at
    print("Comparison")
    print(data)
    good_data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)
    # check that the data is the same as a checked csv file
    print(good_data)

    # print(data.astype(int).equals(data.astype(int)))


@pytest.mark.skip
def test_max_two_way_combinations():
    combs = two_way_max_combinations(num_wts=16)

    assert combs == 63063000


@pytest.mark.skip
def test_loop_iteration():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    fin_results = []
    results = []

    for perm in range(10):
        full_combination = recursive_comb_maker(lst, 3, i=1, recurs_results=[])
        # print(full_combination)
        print(full_combination[1])
        fin_results.append(full_combination)
        # reset result list
        results = []
    print(fin_results[1])


@pytest.mark.skip
def test_generate_two_way_combinations():
    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)
    # can't really test this as of yet
    wt_indx_combinations = distributions.generate_random_two_way_combinations(data, 20)
    return wt_indx_combinations


@pytest.mark.skip
def test_lm_sm():
    data = pd.read_csv("C:/Users/u5823099/Anaconda3/Lib/site-packages/lama/LAMA/lama/tests/working_on/data.csv")
    info = pd.read_csv("C:/Users/u5823099/Anaconda3/Lib/site-packages/lama/LAMA/lama/tests/working_on/treat_info.csv")
    data = data.to_numpy()
    p, t = lm_sm(data=data, info=info, use_staging=True, two_way=True)
    print("p = ", p)
    print("t = ", t)


@pytest.mark.skip
def test_two_way_null_line():
    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)
    # can't really test this as of yet
    baselines = data[data['line'] == 'baseline']
    wt_indx_combinations = distributions.generate_random_two_way_combinations(data, 20)
    results = null_line(wt_indx_combinations, baselines, num_perms=20, two_way=True)

    print(type(results['x3']), results['x3'][0][0], type(results['x3'][0][0]))


@pytest.mark.skip
def test_two_way_null():
    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)

    line_null, specimen_null = distributions.null(input_data=data, num_perm=3, two_way=True)
    print(type(line_null['3'][0]), line_null['3'][0][0], type(line_null['3'][0][0]))
    print(type(specimen_null['22'][0]), specimen_null['22'][0][0], type(specimen_null['22'][0][0]))


@pytest.mark.skip
def test_two_way_alt():
    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)
    line_alt, spec_alt, line_alt_t, spec_alt_t = distributions.alternative(data, two_way=True)
    print(type(line_alt['3'][0]), line_alt['3'][0][0], type(line_alt['3'][0][0]))
    print(type(spec_alt['3'][0]), spec_alt['3'][0][0], type(spec_alt['3'][0][0]))


@pytest.mark.skip
def test_two_way_p_thresholds():
    """
     Testing the p_thresholds calculation
     -------
     TODO: Add more tests for different cases
     """

    def strip_x(dfs):
        for df in dfs:
            df.columns = [x.strip('x') for x in df.columns]

        return df

    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)

    baselines = data[data['line'] == 'baseline']
    wt_indx_combinations = distributions.generate_random_two_way_combinations(data, 100)
    results = null_line(wt_indx_combinations, baselines, num_perms=100, two_way=True)
    line_null = strip_x([results])

    # line_null, specimen_null = distributions.null(input_data=data, num_perm=10, two_way=True)

    line_alt, spec_alt, line_alt_t, spec_alt_t = distributions.alternative(data, two_way=True)

    thresh = p_thresholds.get_thresholds(line_null, line_alt, two_way=True)

    thresh.to_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/spec_out_threshs.csv')

def test_two_spec_thresholds():
    two_way = True
    data = pd.read_csv('E:/Bl6_data/211014_g_by_back/permutation_stats/perm_output/input_data.csv', index_col=0)
    line_null, specimen_null = distributions.null(input_data=data, num_perm=5, two_way=True)

    line_alt, spec_alt, line_alt_t, spec_alt_t = distributions.alternative(data, two_way=True)

    specimen_inter_nulls = [array for label in specimen_null for array in label if len(array) == 3]
    print(specimen_inter_nulls)
    specimen_main_nulls = [array for label in specimen_null for array in label if len(array) == 1]
    specimen_geno_nulls, specimen_treat_nulls = np.vsplit(specimen_main_nulls, 2)

    specimen_inter_alt = [array for label in spec_alt for array in label if len(array) == 3]
    specimen_main_alt = [array for label in spec_alt for array in label if len(array) == 1]

    print(specimen_main_alt[0][0])

    specimen_geno_alt = [array for label in spec_alt for array in label if 'het' in specimen_main_alt['specimen']]
    specimen_treat_alt = [array for label in spec_alt for array in label if 'b6' in specimen_main_alt['specimen']]

    geno_thresholds = p_thresholds.get_thresholds(specimen_geno_nulls, specimen_geno_alt, two_way=two_way)
    treat_thresholds = p_thresholds.get_thresholds(specimen_treat_nulls, specimen_treat_alt, two_way=two_way)
    inter_thresholds = p_thresholds.get_thresholds(specimen_inter_nulls, specimen_inter_alt, two_way=two_way)



@pytest.mark.skip
def test_two_way_fdr_calc():
    wt_pvals = [np.array([0.23185364, 0.13273183, 0.17713702]),
                np.array([0.84430502, 0.47580432, 0.76066349]),
                np.array([0.29232564, 0.72754982, 0.50835225]),
                np.array([0.56155691, 0.3699144, 0.5404814]),
                np.array([0.63025429, 0.93275984, 0.64331066]),
                np.array([0.68972929, 0.8250321, 0.84710903]),
                np.array([0.53804087, 0.45815869, 0.55067356]),
                np.array([0.34301704, 0.33966628, 0.47430639]),
                np.array([0.27797228, 0.7698054, 0.12596086]),
                np.array([0.00737234, 0.14114093, 0.03099726]),
                np.array([0.18246004, 0.86991258, 0.38496491]),
                np.array([0.06500475, 0.11617978, 0.04889024]),
                np.array([0.87257678, 0.43532906, 0.75713247]),
                np.array([0.71441488, 0.8610815, 0.81531772]),
                np.array([0.08655808, 0.06769658, 0.31475581]),
                np.array([0.37445802, 0.69798106, 0.93924833]),
                np.array([0.09487213, 0.0446766, 0.05215738]),
                np.array([0.8475532, 0.79866281, 0.72804985]),
                np.array([0.87584695, 0.40026116, 0.28694841]),
                np.array([0.89372524, 0.86022629, 0.78422224])]
    mut_pvals = [np.array([6.63908071e-02, 5.99845408e-12, 5.85503383e-02])]


@pytest.mark.skip
def test_permutation_stats():
    """
    Run the whole permutation based stats pipeline.
    Copy the output from a LAMA registrations test run, and increase or decrease the volume of the mutants so we get
    some hits

    """
    lama_permutation_stats.run(stats_cfg)

    # # Without label meta file
    # output_no_metdata = permutation_stats_dir / 'output_with_hits_no_metadata'
    # output_no_metdata.mkdir(exist_ok=True)
    # lama_permutation_stats.run(wt_registration_dir / 'output', mut_registration_dir / 'output', output_no_metdata, num_perms,
    #                           label_map_path=label_map)

# @pytest.mark.notest
# def test_permutation_stats_with_qc_flaggs():
#     """
#     Run the permutations stats but include a specimen/organ-level qc file to exclude qc-flagged organs
#     """
#     num_perms = 5  # Would do 1000 or more normally
#     label_info = registration_root / 'target' / 'label_info.csv'
#     label_map = registration_root / 'target' / 'labels.nrrd'
#
#     for qc_file in qc_flags_dir.iterdir():
#
#         out_dir = permutation_stats_dir / qc_file.stem  # Intermediate results go here. Permutation distributions etc.
#         out_dir.mkdir()
#
#         if 'error' in str(qc_file):
#             # These qc flag files with errors in should raise a LamaDataError
#             with pytest.raises(LamaDataException):
#                 run_permutation_stats.run(wt_registration_dir, mut_registration_dir, out_dir, num_perms,
#                                           label_info=label_info, label_map_path=label_map, qc_file=qc_file)
#
#         else:
#             run_permutation_stats.run(wt_registration_dir, mut_registration_dir, out_dir, num_perms,
#                                   label_info=label_info, label_map_path=label_map, qc_file=qc_file)


#     thresh = p_thresholds.get_thresholds(null, alt)
#
#     assert thresh.loc[1, 'p_thresh'] == 0.02  # Gives a p-value threshold of 0.02
#     assert thresh.loc[2, 'p_thresh'] == 1.0    # Gives a p-value threshold of 1.0 as there are no low p-values in the alt distribution


# @pytest.mark.notest
# def test_annotate():
#     # Lines
#     alt_file = Path('/home/neil/git/lama/tests/test_data/stats_test_data/test_output/organ_vols_permutation/alt_line_dist_pvalues.csv')
#     thresholds_file = Path('/home/neil/git/lama/tests/test_data/stats_test_data/test_output/organ_vols_permutation/line_organ_p_thresholds.csv')
#     mutant_dir = Path('/home/neil/git/lama/tests/test_data/registration_test_data/mutant')
#
#     thresholds = pd.read_csv(thresholds_file, index_col=0)
#     alt = pd.read_csv(alt_file, index_col=0)
#
#     run_permutation_stats.annotate(thresholds, alt, mutant_dir)
#
#     # # Specimens
#     alt_file = Path('/home/neil/git/lama/tests/test_data/stats_test_data/test_output/organ_vols_permutation/alt_specimen_dist_pvalues.csv')
#     thresholds_file = Path('/home/neil/git/lama/tests/test_data/stats_test_data/test_output/organ_vols_permutation/specimen_organ_p_thresholds.csv')
#     mutant_dir = Path('/home/neil/git/lama/tests/test_data/registration_test_data/mutant')
#
#     thresholds = pd.read_csv(thresholds_file, index_col=0)
#     alt = pd.read_csv(alt_file, index_col=0)
#
#     run_permutation_stats.annotate(thresholds, alt, mutant_dir)

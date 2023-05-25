import pandas as pd
from mpi4py import MPI
import numpy as np
import statsmodels.formula.api as smf
from typing import List
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import itertools
def _two_way_null_line_thread(*args) -> List[float]:
    """
    same as _null_line_thread but for two way
    TODO: merge two_way and null_threads
    Parameters
    ----------
    data : pd.DataFrame
        Data for which null line is generated
    num_perms : int
        Number of permutations to generate
    wt_indx_combinations : dict
        Dictionary containing combinations of WT indices for each label
    label : str
        Name of the label column

    Returns
    -------
    line_p : list
        List of p-values for each permutation
    """
    # print('Generating null for', label)
    data, num_perms, wt_indx_combinations, label = args

    data = data.astype({label: float, 'staging': float})

    synthetics_sets_done = []

    line_p = []

    perms_done = 0

    # Get combinations of WT indices for current label
    indxs = wt_indx_combinations[label].values
    indxs = [eval(s) for s in indxs]

    formula = f'{label} ~ genotype * treatment + staging'
    for i, comb in enumerate(indxs):

        # set up genotype and treatment
        data.loc[:, 'genotype'] = 'wt'
        data.loc[:,'treatment'] = 'veh'

        # mains
        data.loc[data.index.isin(comb[0]), 'genotype'] = 'synth_mut'
        data.loc[data.index.isin(comb[1]), 'treatment'] = 'synth_treat'

        # interactions
        data.loc[data.index.isin(comb[2]), 'genotype'] = 'synth_mut'
        data.loc[data.index.isin(comb[2]), 'treatment'] = 'synth_treat'

        # _label_synthetic_mutants(data, n, synthetics_sets_done)

        perms_done += 1

        fit = smf.ols(formula=formula, data=data, missing='drop').fit()
        # get all pvals except intercept and staging

        # fit.pvalues is a series - theefore you have to use .index
        p = fit.pvalues[~fit.pvalues.index.isin(['Intercept','staging'])]
        #pvalues go in the order of genotype, treatment, interaction.
        line_p.append(p.values)
    return line_p


def main():
    parser = argparse.ArgumentParser("Parallelisation across multiple nodes for null_dist calculation")
    parser.add_argument('-n', dest='num_perms', help='number of permutations', required=True)
    parser.add_argument('-p', dest='input_dir', help='distribution path', required=True)



    args = parser.parse_args()

    _dir = Path(args.input_dir)
    num_perms = int(args.num_perms)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #added a memory map
    data = pd.read_csv(str(_dir/"data_for_null_calc.csv"), index_col=0, memory_map=True)

    #num_perms = ...  # specify the number of permutations
    wt_indx_combinations = pd.read_csv(str(_dir/"combs_for_null_calc.csv"), index_col=0, memory_map=True)  # specify the WT index combinations

    cols = data.columns.drop(['staging','genotype','treatment'])

    chunk_size = len(cols) // size
    start = rank * chunk_size
    end = start + chunk_size if rank < size - 1 else len(cols)

    # each rank processes a subset of the columns
    pvals_list = []
    for label in tqdm(cols[start:end]):
        pvals_list.append(_two_way_null_line_thread(data, num_perms, wt_indx_combinations, label))

    rank_filename = str(_dir / f"pdist_results_{rank}.csv")
    pd.DataFrame(pvals_list).to_csv(rank_filename)

    # synchronize all ranks before proceeding
    comm.Barrier()

    # rank 0 combines the results from all ranks into a single file
    if rank == 0:
        combined_df = pd.DataFrame()
        for i in range(size):
            # go through each
            rank_filename = str(_dir / f"pdist_results_{i}.csv")
            rank_df = pd.read_csv(rank_filename, memory_map=True)
            if i == 0:
                rank_df.to_csv(str(_dir / "pdist_results.csv"), mode='w', chunksize=500)
            else:
                rank_df.to_csv(str(_dir / "pdist_results.csv"), header=False, mode='a', chunksize=500)


if __name__ == '__main__':
    main()
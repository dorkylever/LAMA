
from pathlib import Path

import joblib
from lama.stats.heatmap import heatmap, clustermap
from lama.lama_radiomics.radiomics import radiomics_job_runner
from lama import common
import os
from lama.common import cfg_load
from lama.img_processing import normalise
from logzero import logger as logging
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pytest
import numpy as np
from lama.lama_radiomics import feature_reduction
from lama.scripts import lama_machine_learning
import pacmap
from lama.scripts import lama_permutation_stats
from lama.lama_radiomics import radiomics, rad_plotting
from lama.stats.penetrence_expressivity_plots import heatmaps_for_permutation_stats, filt_for_shared_feats
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

import SimpleITK as sitk
stats_cfg = Path(
    "C:/Users/Kyle/PycharmProjects/LAMA/lama/tests/configs/permutation_stats/perm_no_qc.yaml")
from lama.stats.permutation_stats.run_permutation_stats import get_radiomics_data


stats_cfg_v2 = Path(
    "C:/LAMA/lama/tests/configs/permutation_stats/perm_no_qc_just_ovs.yaml")



def test_denoising():
    file_path = Path("E:/220204_BQ_dataset/221218_BQ_run/registrations/rigid/flipped/200721_MPTLVo3_CT_4T1_Ms_D7_C1_002.nrrd")

    img = common.LoadImage(file_path).img
    f_out = "E:/220204_BQ_dataset/221218_BQ_run/test.nrrd"

    result = radiomics.denoise(img)
    sitk.WriteImage(result, f_out)



def test_radiomics():
        cpath =  Path('C:/LAMA/lama/tests/configs/lama_radiomics/radiomics_config.toml')
        c = cfg_load(cpath)

        target_dir = Path(c.get('target_dir'))

        labs_of_int = c.get('labs_of_int')

        norm_methods = c.get('norm_methods')

        norm_label = c.get('norm_label')

        spherify = c.get('spherify')

        use_roi = c.get('use_roi')

        print("use roi", use_roi)
        ref_vol_path = Path(c.get('ref_vol_path')) if c.get('ref_vol_path') is not None else None

        norm_dict = {
            "histogram": normalise.IntensityHistogramMatch(),
            "N4": normalise.IntensityN4Normalise(),
            "subtraction": normalise.NonRegMaskNormalise(),
            "none": None
        }
        try:
            norm_meths = [norm_dict[str(x)] for x in norm_methods]

        except KeyError as e:
            print(e)

            norm_meths = None
        num_jobs = 3
        logging.info(f"running with {num_jobs} jobs")

        # Execute the function in parallel using joblib
        def run_lama_radiomics(i):
            logging.info(f"running job {i}")
            radiomics_job_runner(target_dir, labs_of_int=labs_of_int, norm_method=normalise.NonRegMaskNormalise(),
                                 norm_label=norm_label, use_roi=use_roi,
                                 spherify=spherify, ref_vol_path=ref_vol_path, make_job_file=False)


        logging.info("Starting Radiomics")


        Parallel(n_jobs=-1)(delayed(run_lama_radiomics)(i) for i in range(num_jobs))


def test_permutation_stats_just_ovs():
    """
    Run the whole permutation based stats pipeline.
    Copy the output from a LAMA registrations test run, and increase or decrease the volume of the mutants so we get
    some hits

    """
    lama_permutation_stats.run(stats_cfg)



def test_radiomic_plotting():
    _dir = Path("V:/ent_cohort/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    abnormal_embs = ['22300_e8','22300_e6', '50_e5']

    for i, df in enumerate(data):
        df.index.name = 'org'
        df.name = str(file_names[i]).split(".")[0].split("/")[-1]
        df['genotype'] = 'HET' if 'het' in str(file_names[i]) else 'WT'
        df['background'] = 'C57BL6N' if (('b6ku' in str(file_names[i]))|('BL6' in str(file_names[i]))) else \
            'F1' if ('F1' in str(file_names[i])) else 'C3HHEH'

        df['HPE'] = 'abnormal' if any(map(str(file_names[i]).__contains__, abnormal_embs)) else 'normal'

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'org'])

    line_file = _dir / "full_results.csv"

    org_dir =_dir.parent / "organs"

    os.makedirs(org_dir, exist_ok=True)


    #for org in data.index.get_level_values('org').unique():
    #    data[data.index.get_level_values('org') == org].to_csv(str(org_dir)+"/results_" + str(org)+ ".csv")

    data.to_csv(line_file)

    data_subset = data.select_dtypes(include=np.number)

    data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, num_iters=20000, verbose=250)

    data['org'] = data.index.get_level_values('org')
    unique_values = np.unique(data['org'])
    print(unique_values)


    #print(data_subset.dropna(axis='columns'))

    results = embedding.fit_transform(data_subset.dropna(axis='columns'))




    color_class = data.index.get_level_values('org')

    # fig, ax = plt.subplots(figsize=[55, 60])
    # cluster.tsneplot(score=tsne_results, show=True, theme='dark', colorlist=color_class)

    data['PaCMAP-2d-one'] = results[:, 0]
    data['PaCMAP-2d-two'] = results[:, 1]
    data['org'] = data.index.get_level_values('org')
    data['specimen'] = data.index.get_level_values('specimen')
    data['condition'] = data['genotype'] + "_" + data['background']

    corr = data.corr(method='spearman')

    # Extract correlation values for 'PaCMAP-d2-one' column
    corr_values = corr.filter(like='PaCMAP', axis=1)
    corr_values = corr_values.filter(like='shape', axis=0)


    # Plot heatmap
    fig, ax = plt.subplots(figsize=[56, 60])

    sns.heatmap(corr_values, ax=ax, cmap='coolwarm', annot=True, cbar=True)
    plt.savefig("V:/ent_cohort/corr_heatmap.png")
    plt.close()


    # implement dbscan per organ


    # Create a dictionary to store the split arrays
    split_data = {}
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    # Split the NumPy array based on the unique values in 'split_column'
    for value in unique_values:
        print("value", value)
        split_data = data[data['org'] == value]

        dbscan.fit(split_data[['PaCMAP-2d-one','PaCMAP-2d-two']].values)

        # Retrieve the cluster labels
        labels = dbscan.labels_

        # Find the indices of points labeled as outliers (-1)
        outlier_indices = np.where(labels == -1)[0]

        # Access the outlier points from the coordinates
        outlier_points = split_data.iloc[outlier_indices]['specimen']

        print(outlier_points)


    fig, ax = plt.subplots(figsize=[56, 60])
    #data = data[data['condition'] == 'WT_C3HHEH']

    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=data,
        #col_order=['normal', 'abnormal'],
        col='condition',
        col_wrap=2,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two'])-10, np.max(data['PaCMAP-2d-two'])+10),
          xlim=(np.min(data['PaCMAP-2d-one'])-10, np.max(data['PaCMAP-2d-one'])+10))


    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_all_cond_v2.png")
    plt.close()

    fig, ax = plt.subplots(figsize=[56, 60])
    wt_c3h_data = data[data['condition'] == 'WT_C3HHEH']

    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=wt_c3h_data,
        # col_order=['normal', 'abnormal'],
        col='org',
        col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_C3H_wt_org_v2.png")
    plt.close()

    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=wt_c3h_data,
        # col_order=['normal', 'abnormal'],
        #col='specimen',
        #col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_C3H_map_v2.png")
    plt.close()

    het_c3h_data = data[data['condition'] == 'HET_C3HHEH']
    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=het_c3h_data,
        # col_order=['normal', 'abnormal'],
        col='specimen',
        col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_C3H_hets_v2.png")
    plt.close()

    wt_b6_data = data[data['condition'] == 'WT_C57BL6N']
    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=wt_b6_data,
        # col_order=['normal', 'abnormal'],
        #col='specimen',
        #col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_b6_map_v2.png")
    plt.close()

    wt_f1_data = data[data['condition'] == 'WT_C57BL6N']
    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=wt_f1_data,
        # col_order=['normal', 'abnormal'],
        # col='specimen',
        # col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_f1_map_v2.png")
    plt.close()

    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=wt_b6_data,
        # col_order=['normal', 'abnormal'],
        col='specimen',
        col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_b6_specs_v2.png")
    plt.close()

    het_b6_data = data[data['condition'] == 'HET_C57BL6N']
    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=het_b6_data,
        # col_order=['normal', 'abnormal'],
        col='specimen',
        col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_b6_hets_v2.png")
    plt.close()

    het_f1_data = data[data['condition'] == 'HET_F1']
    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=het_f1_data,
        # col_order=['normal', 'abnormal'],
        col='specimen',
        col_wrap=5,
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_f1_hets_v2.png")
    plt.close()

    fig, ax = plt.subplots(figsize=[56, 60])
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=data,
        col_order=['normal', 'abnormal'],
        col='HPE',
        row='condition',
        hue="org",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(data['PaCMAP-2d-two']) - 10, np.max(data['PaCMAP-2d-two']) + 10),
          xlim=(np.min(data['PaCMAP-2d-one']) - 10, np.max(data['PaCMAP-2d-one']) + 10))

    plt.savefig("V:/ent_cohort/radiomics_2D_PaCMAP_HPE_v2.png")
    plt.close()


def test_BQ_concat():
    _dir = Path("Z:/jcsmr/ROLab/Experimental data/Radiomics/Workflow design and trial results/Kyle Drov")
    #_dir = Path("E:/220913_BQ_tsphere/inputs/features/")

    # file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    # file_names.sort()
    #
    # data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]
    #
    # data = pd.concat(
    #     data,
    #     ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
    #     names=['specimen', 'label'])
    #
    # data['specimen'] = data.index.get_level_values('specimen')
    #
    # _metadata = data['specimen'].str.split('_', expand=True)
    #
    #
    #
    # _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
    #                                                                         'Cage_No.', 'Animal_No.']
    #
    #
    #
    #
    # _metadata.reset_index(inplace=True, drop=True)
    # data.reset_index(inplace=True, drop=True)
    # features = pd.concat([_metadata, data], axis=1)
    #
    # features.index.name = 'scanID'
    #
    # print(features)
    #
    # print(str(_dir.parent / "full_results.csv"))
    #
    # features.to_csv(str(_dir.parent / "full_results.csv"))

    features = pd.read_csv(_dir)
    features = features[features.columns.drop(list(features.filter(regex="diagnostics")))]
    features.drop(["scanID"], axis=1, inplace=True)
    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"))

def test_BQ_Pacmap():
    _data = pd.read_csv("E:/220204_BQ_dataset/scans_for_sphere_creation/full_cont_res/results_for_ml/full_results_smoted.csv")

    data_subset = _data.select_dtypes(include=np.number)

    data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    data_subset = data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, num_iters=20000, verbose=1)

    # print(data_subset.dropna(axis='columns'))

    results = embedding.fit_transform(data_subset.dropna(axis='columns'))

    #color_class = _data.index.get_level_values('Exp')

    # fig, ax = plt.subplots(figsize=[55, 60])
    # cluster.tsneplot(score=tsne_results, show=True, theme='dark', colorlist=color_class)

    _data['PaCMAP-2d-one'] = results[:, 0]
    _data['PaCMAP-2d-two'] = results[:, 1]

    fig, ax = plt.subplots(figsize=[56, 60])
    # data = data[data['condition'] == 'WT_C3HHEH']

    print(_data)
    g = sns.lmplot(
        x="PaCMAP-2d-one", y="PaCMAP-2d-two",
        data=_data,
        # col_order=['normal', 'abnormal'],
        #col='Exp',
        #col_wrap=2,
        hue="Tumour_Model",
        palette='husl',
        fit_reg=False)
    g.set(ylim=(np.min(_data['PaCMAP-2d-two']) - 1, np.max(_data['PaCMAP-2d-two']) + 1),
          xlim=(np.min(_data['PaCMAP-2d-one']) - 1, np.max(_data['PaCMAP-2d-one']) + 1))

    plt.savefig("E:/220204_BQ_dataset/scans_for_sphere_creation/full_cont_res/results_for_ml/radiomics_2D_PaCMAP_SMOTED.png")
    plt.close()




def test_BQ_mach_learn_non_tum():
    _dir = Path("E:/220919_non_tum/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'label'])

    data['specimen'] = data.index.get_level_values('specimen')

    _metadata = data['specimen'].str.split('_', expand=True)



    _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
                                                                            'Cage_No.', 'Animal_No.']




    _metadata.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    features = pd.concat([_metadata, data], axis=1)

    features.index.name = 'scanID'

    print(features)

    print(str(_dir.parent / "full_results.csv"))

    features.to_csv(str(_dir.parent / "full_results.csv"))

    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"))



def test_BQ_mach_learn_batch_sp():
    _dir = Path("E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/features/")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'label'])

    data['specimen'] = data.index.get_level_values('specimen')

    _metadata = data['specimen'].str.split('_', expand=True)



    _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
                                                                            'Cage_No.', 'Animal_No.']




    _metadata.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    features = pd.concat([_metadata, data], axis=1)

    features.index.name = 'scanID'

    print(features)

    print(str(_dir.parent / "full_results.csv"))

    features.to_csv(str(_dir.parent / "full_results.csv"))

    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"), batch_test=True)


def test_BQ_concat_batch():
    _dir = Path("Z:/jcsmr/ROLab/Experimental data/Radiomics/Workflow design and trial results/Kyle Drover analysis/220617_BQ_norm_stage_full/sub_normed_features.csv")
    #_dir = Path("E:/220913_BQ_tsphere/inputs/features/")

    # file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    # file_names.sort()
    #
    # data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]
    #
    # data = pd.concat(
    #     data,
    #     ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
    #     names=['specimen', 'label'])
    #
    # data['specimen'] = data.index.get_level_values('specimen')
    #
    # _metadata = data['specimen'].str.split('_', expand=True)
    #
    #
    #
    # _metadata.columns = ['Date', 'Exp', 'Contour_Method', 'Tumour_Model', 'Position', 'Age',
    #                                                                         'Cage_No.', 'Animal_No.']
    #
    #
    #
    #
    # _metadata.reset_index(inplace=True, drop=True)
    # data.reset_index(inplace=True, drop=True)
    # features = pd.concat([_metadata, data], axis=1)
    #
    # features.index.name = 'scanID'
    #
    # print(features)
    #
    # print(str(_dir.parent / "full_results.csv"))
    #
    # features.to_csv(str(_dir.parent / "full_results.csv"))

    features = pd.read_csv(_dir)
    features = features[features.columns.drop(list(features.filter(regex="diagnostics")))]
    features.drop(["scanID"], axis=1, inplace=True)
    feature_reduction.main(features, org = None, rad_file_path = Path(_dir.parent / "full_results.csv"), batch_test=True)


def test_non_tum_feat_norm():
    from lama.lama_radiomics.feature_reduction import non_tum_normalise
    tum = pd.read_csv("E:/220204_BQ_dataset/scans_for_sphere_creation/fold_normed_res/results_for_ml/full_results.csv", index_col=0)
    non_tum = pd.read_csv("E:/220204_BQ_dataset/scans_for_sphere_creation/sphere_non_tum_res/results_for_ml/full_results.csv", index_col=0)
    results = non_tum_normalise(tum, non_tum)
    results.to_csv("E:/220204_BQ_dataset/scans_for_sphere_creation/normed_results.csv")


def test_n_feat_plotting():

    _dir = Path("E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/test_size_0.2/None/")

    out_file = _dir / "full_cv_dataset.csv"
    cv_dataset = rad_plotting.n_feat_plotting(_dir)
    cv_dataset.to_csv(out_file)

def test_subsample_plotting():

    _dir = Path("E:/220204_BQ_dataset/scans_for_sphere_creation/full_cont_res/")

    out_file = _dir / "entire_train_test_part_dataset.csv"
    cv_dataset = rad_plotting.subsample_plotting(_dir)
    cv_dataset.to_csv(out_file)

def test_secondary_dataset_confusion_matrix():
    import catboost
    from sklearn.metrics import confusion_matrix
    from matplotlib.colors import ListedColormap

    model = catboost.CatBoostClassifier()

    model.load_model('E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/test_size_0.2/None/CPU_19_2.cbm')
    X = pd.read_csv("E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/results_for_ml/validation.csv")
    print(X)
    logging.info("Tumour Time!")
    X['Tumour_Model'] = X['Tumour_Model'].map({'4T1R': 0, 'CT26R': 1}).astype(int)
    X.set_index('Tumour_Model', inplace=True)
    X.drop(['Date', 'Animal_No.'], axis=1, inplace=True)
    X = X.loc[:, ~X.columns.str.contains('shape')]
    X.dropna(axis=1, inplace=True)


    X = X.select_dtypes(include=np.number)

    y_pred = model.predict(X)
    print(X.index)
    # Generate a confusion matrix
    conf_matrix = confusion_matrix(X.index, y_pred)

    cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Print row-wise percentages


    print(cm)
    colors = [['red', 'white'], ['white', 'blue']]
    conf_matrix_colors = np.empty(cm.shape, dtype='<U10')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            conf_matrix_colors[i][j] = colors[i][j]

    plt.imshow(cm, cmap='Blues', interpolation='None')
    plt.imshow(cm, interpolation='nearest', alpha=.3)


    # Add labels and ticks to the plot
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, ["Actual 4T1", "Actual CT26"])
    plt.yticks(tick_marks, ["Pred 4T1", "Pred CT26"])
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    # Add values to the plot
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j],
                     horizontalalignment="center",
                     color="white" if cm[i][j] > cm.max() / 2. else "black")

    # Save the plot as a PNG image
    plt.savefig('E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/test_size_0.2/None/confusion_matrix.png')





def test_find_shared_feats():

    inter_dataset = pd.read_csv("V:/230905_head_text_stuff/two_way/inter_organ_hit_dataset.csv", index_col=0)
    geno_dataset = pd.read_csv("V:/230905_head_text_stuff/two_way/geno_organ_hit_dataset.csv", index_col=0)
    treat_dataset = pd.read_csv("V:/230905_head_text_stuff/two_way/treat_organ_hit_dataset.csv", index_col=0)

    print(inter_dataset)
    outdir = Path("V:/230905_head_text_stuff/two_way/")

    results_dict = {}

    input_dict = {"inter":inter_dataset, "geno": geno_dataset, "treat": treat_dataset}


    for key, dset in input_dict.items():

        results = filt_for_shared_feats(inter_dataset, dset)

        results_dict[key] = pd.concat(list(results.values()), axis=1)

        for num_rows, df in results.items():

            df.clip(upper=2, lower=0, inplace=True)
            df = df.transpose()
            if not clustermap(df, title="Hooly", use_sns=True, rad_plot=True):
                logging.info(f'Skipping heatmap for {num_rows} as there are no results')

            plt.tight_layout()

            plt.savefig(outdir / f"{key}_rows{num_rows}_organ_hit_clustermap.png")
            plt.close()



    inters = results_dict.get("inter")



    genos = results_dict.get("geno")
    genos = genos.drop(['220422_BL6_Ku_50_e5_het', '210913_b6ku_22300_e6_het','210913_b6ku_22300_e8_het'], axis=0)


    treats = results_dict.get("treat")

    treats = treats.drop(['220422_BL6_Ku_50_e5_het', '210913_b6ku_22300_e6_het','210913_b6ku_22300_e8_het'], axis=0)



    #assert list(inters.columns) == list(genos.columns) == list(treats.columns), "Column names are not identical"

    assert not inters.index.isin(
        genos.index).any(), "Some index values are duplicated between 'inters' and 'genos' DataFrames."
    assert not inters.index.isin(
        treats.index).any(), "Some index values are duplicated between 'inters' and 'treats' DataFrames."
    assert not genos.index.isin(
        treats.index).any(), "Some index values are duplicated between 'genos' and 'treats' DataFrames."

    # Step 1: Find shared columns
    shared_columns = set(genos.columns).intersection(treats.columns).intersection(inters.columns)

    genos = genos[shared_columns]
    treats = treats[shared_columns]
    unique_feats_HPE = inters.loc[:, ~inters.columns.isin(shared_columns)].transpose()

    inters = inters[shared_columns]

    # Step 2: Perform row-wise merge
    full_dataset = pd.concat([inters, genos, treats], axis=0)

    full_dataset.to_csv(str(outdir/"full_datasets_heatmap.csv"))

    full_dataset.fillna(1, inplace=True)

    full_dataset.clip(upper=2, lower=0, inplace=True)
    full_dataset = full_dataset.transpose()

    unique_feats_HPE.clip(upper=2, lower=0, inplace=True)

    if not clustermap(unique_feats_HPE, title="Hooly", use_sns=True, rad_plot=True):
        logging.info('Skipping heatmap for as there are no results')

    plt.tight_layout()
    plt.savefig(outdir / "Unique_to_inter_organ_hit_clustermap_good.png")
    plt.close()


    if not clustermap(full_dataset, title="Hooly", use_sns=True, rad_plot=True, add_col_labels=True):
        logging.info('Skipping heatmap for as there are no results')

    plt.tight_layout()
    plt.savefig(outdir / "combined_organ_hit_clustermap_good.png")
    plt.close()











def test_pdist_fix():
    pdists = pd.read_csv("E:/fix_pdists/pdist_results.csv", index_col=0)
    pdists.reset_index(inplace=True)
    print(pdists.columns)

    pdists.drop(['index','Unnamed: 0.1'], axis=1, inplace=True)
    print(pdists)
    print(pdists.columns)

    pdists.to_csv("E:/fix_pdists/fixed_results.csv")









def test_feat_reduction():
    #features = pd.read_csv(, index_col=0)
    feature_reduction.main()

def test_mach_learn_pipeline():
    lama_machine_learning.ml_job_runner("E:/220204_BQ_dataset/230427_extras_for_validation/test_all_scans/results_for_ml")

def test_mach_learn_pipeline_v2():
    lama_machine_learning.ml_job_runner("V:/230612_g_by_back/radiomics_output/organs")

def test_mach_learn_pipeline_v3():
    lama_machine_learning.ml_job_runner("E:/220204_BQ_dataset/scans_for_sphere_creation/fold_normed_res/results_for_ml/")

def test_mach_learn_pipeline_w_non_tum_norm():
    non_tum_path = "E:/220204_BQ_dataset/scans_for_sphere_creation/sphere_non_tum_res/results_for_ml/full_results.csv"

    lama_machine_learning.ml_job_runner("E:/220204_BQ_dataset/scans_for_sphere_creation/sphere_15_res/results_for_ml/", non_tum_path = non_tum_path)


def test_radiomic_org_plotting():
    _dir = Path("V:/ent_cohort")

    file_names = [spec for spec in common.get_file_paths(folder=_dir, extension_tuple=".csv")]
    file_names.sort()

    data = [pd.read_csv(spec, index_col=0).dropna(axis=1) for spec in file_names]

    abnormal_embs = ['22300_e8', '22300_e6', '50_e5']

    for i, df in enumerate(data):
        df.index.name = 'org'
        df.name = str(file_names[i]).split(".")[0].split("/")[-1]
        df['genotype'] = 'HET' if 'het' in str(file_names[i]) else 'WT'
        df['background'] = 'C56BL6N' if (('b6ku' in str(file_names[i])) | ('BL6' in str(file_names[i]))) else 'C3HHEH'
        df['HPE'] = 'abnormal' if any(map(str(file_names[i]).__contains__, abnormal_embs)) else 'normal'

    data = pd.concat(
        data,
        ignore_index=False, keys=[os.path.splitext(os.path.basename(spec))[0] for spec in file_names],
        names=['specimen', 'org'])

    line_file = _dir.parent / "full_results.csv"

    data.to_csv(line_file)

    #data_subset = data.select_dtypes(include=np.number)

    for i, org in enumerate(data.index.levels[1]):
        fig, ax = plt.subplots(1, 1, figsize=[56, 60])
        #sns.set(font_scale=0.5)
        o_data = data[np.isin(data.index.get_level_values('org'), org)]

        o_data_subset = o_data.select_dtypes(include=np.number)
        #o_data_subset = o_data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        o_data_subset = o_data_subset.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        tsne = TSNE(perplexity=30,
                    n_components=2,
                    random_state=0,
                    early_exaggeration=250,
                    n_iter=1000,
                    verbose=1)

        tsne_results = tsne.fit_transform(o_data_subset.dropna(axis='columns'))

        o_data['tsne-2d-one'] = tsne_results[:, 0]
        o_data['tsne-2d-two'] = tsne_results[:, 1]
        o_data['org'] = o_data.index.get_level_values('org')
        o_data['specimen'] = o_data.index.get_level_values('specimen')

        o_data['condition'] = o_data['genotype'] + "_" + o_data['background']

        fig, ax = plt.subplots()
        o_data = o_data[o_data['condition'] == 'WT_C3HHEH']
        g = sns.lmplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=o_data,
            # col_order=['WT_C3HHEH','HET_C3HHEH','WT_C57BL6N','HET_C57BL6N'],
            #col='specimen',
            #col_wrap=5,
            hue="specimen",
            palette='husl',
            fit_reg=False)

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val']), fontsize='xx-small')

        label_point(o_data['tsne-2d-one'], o_data['tsne-2d-two'], o_data['specimen'], plt.gca())



        #g.set(ylim=(np.min(o_data['tsne-2d-two']) - 10, np.max(o_data['tsne-2d-two']) + 10),
         #     xlim=(np.min(o_data['tsne-2d-one']) - 10, np.max(o_data['tsne-2d-one']) + 10))
        plt.savefig("E:/220607_two_way/g_by_back_data/radiomics_output/radiomics_2D_tsne_C3H_wt_" + str(org) + ".png")
        plt.close()


def test_get_rad_data_for_perm():
    _dir = Path("Z:/jcsmr/ArkellLab/Lab Members/Kyle/PhD/220428_Hard_drive/221122_two_way/g_by_back_data/radiomics_output")

    wt_dir = Path("Z:/jcsmr/ArkellLab/Lab Members/Kyle/PhD/220428_Hard_drive/221122_two_way/g_by_back_data/baseline")

    mut_dir = Path("Z:/jcsmr/ArkellLab/Lab Members/Kyle/PhD/220428_Hard_drive/221122_two_way/g_by_back_data/mutants")

    treat_dir = Path("Z:/jcsmr/ArkellLab/Lab Members/Kyle/PhD/220428_Hard_drive/221122_two_way/g_by_back_data/treatment")

    inter_dir = Path("Z:/jcsmr/ArkellLab/Lab Members/Kyle/PhD/220428_Hard_drive/221122_two_way/g_by_back_data/mut_treat")

    results = get_radiomics_data(_dir, wt_dir, mut_dir, treat_dir, inter_dir)

    results.to_csv(str(_dir/"test_dataset.csv"))

def test_permutation_stats():
    """
    Run the whole permutation based stats pipeline.
    Copy the output from a LAMA registrations test run, and increase or decrease the volume of the mutants so we get
    some hits

    """
    lama_permutation_stats.run(stats_cfg)


def test_two_way_pene_plots():
    _dir = Path("V:/230905_head_text_stuff/")
    _label_dir = Path("V:/230612_target/target/E14_5_atlas_v24_43_label_info_v5.csv")
    heatmaps_for_permutation_stats(root_dir=_dir, two_way=True, label_info_file=_label_dir, rad_plot=True)


def test_sns_clustermap():
    url = "https://raw.githubusercontent.com/dorkylever/LAMA/master/lama/tests/clustermap_data.csv"
    X = pd.read_csv(url, index_col=0)
    X.dropna(how='all', inplace=True)
    X.fillna(1, inplace=True)
    # replace missing values with 0
    # replace infinite values with 0

    print("Missing values:", X.isnull().sum().sum())
    print("Infinite values:", np.isposinf(X).sum().sum() + np.isneginf(X).sum().sum())

    # mean_data = X.apply(np.mean, axis=1)

    # std_data = X.apply(np.std, axis=1)

    # constant_rows = X.apply(lambda row: row.nunique() == 1, axis=1)

    # X = X[~constant_rows]

    # na_mean = mean_data[mean_data.isna().any()]

    # na_std = std_data[std_data.iszero().any()]

    cg = sns.clustermap(X,
                        metric="euclidean",
                        cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512, center="light",
                                                   as_cmap=True),
                        cbar_kws={'label': 'mean volume ratio'}, square=True,
                        center=1,
                        figsize=[30, len(X) * 0.3])

    # Calculate the mean and standard deviation of each variable
    # X.columns = [col.rsplit('_', 3)[-1] for col in X.columns]

    plt.tight_layout()

    plt.savefig("E:/221122_two_way/permutation_stats/rad_perm_output/two_way/easy_fig.png")
    plt.close()

    X = X.to_numpy()

    print(np.isnan(X).any() | np.isinf(X).any())
    #
    print(X)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    #
    #
    print(np.all(sigma))
    #
    # # Calculate the z-score matrix
    Z = (X - mu) / sigma
    print(Z)
    #
    # # Calculate the Euclidean distance matrix
    d = np.zeros((Z.shape[0], Z.shape[0]))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[0]):
            d[i, j] = np.sqrt(np.sum((Z[i, :] - Z[j, :]) ** 2))
    #
    print(d)
    print(np.isnan(d).any() | np.isinf(d).any())
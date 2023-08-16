"""
Seaborn 0.9.0 gives missing rows for some reason. So use matplotlib directly instead
"""

import matplotlib.pyplot as plt
import pandas as pd
from logzero import logger as logging
import numpy as np
import seaborn as sns
from typing import Union
import matplotlib
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.stats import zscore
import sys

def heatmap(data: pd.DataFrame, title, use_sns=False, rad_plot: bool = False):
    fig, ax = plt.subplots(figsize=[56, 60])
    # use_sns = False

    font_size = 14 if rad_plot else 22

    if use_sns:
        # sns.palplot(sns.color_palette("coolwarm"))
        if data.isnull().values.all():
            return
        try:
            sns.heatmap(data, center=1.00, cmap=sns.color_palette("coolwarm", 100), ax=ax,
                        cbar_kws={'label': 'mean volume ratio',
                                  'fraction': 0.05}, square=(not rad_plot))
        except ValueError:
            ...

        ax.figure.axes[-1].yaxis.label.set_size(font_size)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
        # adjust cbar fraction

    else:
        ax.imshow(data)

    xlabels = data.columns
    ylabels = [x.replace('_', ' ') for x in data.index]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))

    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels, rotation=90, ha="center")
    ax.set_yticklabels(ylabels)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=90, ha='center',
    #          rotation_mode="default")
    # ax.xaxis.set_tick_params(horizontalalignment='left')
    # plt.xticks(np.arange(len(gamma_range)) + 0.5, gamma_range, rotation=45, )
    # plt.xticks(rotation=90)
    # ax.tick_params(axis="x", direction="right", pad=-22)
    # Note for radiomics data make this stuff smaller
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    return True


def clustermap(data: pd.DataFrame, title, use_sns=False, rad_plot: bool = False, z_norm: bool=False, Z=None):

    font_size = 5 if Z is not None else 10 if rad_plot else 22
    sys.setrecursionlimit(10000)
    # use_sns = False
    if use_sns:

        # sns.palplot(sns.color_palette("coolwarm"))
        if data.isnull().values.all():
            return
        try:
            figheight = len(data)*0.05 if Z is not None else len(data)*0.3
            figheight = figheight if figheight * 100 < 65536 else 655
            if rad_plot:
                def get_additional_info(idx, si, ei=None):
                    split_idx = idx.split(' ')
                    start_index = si+1 if 'wavelet' in idx else si
                    if ei is not None:
                        ei = ei + 1 if 'wavelet' in idx else ei
                    processed_string = ' '.join(split_idx[start_index:ei]) if ei else ' '.join(split_idx[start_index:])
                    return processed_string

                #l = 0.5, s = 0.8

                feature_info = pd.read_csv("V:/230612_target/target/feature_info.csv")
                org_list = feature_info.org_name.dropna().to_list()
                org_name = data.index.to_series().apply(get_additional_info, si=3).to_list()
                org_lut = dict(zip(org_list, sns.husl_palette(len(org_list))))

                org_lut_real = {key: value for key, value in org_lut.items() if key in org_name}

                org_row_colors = pd.DataFrame(org_name)[0].map(org_lut)



                filter_name = data.index.to_series().apply(get_additional_info, si=0, ei=1).to_list()
                #filt_lut = dict(zip(set(filter_name), sns.cubehelix_palette(len(set(filter_name)), dark=0, light=1, rot=0.7)))
                filt_list = feature_info.filter_name.dropna().to_list()

                filt_lut = dict(zip(filt_list, sns.color_palette(palette='spring', n_colors=len(filt_list))))

                filt_lut_real = {key: value for key, value in filt_lut.items() if key in filter_name}

                filt_row_colors = pd.DataFrame(filter_name)[0].map(filt_lut)

                print(filt_row_colors)

                type_name = data.index.to_series().apply(get_additional_info, si=1, ei=2).to_list()
                type_list = feature_info.type_name.dropna().to_list()
                type_lut = dict(zip(type_list, sns.color_palette(palette='viridis', n_colors=len(type_list))))
                type_lut_real = {key: value for key, value in type_lut.items() if key in type_name}
                type_row_colors = pd.DataFrame(type_name)[0].map(type_lut)

                print(type_row_colors)


                # row linkage precomputed
                if Z is not None:

                    # just making sure that there's no empty or infinite data somehow in the predcomputed distance matrix
                    assert not np.any(np.isnan(Z)) and not np.any(np.isinf(Z))

                    cg = sns.clustermap(data,
                                        metric="euclidean",
                                        row_linkage=Z,
                                        cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512, center="light",
                                                                   as_cmap=True),
                                        center=1,
                                        cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                        row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                        figsize=[30, figheight])
                else:

                    # So I have no fucking clue why this needs to be inverted when the raw values are fine and the clustmerap is fine
                    # man seaborn is fucking stupid


                    cg = sns.clustermap(data,
                                        metric="euclidean",
                                        cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512, center="light",
                                                                   as_cmap=True),
                                        center=1,
                                        cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                        row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                        figsize=[30, figheight])



                org_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                                 markerfacecolor=color) for label, color in org_lut_real.items()]

                type_legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=label,
                                                  markerfacecolor=color) for label, color in type_lut_real.items()]

                filt_legend_handles = [plt.Line2D([0], [0], marker='^', color='w', label=label,
                                                  markerfacecolor=color) for label, color in filt_lut_real.items()]

                all_legend_handles = org_legend_handles + type_legend_handles + filt_legend_handles
                all_labels = [h.get_label() for h in all_legend_handles]

                # Create a custom legend using the combined handles and labels
                plt.legend(all_legend_handles, all_labels, loc='upper left', prop={'size': 14})

                ylabels = [x.replace('_', ' ') for x in data.index]
                reordered_ind = cg.dendrogram_row.reordered_ind

                xlabels = [x.replace('_', ' ') for x in data.columns]

                reordered_col = cg.dendrogram_col.reordered_ind


                cg.ax_heatmap.tick_params(axis='y', labelsize=font_size)

                cg.ax_heatmap.set_xticks(np.arange(len(data.columns)) + 0.5)
                cg.ax_heatmap.set_xticklabels([data.columns[i] for i in reordered_col], rotation=90, ha='center')


                #ylabels = [x.replace('_', ' ') for x in data.index]
                cg.ax_heatmap.set_yticks(np.arange(len(ylabels))+0.5)

                cg.ax_heatmap.set_yticklabels([ylabels[i] for i in reordered_ind], fontsize=font_size, rotation=0, va='center')


            elif z_norm:
                cg = sns.clustermap(data,
                                    z_score=0,
                                    metric="euclidean",
                                    cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=40, n=512, center="light",
                                                               as_cmap=True),
                                    cbar_kws={'label': 'mean volume ratio'},
                                    square=True)

            else:
                cg = sns.clustermap(data,
                                    metric="euclidean",
                                    cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=40, n=512, center="light",
                                                               as_cmap=True),
                                    center=1,
                                    cbar_kws={'label': 'mean volume ratio'},
                                    square=True)


        except ValueError as e:
            print(e)

            ...

    sys.setrecursionlimit(1000)

    return True

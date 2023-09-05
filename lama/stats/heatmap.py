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
import matplotlib.patches as mpatches

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


def get_additional_info(idx, si, ei=None):
    split_idx = idx.split(' ')
    start_index = si + 1 if 'wavelet' in idx else si
    if ei is not None:
        ei = ei + 1 if 'wavelet' in idx else ei
    processed_string = ' '.join(split_idx[start_index:ei]) if ei else ' '.join(split_idx[start_index:])
    return processed_string

def clustermap(data: pd.DataFrame, title, use_sns=False, rad_plot: bool = False, z_norm: bool=False, Z=None, add_col_labels=None):

    print("add_col", add_col_labels)
    font_size = 5 if Z is not None else 10 if rad_plot else 22
    sys.setrecursionlimit(10000)
    # use_sns = False
    if use_sns:

        # sns.palplot(sns.color_palette("coolwarm"))
        if data.isnull().values.all():
            return
        try:
            #figheight = len(data)*0.05 if Z is not None else len(data)*0.3
            figheight=45
            figheight = figheight if figheight * 100 < 65536 else 655
            if rad_plot:
                #l = 0.5, s = 0.8
                #create row colours by organ, filter and feature type and create legends
                # feature names
                feature_info = pd.read_csv("V:/230612_target/target/feature_info_unfilt.csv")

                # get all organ names and create a consistent colour code
                org_list = feature_info.org_name.dropna().to_list()
                org_lut = dict(zip(org_list, sns.husl_palette(len(org_list))))

                # extract the organ name for each row in the dataset and map the corresponding colour
                org_name = data.index.to_series().apply(get_additional_info, si=3).to_list()

                org_row_colors = pd.DataFrame(org_name)[0].map(org_lut)

                # For the legend - only get the names exists (i.e. are statistically significant)
                org_lut_real = {key: value for key, value in org_lut.items() if key in org_name}


                # same thing but for filters and types
                filter_name = data.index.to_series().apply(get_additional_info, si=0, ei=1).to_list()
                #filt_lut = dict(zip(set(filter_name), sns.cubehelix_palette(len(set(filter_name)), dark=0, light=1, rot=0.7)))
                filt_list = feature_info.filter_name.dropna().to_list()

                filt_lut = dict(zip(filt_list, sns.color_palette(palette='spring', n_colors=len(filt_list))))

                filt_lut_real = {key: value for key, value in filt_lut.items() if key in filter_name}

                filt_row_colors = pd.DataFrame(filter_name)[0].map(filt_lut)


                type_name = data.index.to_series().apply(get_additional_info, si=1, ei=2).to_list()
                type_list = feature_info.type_name.dropna().to_list()
                type_lut = dict(zip(type_list, sns.color_palette(palette='viridis', n_colors=len(type_list))))
                type_lut_real = {key: value for key, value in type_lut.items() if key in type_name}
                type_row_colors = pd.DataFrame(type_name)[0].map(type_lut)

                # Now lets tidy up the rows:

                metadata = pd.read_csv("V:/230612_target/target/staging_info_volume.csv", index_col=0)

                merged_data = data.transpose().merge(metadata, left_index=True, right_on='vol')

                if add_col_labels:
                    print("yo dude")
                    genos = merged_data["Genotype"]

                    #genos = [row for row in genos if row.index in data.columns]
                    genos_lut = dict(zip(set(genos), sns.color_palette(palette='Set1', n_colors=len(set(genos)))))

                    geno_row_colors = genos.map(genos_lut).loc[data.columns].dropna()

                    backs = merged_data["Background"]
                    #backs = [row for row in backs if row.index in data.columns]
                    backs_lut = dict(zip(set(backs), sns.color_palette(palette='Set2', n_colors=len(set(backs)))))
                    print(backs.map(backs_lut))
                    print(backs.map(backs_lut).loc[data.columns])
                    back_row_colors = backs.map(backs_lut).loc[data.columns].dropna()
                    print(back_row_colors)

                    geno_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                                     markerfacecolor=color) for label, color in genos_lut.items()]

                    print(geno_legend_handles)


                    back_legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=label,
                                                      markerfacecolor=color) for label, color in backs_lut.items()]
                    print(back_legend_handles)


                # row linkage precomputed
                if Z is not None:

                    # just making sure that there's no empty or infinite data somehow in the predcomputed distance matrix
                    assert not np.any(np.isnan(Z)) and not np.any(np.isinf(Z))

                    if add_col_labels:
                        cg = sns.clustermap(data,
                                            metric="euclidean",
                                            row_linkage=Z,
                                            cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512,
                                                                       center="light",
                                                                       as_cmap=True),
                                            center=1,
                                            cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                            row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                            col_colors=[geno_row_colors, back_row_colors],
                                            figsize=[60, figheight])
                    else:
                        cg = sns.clustermap(data,
                                            metric="euclidean",
                                            row_linkage=Z,
                                            cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512,
                                                                       center="light",
                                                                       as_cmap=True),
                                            center=1,
                                            cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                            row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                            # col_colors=[geno_row_colors, back_row_colors],
                                            figsize=[45, figheight])

                else:

                    # So I have no fucking clue why this needs to be inverted when the raw values are fine and the clustmerap is fine
                    # man seaborn is fucking stupid

                    if add_col_labels:
                        cg = sns.clustermap(data,
                                            metric="euclidean",
                                            row_linkage=Z,
                                            cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512,
                                                                       center="light",
                                                                       as_cmap=True),
                                            center=1,
                                            cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                            col_colors=[geno_row_colors, back_row_colors],
                                            row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                            # col_colors=[geno_row_colors, back_row_colors],
                                            figsize=[60, figheight])
                    else:
                        cg = sns.clustermap(data,
                                            metric="euclidean",
                                            row_linkage=Z,
                                            cmap=sns.diverging_palette(250, 15, l=70, s=400, sep=1, n=512,
                                                                       center="light",
                                                                       as_cmap=True),
                                            center=1,
                                            cbar_kws={'label': 'mean ratio of radiological measurement'}, square=True,
                                            row_colors=[org_row_colors, type_row_colors, filt_row_colors],
                                            # col_colors=[geno_row_colors, back_row_colors],
                                            figsize=[45, figheight])

                # create an entry in the figure legend for organ, type and filter
                print("just test this shit")
                org_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                                 markerfacecolor=color) for label, color in org_lut_real.items()]

                type_legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=label,
                                                  markerfacecolor=color) for label, color in type_lut_real.items()]

                filt_legend_handles = [plt.Line2D([0], [0], marker='^', color='w', label=label,
                                                  markerfacecolor=color) for label, color in filt_lut_real.items()]

                org_legend_patch = mpatches.Patch(label='Organ:', color='None', linestyle='None')
                type_legend_patch = mpatches.Patch(label='Feature Category:', color='None', linestyle='None')
                filt_legend_patch = mpatches.Patch(label='Filter:', color='None', linestyle='None')



                if add_col_labels:
                    geno_legend_patch = mpatches.Patch(label='Genotype:', color='None', linestyle='None')
                    back_legend_patch = mpatches.Patch(label='Backgound:', color='None', linestyle='None')

                    all_legend_handles = [org_legend_patch] +org_legend_handles + \
                                         [type_legend_patch] + type_legend_handles +\
                                         [filt_legend_patch] + filt_legend_handles+\
                                         [geno_legend_patch] +geno_legend_handles+\
                                         [back_legend_patch] +back_legend_handles

                    all_labels = ["Organs:"]+[h.get_label() for h in org_legend_handles]+\
                                 ["Feature Category:"] + [h.get_label() for h in type_legend_handles]+\
                                 ["Filter:"]+ [h.get_label() for h in filt_legend_handles]+\
                                  ["Genotype:"]+ [h.get_label() for h in filt_legend_handles]+\
                                  ["Background:"]+ [h.get_label() for h in filt_legend_handles]



                else:
                # you can only do one legend - so add them together???
                    all_legend_handles = [org_legend_patch] + org_legend_handles + \
                                         [type_legend_patch] + type_legend_handles + \
                                         [filt_legend_patch] + filt_legend_handles

                all_labels = ["Organ:"] + [h.get_label() for h in org_legend_handles] + \
                             ["Feature Category:"] + [h.get_label() for h in type_legend_handles] + \
                             ["Filter:"] + [h.get_label() for h in filt_legend_handles]

                #all_labels = [h.get_label() for h in all_legend_handles]




                #all_legend_handles = [org_legend_patch, type_legend_patch, filt_legend_patch] + all_legend_handles
                #all_labels = ['Organ', 'Feature Category', 'Filter'] + all_labels

                # Create a custom legend using the combined handles and labels
                plt.legend(all_legend_handles, all_labels, loc='center', prop={'size': 18}, ncol=2)



                plt.subplots_adjust(right=1.5, top=1)

                ylabels = [x.replace('_', ' ') for x in data.index]
                reordered_ind = cg.dendrogram_row.reordered_ind

                # Fix up embryo IDs and add column colours

                merged_data = data.transpose().merge(metadata, left_index=True, right_on='vol')
                merged_data.set_index('Embryo_ID', inplace=True)
                merged_data = merged_data.transpose()


                #xlabels = [x.replace('_', ' ') for x in merged_data.columns]


                reordered_col = cg.dendrogram_col.reordered_ind

                print(cg.fig)
                cg.ax_heatmap.tick_params(axis='y', labelsize=font_size)

                cg.ax_heatmap.set_xticks(np.arange(len(data.columns)) + 0.5)

                cg.ax_heatmap.set_xticklabels([merged_data.columns[i] for i in reordered_col],fontsize=24, rotation=90, ha='center')

                #col = cg.ax_col_dendrogram.get_position()
                #cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height])
                #ylabels = [x.replace('_', ' ') for x in data.index]
                #cg.ax_heatmap.set_yticks(np.arange(len(ylabels))+0.5)

                #cg.ax_heatmap.set_yticklabels([ylabels[i] for i in reordered_ind], fontsize=font_size, rotation=0, va='center')



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

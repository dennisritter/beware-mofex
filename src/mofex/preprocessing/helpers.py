"""This module contains 3-D transformation function as well as geometric calculations."""
import math
import numpy as np
from mofex.preprocessing.filters import filter_outliers_iqr
import plotly.graph_objects as go


# Ignore outer 1%
# x = x[math.floor(0.01 * len(x)):math.floor(len(x) - 0.01 * len(x))]
# y = y[math.floor(0.01 * len(y)):math.floor(len(y) - 0.01 * len(y))]
# z = z[math.floor(0.01 * len(z)):math.floor(len(z) - 0.01 * len(z))]
# Z Score outlier filtering
# ZSCORE_ABS_THRESHOLD = 2.5
# x = x[abs(stats.zscore(x)) < 3]
# y = y[abs(stats.zscore(y)) < 3]
# z = z[abs(stats.zscore(z)) < 2.5]
# IQR outlier filtering
def xyz_minmax_coords(seqs, iqr_factors_xyz, plot_histogram=False):
    x = []
    y = []
    z = []
    for seq in seqs:
        x.extend(seq.positions[:, :, 0].flatten().tolist())
        y.extend(seq.positions[:, :, 1].flatten().tolist())
        z.extend(seq.positions[:, :, 2].flatten().tolist())

    # make array from list
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # sort
    x.sort()
    y.sort()
    z.sort()
    # Filter outliers
    x = filter_outliers_iqr(x, factor=iqr_factors_xyz[0])
    y = filter_outliers_iqr(y, factor=iqr_factors_xyz[1])
    z = filter_outliers_iqr(z, factor=iqr_factors_xyz[2])
    xmin = math.floor(x.min())
    xmax = math.ceil(x.max())
    ymin = math.floor(y.min())
    ymax = math.ceil(y.max())
    zmin = math.floor(z.min())
    zmax = math.ceil(z.max())

    if plot_histogram:
        xgroups = []
        ygroups = []
        zgroups = []
        min = np.array([xmin, ymin, zmin]).min()
        max = np.array([xmax, ymax, zmax]).max()
        for min_rng in range(min, max):
            xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
            ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
            zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
        xgroup_sizes = [len(group) for group in xgroups]
        ygroup_sizes = [len(group) for group in ygroups]
        zgroup_sizes = [len(group) for group in zgroups]

        group_labels = [f'[{g},{(g+1)}]' for g in range(min, max)]
        fig = go.Figure(data=[
            go.Bar(name='X Coords', x=group_labels, y=xgroup_sizes),
            go.Bar(name='Y Coords', x=group_labels, y=ygroup_sizes),
            go.Bar(name='Z Coords', x=group_labels, y=zgroup_sizes)
        ])
        fig.update_layout(barmode='group')
        fig.show()

    return [(xmin, xmax), (ymin, ymax), (zmin, zmax)]


def xyz_minmax_coords_per_bodypart(seqs, iqr_factors_xyz, plot_histogram=False):
    minmax_per_bp = np.zeros((len(seqs[0].positions[0]), 3, 2))
    # Just iterate over all bodyparts
    for bp_idx, bp in enumerate(seqs[0].positions[0]):
        x = []
        y = []
        z = []
        for seq in seqs:
            x.extend(seq.positions[:, bp_idx, 0].flatten().tolist())
            y.extend(seq.positions[:, bp_idx, 1].flatten().tolist())
            z.extend(seq.positions[:, bp_idx, 2].flatten().tolist())
        # make array from list
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        # Pelvis is always 0.0, 0.0, 0.0 --> Just use arbitrary range
        if np.all((x == 0)) and np.all((y == 0)) and np.all((z == 0)):
            xmin = -1
            xmax = 1
            ymin = -1
            ymax = 1
            zmin = -1
            zmax = 1
        else:
            # sort
            x.sort()
            y.sort()
            z.sort()
            # Filter outliers
            x = filter_outliers_iqr(x, factor=iqr_factors_xyz[0])
            y = filter_outliers_iqr(y, factor=iqr_factors_xyz[1])
            z = filter_outliers_iqr(z, factor=iqr_factors_xyz[2])
            xmin = math.floor(x.min())
            xmax = math.ceil(x.max())
            ymin = math.floor(y.min())
            ymax = math.ceil(y.max())
            zmin = math.floor(z.min())
            zmax = math.ceil(z.max())

        minmax_per_bp[bp_idx] = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])

        if plot_histogram:
            xgroups = []
            ygroups = []
            zgroups = []
            min = np.array([xmin, ymin, zmin]).min()
            max = np.array([xmax, ymax, zmax]).max()
            for min_rng in range(min, max):
                xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
                ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
                zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
            xgroup_sizes = [len(group) for group in xgroups]
            ygroup_sizes = [len(group) for group in ygroups]
            zgroup_sizes = [len(group) for group in zgroups]

            group_labels = [f'[{g},{(g+1)}]' for g in range(min, max)]
            fig = go.Figure(data=[
                go.Bar(name='X Coords', x=group_labels, y=xgroup_sizes),
                go.Bar(name='Y Coords', x=group_labels, y=ygroup_sizes),
                go.Bar(name='Z Coords', x=group_labels, y=zgroup_sizes)
            ])
            fig.update_layout(barmode='group')
            fig.show()

    return minmax_per_bp

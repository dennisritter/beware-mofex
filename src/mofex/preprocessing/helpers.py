"""This module contains 3-D transformation function as well as geometric calculations."""
import math
import numpy as np
from mofex.preprocessing.filters import filter_outliers_iqr
import plotly.graph_objects as go
import cv2


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
            # x = filter_outliers_iqr(x, factor=iqr_factors_xyz[0])
            # y = filter_outliers_iqr(y, factor=iqr_factors_xyz[1])
            # z = filter_outliers_iqr(z, factor=iqr_factors_xyz[2])
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


def to_motionimg_bp_minmax(
        seq,
        output_size=(256, 256),
        minmax_per_bp: np.ndarray = None,
        show_img=False,
) -> np.ndarray:
    """ Returns a Motion Image, that represents this sequences' positions.

            Creates an Image from 3-D position data of motion sequences.
            Rows represent a body part (or some arbitrary position instance).
            Columns represent a frame of the sequence.

            Args:
                output_size (int, int): The size of the output image in pixels
                    (height, width). Default=(200,200)
                minmax_per_bp (int, int): The minimum and maximum xyx-positions.
                    Mapped to color range (0, 255) for each body part separately.
        """
    # Create Image container
    img = np.zeros((len(seq.positions[0, :]), len(seq.positions), 3), dtype='uint8')
    # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
    # 2. Swap Axes of and frames(0) body parts(1) so rows represent body
    # parts and cols represent frames.
    for i, bp in enumerate(seq.positions[0]):
        bp_positions = seq.positions[:, i]
        x_colors = np.interp(bp_positions[:, 0], [minmax_per_bp[i, 0, 0], minmax_per_bp[i, 0, 1]], [0, 255])
        img[i, :, 0] = x_colors
        img[i, :, 1] = np.interp(bp_positions[:, 1], [minmax_per_bp[i, 1, 0], minmax_per_bp[i, 1, 1]], [0, 255])
        img[i, :, 2] = np.interp(bp_positions[:, 2], [minmax_per_bp[i, 2, 0], minmax_per_bp[i, 2, 1]], [0, 255])
    img = cv2.resize(img, output_size)

    if show_img:
        cv2.imshow(seq.name, img)
        print(f'Showing motion image from [{seq.name}]. Press any key to' ' close the image and continue.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img
import numpy as np
import mofex.feature_vectors as featvec
from scipy.signal import argrelextrema, savgol_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import mofex.preprocessing.normalizations as mofex_norm
import mana.utils.math.normalizations as normalizations
import cv2

# * Must work for all tracking formats. Add params or find better solution
# Indices constants for body parts that define normalized orientation of the skeleton
# center -> pelvis
CENTER_IDX = 0
# left -> hip_left
LEFT_IDX = 18
# right -> hip_right
RIGHT_IDX = 22
# up -> spinenavel
UP_IDX = 1

# Min/Max values used for the color mapping when transforming sequences to motion images
# min values are mapped to RGB(0,0,0), max values to RGB(255,255,255)
minmax_per_bp = np.array([[[-1.000e+00, 1.000e+00], [-1.000e+00, 1.000e+00], [-1.000e+00, 1.000e+00]],
                          [[-9.000e+00, 7.000e+00], [-1.400e+01, 1.600e+01], [1.740e+02, 1.970e+02]],
                          [[-1.400e+01, 1.200e+01], [-5.400e+01, 6.600e+01], [3.110e+02, 3.550e+02]],
                          [[-2.000e+01, 2.900e+01], [-2.130e+02, 2.580e+02], [5.040e+02, 5.940e+02]],
                          [[-5.300e+01, -1.100e+01], [-1.920e+02, 2.180e+02], [4.790e+02, 5.530e+02]],
                          [[-2.020e+02, -1.520e+02], [-2.060e+02, 1.920e+02], [4.300e+02, 5.850e+02]],
                          [[-3.770e+02, -1.540e+02], [-3.590e+02, 2.550e+02], [1.380e+02, 8.260e+02]],
                          [[-3.760e+02, -1.250e+02], [-3.880e+02, 5.550e+02], [-8.000e+01, 1.219e+03]],
                          [[-4.140e+02, -5.800e+01], [-4.730e+02, 6.750e+02], [-1.550e+02, 1.303e+03]],
                          [[-3.880e+02, -4.400e+01], [-5.110e+02, 5.540e+02], [-2.700e+02, 1.410e+03]],
                          [[-3.410e+02, -4.600e+01], [-4.170e+02, 6.120e+02], [-2.030e+02, 1.342e+03]],
                          [[1.500e+01, 6.200e+01], [-1.870e+02, 2.320e+02], [4.670e+02, 5.550e+02]],
                          [[1.400e+02, 1.960e+02], [-2.020e+02, 2.290e+02], [3.940e+02, 5.970e+02]],
                          [[9.900e+01, 3.760e+02], [-1.710e+02, 1.840e+02], [1.190e+02, 8.210e+02]],
                          [[-4.200e+01, 4.370e+02], [-1.030e+02, 3.510e+02], [-1.050e+02, 1.177e+03]],
                          [[-4.900e+01, 4.220e+02], [-2.030e+02, 4.260e+02], [-1.940e+02, 1.269e+03]],
                          [[-4.400e+01, 3.680e+02], [-1.560e+02, 4.550e+02], [-3.080e+02, 1.375e+03]],
                          [[-6.500e+01, 3.670e+02], [-1.640e+02, 5.040e+02], [-2.080e+02, 1.294e+03]],
                          [[-1.010e+02, -8.900e+01], [-9.000e+00, 8.000e+00], [-4.000e+00, 4.000e+00]],
                          [[-2.150e+02, -4.000e+01], [-1.080e+02, 4.050e+02], [-4.710e+02, -2.160e+02]],
                          [[-2.660e+02, -9.000e+00], [-2.310e+02, 3.740e+02], [-9.190e+02, -4.280e+02]],
                          [[-3.100e+02, -1.700e+01], [-1.060e+02, 5.520e+02], [-1.052e+03, -6.070e+02]],
                          [[8.000e+01, 9.100e+01], [-7.000e+00, 8.000e+00], [-4.000e+00, 4.000e+00]],
                          [[4.500e+01, 2.450e+02], [-1.020e+02, 4.220e+02], [-4.640e+02, -1.630e+02]],
                          [[3.700e+01, 2.600e+02], [-2.260e+02, 3.760e+02], [-9.090e+02, -4.140e+02]],
                          [[7.100e+01, 2.940e+02], [-1.090e+02, 5.350e+02], [-1.042e+03, -5.820e+02]],
                          [[-2.500e+01, 3.600e+01], [-2.550e+02, 3.400e+02], [5.690e+02, 6.900e+02]],
                          [[-2.900e+01, 5.300e+01], [-1.250e+02, 4.870e+02], [4.480e+02, 8.450e+02]],
                          [[-5.300e+01, 1.900e+01], [-1.730e+02, 5.080e+02], [5.130e+02, 8.550e+02]],
                          [[-1.130e+02, -5.100e+01], [-2.760e+02, 4.100e+02], [6.150e+02, 7.610e+02]],
                          [[1.000e+00, 7.400e+01], [-1.680e+02, 5.050e+02], [5.090e+02, 8.530e+02]],
                          [[5.200e+01, 1.300e+02], [-2.750e+02, 4.090e+02], [6.110e+02, 7.820e+02]]])


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


def _normalize_seq(seq):
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])
    return seq


class RepCounter:
    """Counts Repititions of motions from 3-D MoCap Sequences"""
    def __init__(self, seq_gt, subseq_len, savgol_win, model, feature_size, preprocess):
        self.model = model
        self.feature_size = feature_size
        self.preprocess = preprocess

        self.seq_gt = _normalize_seq(seq_gt)
        self.motion_image_gt = to_motionimg_bp_minmax(seq_gt, output_size=(256, 256), minmax_per_bp=minmax_per_bp)
        self.featvec_gt = featvec.load_from_motion_imgs(motion_images=[self.motion_image_gt],
                                                        model=self.model,
                                                        feature_size=self.feature_size,
                                                        preprocess=self.preprocess)[0]
        self.seq_q_original = None
        self.seqs_q_normalized = []
        self.motion_images_q = []
        self.featvecs_q = []
        self.subseq_len = subseq_len
        self.savgol_win = savgol_win
        self.distances = []
        self.keyframes = []
        self.history = []

    def append_seq_q(self, seq):
        # Add new seq to original sequence
        if not self.seq_q_original:
            self.seq_q_original = seq
        else:
            self.seq_q_original.append(seq)

        # Postprocessing for unprocessed sequence frames
        start = len(self.seqs_q_normalized) * self.subseq_len
        seq_split_original = self.seq_q_original[start:].split(overlap=0, subseq_size=self.subseq_len)
        seq_split_normalized = [_normalize_seq(seq) for seq in seq_split_original]
        self.seqs_q_normalized += seq_split_normalized
        mi_split = [to_motionimg_bp_minmax(seq, output_size=(256, 256), minmax_per_bp=minmax_per_bp) for seq in seq_split_normalized]
        self.motion_images_q += seq_split_normalized
        featvec_split = featvec.load_from_motion_imgs(motion_images=mi_split, model=self.model, feature_size=self.feature_size, preprocess=self.preprocess)
        self.featvecs_q += featvec_split
        self.distances += [np.linalg.norm(self.featvec_gt - featvec_q) for featvec_q in featvec_split]

        self.savgol_distances = savgol_filter(self.distances, self.savgol_win, 7, mode='nearest')
        self.savgol_distance_minima = argrelextrema(self.savgol_distances, np.less_equal, order=5)[0]

        # Keyframe indices per frame
        self.keyframes = self.savgol_distance_minima * self.subseq_len

        min_dists = [self.savgol_distances[idx] for idx in self.savgol_distance_minima]
        self.history.append({
            "distances": self.distances[:],
            "savgol_distances": self.savgol_distances[:],
            "savgol_distance_minima": self.savgol_distance_minima[:],
            "min_dists": min_dists[:]
        })

    def show(self):
        fig = make_subplots(rows=3, cols=1)
        fig.append_trace(go.Scatter(x=np.arange(len(self.distances)), y=self.distances, name=f'Distances'), row=1, col=1)
        fig.append_trace(go.Scatter(x=np.arange(len(self.savgol_distances)), y=self.savgol_distances, name=f'Savgol Distances'), row=2, col=1)

        # Plot Minima
        min_dists = [self.savgol_distances[idx] for idx in self.savgol_distance_minima]
        fig.append_trace(
            go.Scatter(x=self.savgol_distance_minima, y=min_dists, name=f'key segments', mode='markers', marker_color='red'),
            row=2,
            col=1,
        )
        fig.update_layout(height=1000, width=1000, title_text="Repcounter Results")
        fig.show()

    def show_animated(self):
        frames = []
        for snapshot in self.history:
            frame = go.Frame(data=[
                go.Scatter(x=np.arange(len(snapshot["savgol_distances"])), y=snapshot["savgol_distances"]),
                go.Scatter(x=snapshot["savgol_distance_minima"], y=snapshot["min_dists"])
            ])
            frames.append(frame)

        fig = go.Figure(
            data=[
                go.Scatter(x=np.arange(len(self.history[0]["savgol_distances"])), y=self.history[0]["savgol_distances"]),
                go.Scatter(x=self.history[0]["savgol_distance_minima"], y=self.history[0]["min_dists"], mode="markers"),
                go.Scatter(x=np.arange(len(self.distances)), y=np.zeros(len(self.distances)), mode="markers")
            ],
            layout=go.Layout(
                xaxis=dict(range=[0, len(self.savgol_distances)], autorange=False),
                yaxis=dict(range=[0, max(self.savgol_distances)], autorange=False),
                title="Animated RepCounter Results",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[{
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": (1000 / 30) * 10,  # (1s / fps) * batchsize
                                        "redraw": False
                                    },
                                    "fromcurrent": True,
                                    "transition": {
                                        "easing": "quadratic-in-out"
                                    }
                                }
                            ],
                            "label":
                            "Play",
                            "method":
                            "animate"
                        }])
                ]),
            frames=frames)

        fig.show()

import numpy as np
from scipy.signal import argrelextrema, savgol_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import mana.utils.math.normalizations as normalizations
from torchvision import transforms
import mofex.preprocessing.normalizations as mofex_norm
from mofex.preprocessing.helpers import to_motionimg_bp_minmax
import mofex.feature_vectors as featvec
import mofex.model_loader as model_loader

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
                          [[-3.100e+01, 3.300e+01], [-4.000e+01, 1.150e+02], [1.340e+02, 2.100e+02]],
                          [[-4.500e+01, 4.300e+01], [-8.500e+01, 2.300e+02], [2.150e+02, 3.770e+02]],
                          [[-7.500e+01, 6.800e+01], [-2.130e+02, 3.690e+02], [3.760e+02, 6.190e+02]],
                          [[-9.800e+01, 2.700e+01], [-1.920e+02, 3.420e+02], [3.540e+02, 5.790e+02]],
                          [[-2.270e+02, -1.060e+02], [-2.580e+02, 3.790e+02], [3.450e+02, 6.520e+02]],
                          [[-5.290e+02, -1.070e+02], [-3.630e+02, 6.300e+02], [1.380e+02, 9.610e+02]],
                          [[-7.680e+02, 0.000e+00], [-3.880e+02, 8.580e+02], [-8.000e+01, 1.219e+03]],
                          [[-8.500e+02, 2.600e+01], [-4.730e+02, 9.500e+02], [-1.550e+02, 1.303e+03]],
                          [[-9.650e+02, 8.400e+01], [-5.110e+02, 1.049e+03], [-2.700e+02, 1.410e+03]],
                          [[-8.720e+02, 7.800e+01], [-4.170e+02, 9.790e+02], [-2.030e+02, 1.342e+03]],
                          [[-3.500e+01, 9.200e+01], [-1.870e+02, 3.470e+02], [3.470e+02, 5.780e+02]],
                          [[4.700e+01, 2.250e+02], [-2.310e+02, 4.140e+02], [3.310e+02, 6.100e+02]],
                          [[-1.530e+02, 5.400e+02], [-3.220e+02, 6.720e+02], [1.190e+02, 9.120e+02]],
                          [[-3.100e+02, 7.300e+02], [-4.060e+02, 8.740e+02], [-1.050e+02, 1.177e+03]],
                          [[-2.330e+02, 8.180e+02], [-4.950e+02, 9.770e+02], [-1.940e+02, 1.269e+03]],
                          [[-2.110e+02, 9.150e+02], [-5.190e+02, 1.042e+03], [-3.080e+02, 1.375e+03]],
                          [[-2.240e+02, 8.380e+02], [-3.960e+02, 9.870e+02], [-2.080e+02, 1.294e+03]],
                          [[-1.070e+02, -8.000e+01], [-4.800e+01, 4.100e+01], [-1.400e+01, 1.200e+01]],
                          [[-3.480e+02, 1.850e+02], [-1.080e+02, 4.050e+02], [-4.710e+02, 5.000e+01]],
                          [[-3.000e+02, 1.440e+02], [-2.310e+02, 7.080e+02], [-9.190e+02, -2.760e+02]],
                          [[-3.530e+02, 1.560e+02], [-1.060e+02, 8.500e+02], [-1.052e+03, -3.130e+02]],
                          [[7.200e+01, 9.700e+01], [-3.700e+01, 4.300e+01], [-1.100e+01, 1.200e+01]],
                          [[-7.400e+01, 2.670e+02], [-1.020e+02, 4.220e+02], [-4.640e+02, 5.800e+01]],
                          [[-1.590e+02, 5.190e+02], [-3.710e+02, 7.270e+02], [-9.090e+02, -2.670e+02]],
                          [[-2.240e+02, 6.050e+02], [-3.020e+02, 8.620e+02], [-1.042e+03, -1.800e+02]],
                          [[-9.000e+01, 8.600e+01], [-2.550e+02, 4.310e+02], [4.270e+02, 7.140e+02]],
                          [[-1.790e+02, 1.790e+02], [-1.250e+02, 5.860e+02], [3.490e+02, 8.450e+02]],
                          [[-1.670e+02, 1.370e+02], [-1.730e+02, 5.670e+02], [3.970e+02, 8.550e+02]],
                          [[-1.690e+02, 1.100e+01], [-2.760e+02, 4.730e+02], [4.760e+02, 8.030e+02]],
                          [[-1.460e+02, 1.740e+02], [-1.680e+02, 5.590e+02], [3.900e+02, 8.530e+02]],
                          [[-1.300e+01, 1.600e+02], [-2.750e+02, 4.550e+02], [4.720e+02, 8.050e+02]]])


def _normalize_seq(seq):
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])
    return seq


class RepCounter:
    """Counts Repititions of motions from 3-D MoCap Sequences"""
    def __init__(self, seq_gt, subseq_len):
        # The Model that creates features from motion images
        self.model = model_loader.load_trained_model(model_name='resnet101_mka-beware-1.1',
                                                     remove_last_layer=True,
                                                     state_dict_path='./data/trained_models/mka-beware-1.1/resnet101_mka-beware-1.1_e5.pt')
        # Size of the features created by the model
        self.feature_size = 2048
        # Transforms that are applied to motion images before send to model
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.savgol_win = 31
        self.savgol_order = 7

        # The Ground Truth sequence that defines the start/end of the motion to count
        self.seq_gt = _normalize_seq(seq_gt)
        # The motion image of the Ground Truth sequence
        self.motion_image_gt = to_motionimg_bp_minmax(seq_gt, output_size=(256, 256), minmax_per_bp=minmax_per_bp)
        # The Ground Truths' feature vector determined from its motion image
        self.featvec_gt = featvec.load_from_motion_imgs(motion_images=[self.motion_image_gt],
                                                        model=self.model,
                                                        feature_size=self.feature_size,
                                                        preprocess=self.preprocess)[0]

        self.seq_q_original = None
        self.seqs_q_normalized = []
        self.motion_images_q = []
        self.featvecs_q = []
        self.subseq_len = subseq_len
        self.distances = []
        self.keyframes = []
        self.history = []

    def append_seq_q(self, seq):
        # Append the given sequence to the unnormalized query sequence
        if not self.seq_q_original:
            self.seq_q_original = seq
        else:
            self.seq_q_original.append(seq)

        ## Processing of new positions
        # Determine the frame in the unnormalized query sequence from where no postprocessing has been done yet
        unprocessed_start = len(self.seqs_q_normalized) * self.subseq_len
        # Check whether enough new frames have been appended for further processing
        if len(self.seq_q_original[unprocessed_start:]) >= self.subseq_len:
            # Split unprocessed frames into several smaller sequences
            seq_split_original = self.seq_q_original[unprocessed_start:].split(overlap=0, subseq_size=self.subseq_len)
            # Normalize the split sequences
            seq_split_normalized = [_normalize_seq(seq) for seq in seq_split_original]
            self.seqs_q_normalized += seq_split_normalized
            # Create motion images from the new split sequences
            mi_split = [to_motionimg_bp_minmax(seq, output_size=(256, 256), minmax_per_bp=minmax_per_bp) for seq in seq_split_normalized]
            self.motion_images_q += seq_split_normalized
            # Create feature vectors from the new motion images
            featvec_split = featvec.load_from_motion_imgs(motion_images=mi_split, model=self.model, feature_size=self.feature_size, preprocess=self.preprocess)
            self.featvecs_q += featvec_split
            # Determine the distances of the ground truth sequence and each of the new split feature vectors
            self.distances += [np.linalg.norm(self.featvec_gt - featvec_q) for featvec_q in featvec_split]

        ## Counting
        # Smoothen the list of all distances
        self.savgol_distances = savgol_filter(self.distances, self.savgol_win, self.savgol_order, mode='nearest')
        # Find the chunks that are most similar to the ground truth sequence
        # by identifying local minima in the smoothened distances
        self.savgol_distance_minima = argrelextrema(self.savgol_distances, np.less_equal, order=5)[0]
        # Determine the start/end keyframe positions
        self.keyframes = self.savgol_distance_minima * self.subseq_len

        # Add some data to a history
        self.history.append({
            "distances": self.distances[:],
            "savgol_distances": self.savgol_distances[:],
            "savgol_distance_minima": self.savgol_distance_minima[:],
            "min_dists": [self.savgol_distances[idx] for idx in self.savgol_distance_minima]
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
                go.Scatter(x=np.arange(len(snapshot["savgol_distances"])) * self.subseq_len, y=snapshot["savgol_distances"]),
                go.Scatter(
                    x=snapshot["savgol_distance_minima"] * self.subseq_len,
                    y=snapshot["min_dists"],
                    text=snapshot["savgol_distance_minima"] * self.subseq_len,
                    textposition="bottom center",
                    mode="markers+text",
                    marker=dict(size=20),
                )
            ])
            frames.append(frame)

        fig = go.Figure(
            data=[
                go.Scatter(x=[0], y=[0], name="Smoothed Distances"),
                go.Scatter(x=[0], y=[0], name="Start/End Keyframes"),
            ],
            layout=go.Layout(
                xaxis=dict(range=[0, len(self.savgol_distances) * self.subseq_len], autorange=False),
                yaxis=dict(range=[min(self.savgol_distances) - 1, max(self.savgol_distances)], autorange=False),
                title="Animated RepCounter Results",
                xaxis_title='Frame',
                yaxis_title='Distances',
                font=dict(size=18),
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

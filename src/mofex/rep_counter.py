import numpy as np
import mofex.feature_vectors as featvec
from scipy.signal import argrelextrema, savgol_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# * Must work for all tracking formats. Add params or find better solution
# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

# Min/Max values used for the color mapping when transforming sequences to motion images
# min values are mapped to RGB(0,0,0), max values to RGB(255,255,255)
xmin, xmax = (-14.772495736531305, 14.602030756418097)
ymin, ymax = (-14.734704969722216, 14.557769829141042)
zmin, zmax = (-19.615324010444805, 19.43983405425556)


def _normalize_seq(seq):
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    return seq


class RepCounter:
    """Counts Repititions of motions from 3-D MoCap Sequences"""
    def __init__(self, seq_gt, subseq_len, savgol_win, model, feature_size, preprocess):
        self.model = model
        self.feature_size = feature_size
        self.preprocess = preprocess

        self.seq_gt = _normalize_seq(seq_gt)
        self.motion_image_gt = seq_gt.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
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
        mi_split = [
            seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
            for seq in seq_split_normalized
        ]
        self.motion_images_q += seq_split_normalized
        featvec_split = featvec.load_from_motion_imgs(motion_images=mi_split, model=self.model, feature_size=self.feature_size, preprocess=self.preprocess)
        self.featvecs_q += featvec_split
        self.distances += [np.linalg.norm(self.featvec_gt - featvec_q) for featvec_q in featvec_split]

        self.savgol_distances = savgol_filter(self.distances, self.savgol_win, 3, mode='nearest')
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

        fig = go.Figure(data=[
            go.Scatter(x=np.arange(len(self.history[0]["savgol_distances"])), y=self.history[0]["savgol_distances"]),
            go.Scatter(x=self.history[0]["savgol_distance_minima"], y=self.history[0]["min_dists"], mode="markers"),
            go.Scatter(x=np.arange(len(self.distances)), y=np.zeros(len(self.distances)), mode="markers")
        ],
                        layout=go.Layout(xaxis=dict(range=[0, len(self.savgol_distances)], autorange=False),
                                         yaxis=dict(range=[0, max(self.savgol_distances)], autorange=False),
                                         title="Animated RepCounter Results",
                                         updatemenus=[
                                             dict(type="buttons",
                                                  buttons=[{
                                                      "args": [
                                                          None, {
                                                              "frame": {
                                                                  "duration": 100,
                                                                  "redraw": False
                                                              },
                                                              "fromcurrent": True,
                                                              "transition": {
                                                                  "duration": 30,
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

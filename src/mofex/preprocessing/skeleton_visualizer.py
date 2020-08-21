"""Visualises a skeleton from a Sequence in a webbrowser 3-D canvas."""
import numpy as np
import plotly.graph_objects as go
import chart_studio.plotly as py
import mofex.preprocessing.transformations as transformations


class SkeletonVisualizer:
    """Visualises a human pose skeleton as an animated 3D Scatter Plot.

    Attributes:
        sequence (Sequence): The motion sequence to visualise the skeleton from.
    """
    def __init__(
        self,
        sequence,
    ):

        self.sequence = sequence[:]

    def show(self, auto_open=True):
        """Visualises the human pose skeleton as an animated 3D Scatter Plot."""
        traces = self._get_traces(0)
        layout = self._get_layout()
        frames = self._get_frames()

        print("Generating Skeleton Plot...")
        fig = go.Figure(data=traces, layout=layout, frames=frames)
        fig.write_html('skeleton.html', auto_open=auto_open, auto_play=False)
        # print(f"Plot URL: {py.plot(fig, filename='skeleton', auto_open=False)}")
        # fig.show()

    def _get_layout(self):
        """Returns a Plotly layout."""
        updatemenus = []
        sliders = []
        if len(self.sequence) > 1:
            updatemenus = self._make_buttons()
            sliders = self._make_sliders()

        scene = dict(
            xaxis=dict(range=[-1000, 1000], ),
            yaxis=dict(range=[-1000, 1000], ),
            zaxis=dict(range=[-1000, 1000], ),
            camera=dict(up=dict(x=0, y=0, z=1.25), eye=dict(x=-1.2, y=-1.2, z=1.2)),
        )

        layout = go.Layout(
            scene=scene,
            scene_aspectmode="cube",
            updatemenus=updatemenus,
            sliders=sliders,
            showlegend=False,
        )
        return layout

    def _make_sliders(self):
        """Returns a list including one Plotly slider that allows users to
        controll the displayed frame."""
        pos = self.sequence.positions
        # Frame Slider
        slider = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {
                    "size": 20
                },
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {
                "duration": 0,
                "easing": "linear"
            },
            "pad": {
                "b": 10,
                "t": 50
            },
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }
        for i in range(0, len(pos)):
            # Create slider step for each frame
            slider_step = {
                "args": [[i], {
                    "frame": {
                        "duration": 0,
                        "redraw": True
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0
                    }
                }],
                "label": i,
                "method": "animate"
            }
            slider["steps"].append(slider_step)

        return [slider]

    def _make_buttons(self):
        """Returns a list of Plotly buttons to start and stop the animation."""
        # Play / Pause Buttons
        buttons = [{
            "buttons": [{
                "label": "Play",
                "args": [None, {
                    "frame": {
                        "duration": 0,
                        "redraw": True
                    },
                    "fromcurrent": True,
                    "transition": {
                        "duration": 0,
                        "easing": "linear"
                    }
                }],
                "method": "animate"
            }, {
                "label": "Pause",
                "args": [[None], {
                    "frame": {
                        "duration": 0,
                        "redraw": False
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0
                    }
                }],
                "method": "animate"
            }],
            "direction": "left",
            "pad": {
                "r": 10,
                "t": 87
            },
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]  # yapf: disable
        return buttons

    def _get_frames(self):
        """Returns a list of frames.

        Each frame represents a single scatter plot showing the
        skeleton.
        """
        # No animation frames needed when visualising only one frame
        if len(self.sequence) <= 1:
            return []

        pos = self.sequence.positions
        frames = []
        for i in range(0, len(pos)):
            frame = {"data": self._get_traces(i), "name": i}
            frames.append(frame)
        return frames

    def _make_joint_traces(self, frame):
        pos = self.sequence.positions
        trace_joints = go.Scatter3d(x=pos[frame, :, 0],
                                    y=pos[frame, :, 1],
                                    z=pos[frame, :, 2],
                                    text=np.arange(len(pos[frame])),
                                    textposition='top center',
                                    mode="markers+text",
                                    marker=dict(color="royalblue", size=5))
        # root_joints = go.Scatter3d(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]), mode="markers", marker=dict(color="red", size=5))
        # hipl = go.Scatter3d(x=np.array(pos[frame, 1, 0]),
        #                     y=np.array(pos[frame, 1, 1]),
        #                     z=np.array(pos[frame, 1, 2]),
        #                     mode="markers",
        #                     marker=dict(color="red", size=10))
        # hipr = go.Scatter3d(x=np.array(pos[frame, 6, 0]),
        #                     y=np.array(pos[frame, 6, 1]),
        #                     z=np.array(pos[frame, 6, 2]),
        #                     mode="markers",
        #                     marker=dict(color="green", size=10))
        # lowerback = go.Scatter3d(x=np.array(pos[frame, 11, 0]),
        #                      y=np.array(pos[frame, 11, 1]),
        #                      z=np.array(pos[frame, 11, 2]),
        #                      mode="markers",
        #                      marker=dict(color="black", size=10))
        return [trace_joints]  # + [hipl, hipr, lowerback]

    # def _make_limb_traces(self, frame):
    #     pos = self.sequence.positions
    #     bps = self.sequence.body_parts
    #     # Each element represents a pair of body part indices in sequence.positions that will be connected with a line
    #     limb_connections = [
    #         [bps["head"], bps["neck"]],
    #         [bps["neck"], bps["shoulder_l"]],
    #         [bps["neck"], bps["shoulder_r"]],
    #         [bps["shoulder_l"], bps["elbow_l"]],
    #         [bps["shoulder_r"], bps["elbow_r"]],
    #         [bps["elbow_l"], bps["wrist_l"]],
    #         [bps["elbow_r"], bps["wrist_r"]],
    #         [bps["neck"], bps["torso"]],
    #         [bps["torso"], bps["pelvis"]],
    #         [bps["pelvis"], bps["hip_l"]],
    #         [bps["pelvis"], bps["hip_r"]],
    #         [bps["hip_l"], bps["knee_l"]],
    #         [bps["hip_r"], bps["knee_r"]],
    #         [bps["knee_l"], bps["ankle_l"]],
    #         [bps["knee_r"], bps["ankle_r"]]
    #     ]  # yapf: disable
    #     limb_traces = []
    #     for limb in limb_connections:
    #         limb_trace = go.Scatter3d(x=[pos[frame, limb[0], 0], pos[frame, limb[1], 0]],
    #                                   y=[pos[frame, limb[0], 1], pos[frame, limb[1], 1]],
    #                                   z=[pos[frame, limb[0], 2], pos[frame, limb[1], 2]],
    #                                   mode="lines",
    #                                   line=dict(color="firebrick", width=5))
    #         limb_traces.append(limb_trace)
    #     return limb_traces

    # def _make_lcs_trace(self, origin, x_direction, y_direction, z_direction):
    #     """Returns a list that contains a plotly trace object the X, Y and Z
    #     axes of the local joint coordinate system calculated from an origin, a
    #     X-axis-direction and a Y-axis-direction."""

    #     # Set Local Coordinate System vectors' length to 100 and move relative to local origin.
    #     x_direction = x_direction * 100 + origin
    #     y_direction = y_direction * 100 + origin
    #     z_direction = z_direction * 100 + origin
    #     trace_x = go.Scatter3d(x=[origin[0], x_direction[0]],
    #                            y=[origin[1], x_direction[1]],
    #                            z=[origin[2], x_direction[2]],
    #                            mode="lines",
    #                            marker=dict(color="red"))
    #     trace_y = go.Scatter3d(x=[origin[0], y_direction[0]],
    #                            y=[origin[1], y_direction[1]],
    #                            z=[origin[2], y_direction[2]],
    #                            mode="lines",
    #                            marker=dict(color="green"))
    #     trace_z = go.Scatter3d(x=[origin[0], z_direction[0]],
    #                            y=[origin[1], z_direction[1]],
    #                            z=[origin[2], z_direction[2]],
    #                            mode="lines",
    #                            marker=dict(color="blue"))
    #     return [trace_x, trace_y, trace_z]

    # def _make_pelvis_cs_trace(self, frame):
    #     # TODO: REMOVE, it is already included in _make_jcs_traces
    #     bps = self.sequence.body_parts
    #     pcs = transformations.get_pelvis_coordinate_system(self.sequence.positions[frame][bps["pelvis"]], self.sequence.positions[frame][bps["torso"]],
    #                                                        self.sequence.positions[frame][bps["hip_l"]], self.sequence.positions[frame][bps["hip_r"]])
    #     p_origin = pcs[0][0]
    #     pcs[0][1][0] = pcs[0][1][0] * 100 + p_origin
    #     pcs[0][1][1] = pcs[0][1][1] * 100 + p_origin
    #     pcs[0][1][2] = pcs[0][1][2] * 100 + p_origin
    #     trace_x = go.Scatter3d(x=[p_origin[0], pcs[0][1][0][0]],
    #                            y=[p_origin[1], pcs[0][1][0][1]],
    #                            z=[p_origin[2], pcs[0][1][0][2]],
    #                            mode="lines",
    #                            marker=dict(color="red"))
    #     trace_y = go.Scatter3d(x=[p_origin[0], pcs[0][1][1][0]],
    #                            y=[p_origin[1], pcs[0][1][1][1]],
    #                            z=[p_origin[2], pcs[0][1][1][2]],
    #                            mode="lines",
    #                            marker=dict(color="green"))
    #     trace_z = go.Scatter3d(x=[p_origin[0], pcs[0][1][2][0]],
    #                            y=[p_origin[1], pcs[0][1][2][1]],
    #                            z=[p_origin[2], pcs[0][1][2][2]],
    #                            mode="lines",
    #                            marker=dict(color="blue"))
    #     return [trace_x, trace_y, trace_z]

    # def _make_jcs_traces(self, frame):
    #     """Returns a list of Plotly  traces that display a Joint Coordinate
    #     System for each ball joint respectively."""
    #     # p = self.sequence.positions
    #     # bps = self.sequence.body_parts
    #     # ls_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftShoulder"]], p[frame, bps["RightShoulder"]], p[frame, bps["Torso"]])
    #     # rs_lcs_traces = self._make_lcs_trace(p[frame, bps["RightShoulder"]], p[frame, bps["LeftShoulder"]], p[frame, bps["Torso"]])
    #     # lh_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftHip"]], p[frame, bps["RightHip"]], p[frame, bps["Torso"]])
    #     # rh_lcs_traces = self._make_lcs_trace(p[frame, bps["RightHip"]], p[frame, bps["LeftHip"]], p[frame, bps["Torso"]])

    #     # Get Local Coordinate System vectors
    #     jcs_traces = []
    #     scene_graph = self.sequence.scene_graph
    #     nodes = list(scene_graph.nodes)
    #     for node in nodes:
    #         node_data = scene_graph.nodes[node]['coordinate_system']
    #         origin = node_data['origin'][frame]
    #         x_axis = node_data['x_axis'][frame]
    #         y_axis = node_data['y_axis'][frame]
    #         z_axis = node_data['z_axis'][frame]
    #         jcs_traces += self._make_lcs_trace(origin, x_axis, y_axis, z_axis)

    #     return jcs_traces

    def _get_traces(self, frame):
        """Returns joint, limb and JCS Plotly traces."""
        joint_traces = self._make_joint_traces(frame)
        # limb_traces = self._make_limb_traces(frame)
        # jcs_traces = self._make_jcs_traces(frame)

        return joint_traces  #+ limb_traces + jcs_traces

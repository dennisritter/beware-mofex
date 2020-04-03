import numpy as np
import cv2
import json
from pathlib import Path
from mofex.models.sequence import Sequence

# h = 227
# w = 227
# img = np.full((h, w, 3), 123, dtype=np.uint8)
# img[:, 0:114] = (15, 99, 201)
# cv2.imshow('Cool Image.', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load Sequences from json files
root = 'data/sequences/mka_samples/'
sequences = []
for filename in Path(root).rglob('sample_squat_1.json'):
    sequences.append(Sequence.from_mka_file(filename))

# min/max tracking positions to map to [0,255]
# TODO: Define better min/max tracked positions
MIN = -1000
MAX = 1000
for seq in sequences:
    img = np.zeros((len(seq.positions[0, :]), len(seq), 3), dtype='uint8')
    pos_colors_x = np.interp(seq.positions[:, :, 0], [-1000, 1000], [0, 255]).swapaxes(0, 1)
    pos_colors_y = np.interp(seq.positions[:, :, 1], [-1000, 1000], [0, 255]).swapaxes(0, 1)
    pos_colors_z = np.interp(seq.positions[:, :, 2], [-1000, 1000], [0, 255]).swapaxes(0, 1)
    img[:, :, 0] = pos_colors_x
    img[:, :, 1] = pos_colors_y
    img[:, :, 2] = pos_colors_z
    img = cv2.resize(img, (227, 227))
    cv2.imshow('Pos Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cv2.__file__)

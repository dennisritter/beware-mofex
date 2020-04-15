import numpy as np
import cv2
import json
from pathlib import Path
from mofex.models.sequence import Sequence

# Load Sequences from json files
root = 'data/sequences/'
sequences = []
for filename in Path(root).rglob('sample_squat_1.json'):
    sequences.append(Sequence.from_mka_file(filename, name=str(filename)))

# min/max tracking positions to map to [0,255]
# TODO: Define better min/max tracked positions
for seq in sequences:
    img = seq.to_motionimg(output_size=(200, 200))
    cv2.imshow(seq.name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

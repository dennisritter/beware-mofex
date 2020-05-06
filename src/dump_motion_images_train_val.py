import numpy as np
import cv2
import json
import os
from pathlib import Path
from mofex.preprocessing.sequence import Sequence
from sklearn.model_selection import train_test_split
# Root folder for JSON Sequence files
# root = 'data/sequences/hdm05-122/c3d'
root = 'data/sequences/hdm05-122/c3d'

### Load Sequences
filenames = []
sequences = []
labeled_sequences_dict = {}
for filename in Path(root).rglob('*.c3d'):
    print(filename)
    name = str(filename).split("\\")[-1]
    desc = str(filename).split("\\")[-2]  # desc = class
    # Append sequence to label key or add new label key if not present already
    # * Change the sequence loading function depending on the MoCap format (from_mir_file, from_mka_file, ...)
    if desc in labeled_sequences_dict.keys():
        labeled_sequences_dict[desc].append(Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc))
    else:
        labeled_sequences_dict[desc] = [Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc)]
    # sequences.append(Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc))
print(f"Loaded Sequence Files")
train_seqs = []
val_seqs = []
for label in labeled_sequences_dict.keys():
    label_split = train_test_split(labeled_sequences_dict[label], test_size=0.1, random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'data/motion_images/hdm05-122/train/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256))
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'data/motion_images/hdm05-122/val/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256))
    cv2.imwrite(out, img)

print(f"Created, split and dumped Motion Images")

import numpy as np
import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import torch
from torchvision import transforms

start = datetime.now()

root = 'data/sequences/'
show_img = False
visualize_skeleton = False

### Load Sequences
filenames = []
sequences = []
for filename in Path(root).rglob('*.json'):
    sequences.append(Sequence.from_mka_file(filename, name=str(filename)))
    filenames.append(filename)

### Create Motion Images
motion_images = []
for seq in sequences:
    img = seq.to_motionimg(output_size=(256, 256))
    if show_img:
        cv2.imshow(seq.name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if visualize_skeleton:
        sv = SkeletonVisualizer(seq)
        sv.show()

    motion_images.append(img)

### Get feature Vectors from CNN
model = resnet.load_resnet18(pretrained=True, remove_last_layer=True)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_vectors = []
for img in motion_images:
    model.eval()
    img_tensor = torch.from_numpy(img)

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)
    output = output.cpu().detach().numpy().reshape((512))
    feature_vectors.append(output)

### Compare Feature Vectors
for i, gt_feat in enumerate(feature_vectors):
    gt_filename = filenames[i]
    # print("------------------------------")
    # print(f"Distances for [{gt_filename}]")
    for j, test_feat in enumerate(feature_vectors):
        test_filename = filenames[j]
        distance = np.linalg.norm(gt_feat - test_feat)
        # print(f"[{test_filename}] : {distance}")

    # print("------------------------------")

print(
    f"Creating Motion Images [{len(motion_images)}], Creating Feature Vectors [{len(feature_vectors)}], Comparing Feature Vectors [{len(feature_vectors)*len(feature_vectors)}]"
)
print(f"Runtime of script: {datetime.now() - start}")
### ESTIMATE IMAGENET CLASS OF "NORMAL" RGB IMAGE
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# softmax_output = torch.nn.functional.softmax(output[0], dim=0)
# output_class_idx = torch.max(softmax_output, 0).indices
# output_class_probability = torch.max(softmax_output, 0)
# print(output_class_probability)

# # Print the identified class
# dir = os.path.dirname(__file__)
# with open(os.path.join(dir, 'mofex/postprocessing/imagenet-simple-labels.json')) as f:
#     labels = json.load(f)

# def class_id_to_label(i):
#     return labels[i]

# print(class_id_to_label(output_class_idx))

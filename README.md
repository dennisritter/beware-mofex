# MOFEX - MoCap Feature Extractor

MOFEX is a Feature Extractor for motion capturing data.

## Desription

### Summary

The process is heavily based on the paper *[Effective and efficient similarity searching in motion capture data](http://link.springer.com/10.1007/s11042-017-4859-7)*

The Motion Feature Extractor (MOFEX) generates feature vectors from 3-D Motion Capturing Data that represent the recorded motion. Therefore, the following steps are executed:

1. Generate 'Motion Images' from 3-D MoCap Data
2. Finetune ImageNet pretrained CNN to classify generated Motion Images
3. Remove classification layer (last layer) from CNN
4. (optional) Add new last layer with desired output dimensionality
5. Create Motion Feature Vectors by processing Images in forward pass of the finetuned CNN.
6. Use Feature Vectors as Motion representation for several tasks

### Motivation

* Measure difference between motion sequences
  * Retrieve motion sequences from databases
  * Identify type of motion
  * Rate quality of execution (sport exercises)
  * Identify repeating sub-motions in motions (e.g. 1 squat in sequence of 10 squats)

### Method

#### Motion Image Generation

(Method from [Effective and efficient similarity searching in motion capture data](http://link.springer.com/10.1007/s11042-017-4859-7))
1. (Normalize MoCap data) 
1. Limit min and max values for X,Y,Z positions of captured positions (body part positions)
2. Translate X,Y,Z positions to respective R,G,B color channel values [0,255] so a RGB color value represents a X,Y,Z coordinate
3. Create a RGB image where each tracked body part is represented by a line and each frame is represented by a column
4. Resize image to CNN input size (256x256 at the moment, so max number of body parts and frames is 256 right now)

##### Note
In the current state, we use the HDM05-122 MoCap Dataset as input data (2326 motion sequences, 122 classes). The Motion Image generation is simplified by using **no normalization** and **limiting the X,Y,Z space to [-1000, 1000]**. We have to find out which normalizations should be applied to the input data in order to get the most meaningful Motion Image representation and how to *smartly* limit the X,Y,Z coordinate values.
*(Already implemented normalizations: Center Positions, Positions relative to defined position, Rotate first frame frontal to camera)*

#### Feature Vector Generation

1. Get pretrained ImageNet CNN
2. Finetune CNN with domain specific Motion Images
3. Remove softmax layer
4. Add layer to define output dimensions or simply take generated output features

##### Note
In the current state, we used ResNet18, ResNet50 and ResNet101 pretrained on ImageNet and finetuned on HDM05-122 Motion Images as stated above. For now, best performance has been achieved with a ResNet101 CNN after 50 epochs of training with a 90/10 (train/val) dataset split.

Correct: 271/285; Incorrect: 14/285; Accuracy: 0.9508771929824561

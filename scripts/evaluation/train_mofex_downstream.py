# The idea is to firstly train mofex (with motion images) on HDM05 with a classification task,
# then fine-tune on downstream repetition identification task.

# 1. Setup HDM05 dataset
#     a. Full reps -> Classification
#     b. Split full reps into random chunk sizes -> Classification
#         1. 1, 2, 4, 8, ..., frames?
#         2. Generate different chunk sizes for each class not one random size foreach example
#     c. Implement suitable DataLoader
# 2. Pre-Train MOFEX
#     a. Pre-pare MOFEX model with additional (easy to remove) classification layer1
#     b. Pre-train ResNet on new Dataset
#     c. save best model
# 3. Setup MKA dataset
#     a. Full reps -> rep identification
#     b. Split full reps into stream-like chunks
#         1. 1 frame, 1+1 frame, 1+1+1 frame, etc...
#         2. Two folders: successful & fail rep
#         3. There will be a lot of ! reps and less correct/ full ones
#     c. Implement suitable DataLoader
# 4. Fine-tune MOFEX
#     a. Add fine-tune layer to pre-trained ResNet
#     b. Fine-tune on MKA dataset
#     c. save best model
# 5. (OPTIONAL) Evaluate against old RepNet implementation
#     a. Probably re-train ResNet on MKA with splitting logic from 1.b.
#     b. How to evaluate?

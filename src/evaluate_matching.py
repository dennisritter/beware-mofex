import os
import mofex.feature_vectors as featvec
import mofex.motion_matching as motion_matching

train_featvecs_path = 'data/feature_vectors/hdm05-122/resnet18_hdm05-122_50-50/train/resnet18_hdm05-122_50-50_train.json'
val_featvecs_path = 'data/feature_vectors/hdm05-122/resnet18_hdm05-122_50-50/val/resnet18_hdm05-122_50-50_val.json'
matching_top1_result_output_path = 'data/feature_vectors/hdm05-122/resnet18_hdm05-122_50-50/matching_results'

# * featvec.load_from_file(<path>) returns a list of tuples -> [(<id_name>,<feature_vector>,<label>), ...]
train_featvecs = featvec.load_from_file(train_featvecs_path)
val_featvecs = featvec.load_from_file(val_featvecs_path)

motion_matching.evaluate_matching_top1(train_featvecs=train_featvecs,
                                       val_featvecs=val_featvecs,
                                       make_graph=False,
                                       result_dir_path=matching_top1_result_output_path)

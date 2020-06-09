import os
import mofex.feature_vectors as featvec
import mofex.motion_matching as motion_matching

dataset_name = 'cmu-30_80-20_256'
model_name = 'resnet101_cmu-30_80-20_256'
train_featvecs_path = f'data/feature_vectors/{dataset_name}/{model_name}/train/{model_name}_train.json'
val_featvecs_path = f'data/feature_vectors/{dataset_name}/{model_name}/val/{model_name}_val.json'
matching_top1_result_output_path = f'data/feature_vectors/{dataset_name}/{model_name}/matching_results'

# * featvec.load_from_file(<path>) returns a list of tuples -> [(<id_name>,<feature_vector>,<label>), ...]
train_featvecs = featvec.load_from_file(train_featvecs_path)
val_featvecs = featvec.load_from_file(val_featvecs_path)

motion_matching.evaluate_matching_top1(train_featvecs=train_featvecs,
                                       val_featvecs=val_featvecs,
                                       make_graph=False,
                                       result_dir_path=matching_top1_result_output_path)

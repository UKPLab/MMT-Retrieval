import base64
import csv
import ctypes
import json

from mmt_retrieval.model.models import OSCAR, ClassificationHead
from mmt_retrieval import MultimodalTransformer
import torch
import os
import numpy as np

def convert_finetuned_oscar(oscar_model_folder, model_args={}):
    """
    Convert a fine-tuned OSCAR model downloaded from https://github.com/microsoft/Oscar/blob/master/MODEL_ZOO.md
    to a model in our format.
    :param oscar_model_folder: Folder where the fine-tuned model is stored
    :param model_args: Arguments for the created OSCAR model. See mmt_retrieval.model.models.OSCAR
    :return: A MMEMBTransformer with the OSCAR model and a classification head matching the fine-tuned model
    """
    oscar = OSCAR(model_path=oscar_model_folder, **model_args)
    classifier_state_dict = torch.load(os.path.join(oscar_model_folder, "pytorch_model.bin"))
    config = json.load(open(os.path.join(oscar_model_folder, "config.json")))
    classifier = ClassificationHead(num_labels=config.get("num_labels", 2), input_key="pooled_cls_token_embeddings",
                                    input_dim=config["hidden_size"], classifier_type=config.get("classifier", "linear"),
                                    scaling_factor=config.get("cls_hidden_scale", 2), dropout=config["hidden_dropout_prob"])
    classifier.load_state_dict(classifier_state_dict, strict=False)
    model = MultimodalTransformer(modules=[oscar, classifier])
    return model



def split_oscar_image_feature_file_to_npz(oscar_image_feature_file, out_folder):
    """
    Split the Image Features downloaded from here: https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md
    to the .npz single file format that we support.
    Intended for just-in-time loading support.
    :param oscar_image_feature_file: The downloaded .pt file with the image features
    :param out_folder: Folder where the .npz files will be saved
    """
    features = torch.load(oscar_image_feature_file)
    os.makedirs(out_folder, exist_ok=True)
    for k, v in features.items():
        res = {'img_h': 1.0, 'img_w': 1.0, 'num_boxes': v.shape[0],
               "features": v[:, 0:2048].numpy(), "boxes": v[:, 2048:2052].numpy()}
        output_file = os.path.join(out_folder, f"{k}.npz")
        np.savez_compressed(output_file, x=res["features"], bbox=res["boxes"], num_bbox=res["num_boxes"], image_h=1.0, image_w=1.0)

def split_tsv_features_to_npz(tsv_image_feature_file, out_folder):
    """
    Split the Image Features generated in the .tsv format from https://github.com/peteanderson80/bottom-up-attention
    to the .npz single file format that we support.
    Intended for just-in-time loading support.
    :param tsv_image_feature_file: the .tsv file with the image features
    :param out_folder: Folder where the .npz files will be saved
    """
    FIELDNAMES = ["img_id", "img_w", "img_h", "num_boxes", "boxes",
                  "features"]
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
    os.makedirs(out_folder, exist_ok=True)
    with open(tsv_image_feature_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for item in reader:
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            boxes = item['num_boxes']
            decode_config = [
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            output_file = os.path.join(out_folder, f"{item['img_id']}.npz")
            np.savez_compressed(output_file, x=item["features"], bbox=item["boxes"], num_bbox=item["num_boxes"], image_h=item["img_h"], image_w=item["img_w"])
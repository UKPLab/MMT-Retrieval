import base64
import concurrent
import csv
import ctypes
import os
from typing import Set, Optional, Union, List

import torch
import numpy as np

class ImageFeature:
    def __init__(self, path=None, value_dict=None):
        assert path is not None or value_dict is not None
        assert value_dict is None or all(key in value_dict for key in ["features", "boxes", "num_boxes", "img_h", "img_w"])
        self.path = path
        self.value_dict = value_dict


class ImageDict(dict):
    def __init__(self):
        self.__dict__ = dict()

    def __setitem__(self, key, item):
        if not isinstance(item, ImageFeature):
            if isinstance(item, dict):
                item = ImageFeature(value_dict=item)
            else:
                item = ImageFeature(path=item)
        self.__dict__[key] = item

    def __getitem__(self, key):
        value = self.__dict__[key]
        if value.value_dict is not None:
            return value.value_dict
        else:
            path = value.path
            folder, file = os.path.split(path)
            name, ending = file.split(".")
            if ending == "npz":
                res = np.load(os.path.join(folder, file))
                data = {"features": res["x"], "boxes": res["bbox"], "num_boxes": res["num_bbox"],
                              "img_h": res["image_h"], "img_w": res["image_w"]}
            else:
                res = torch.load(os.path.join(folder, file))
                data = {"features": res["features"], "boxes": res["boxes"], "num_boxes": res["num_boxes"],
                              "img_h": res["img_h"], "img_w": res["img_w"]}
            return data

    def update(self, *args, **kwargs):
        for key, item in args[0].items():
            self[key] = item
        return self

    def load_file_names(self, folder, select: Optional[Union[List, Set]]=None):
        """
        Load all feature files from a folder into the dict for later just-in-time loading
        :param folder: The folder with the feature files
        :param select: Only load features with id (i.e., file name) in the set/ list.
        """
        if select and not isinstance(set, select):
            select = set(select)
        if select is None:
            self.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder)})
        else:
            self.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder) if key.split(".")[0] in select})

    def load_obj_tsv(self, fname : str, topk : int=None):
        """Load object features from tsv file.

        :param fname: The path to the tsv file.
        :param topk: Only load features for top K images (lines) in the tsv file.
            Will load all the features if topk is either -1 or None.
        """
        FIELDNAMES = ["img_id", "img_w", "img_h", "num_boxes", "boxes",
                      "features"]
        csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
        with open(fname, encoding="utf-8") as f:
            reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
            for i, item in enumerate(reader):
                if topk is not None and i == topk:
                    break
                if item['img_id'] in self:
                    continue
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
                self[item['img_id']] = item

    def load_oscar_format_image_features(self, fname : str, topk : int=None, select: Optional[Union[List, Set]]=None):
        """Load object features from OSCAR-formatted feature file.

        :param fname: The path to the file.
        :param topk: Only load features for top K images.
            Will load all the features if topk is either -1 or None.
        :param select: Only load features with id in the set/ list.
        """
        if select and not isinstance(set, select):
            select = set(select)
        feature_dict = torch.load(fname)
        for i, (key, item) in enumerate(feature_dict.items()):
            key = str(key)
            if select is not None and key not in select:
                continue
            if topk is not None and i == topk:
                break
            if key in self:
                continue
            res = {'img_h': 1.0, 'img_w': 1.0, 'num_boxes': item.shape[0],
                   "features": item[:, 0:2048].numpy(), "boxes": item[:, 2048:2052].numpy()} # width and height are already normalized
            self[key] = res


    def load_features_folder(self, folder, max_workers=0, extension="npz", select: Optional[Union[List, Set]]=None, min_len=0):
        """
        Load object features from files in a folder using a thread pool for quick loading.
        The id of the features will be the file name.

        :param folder: The folder with the feature files
        :param max_workers: Max number of workers in thread pool. Default min(32, os.cpu_count()*5)
        :param extension: The file extension of the features. Either 'pt' or 'npz'
        :param select: Only load features with id (i.e., file name) in the set/ list
        :param min_len: Minimum length for the id. Ids shorter will be padded with a 0 followed by 9s until the length fits.
            This parameter can be used to avoid collisions between ids.
        :return:
        """
        if select and not isinstance(set, select):
            select = set(select)
        if max_workers == 0:
            max_workers=min(32, os.cpu_count()*5)
        if select is None:
            files = os.listdir(folder)
        else:
            files = [f"{iid}.{extension}" for iid in select]
        def load_file(file):
            image_id, ext = file.split(".")
            if len(image_id) < min_len:
                image_id = "9" * (min_len-1-len(image_id)) + "0" + image_id
            if extension != ext:
                return
            if image_id in self:
                return
            if ext == "pt":
                res = torch.load(os.path.join(folder, file))
                self[image_id] = {"features": res["features"], "boxes": res["boxes"], "num_boxes": res["num_boxes"],
                                  "img_h": res["img_h"], "img_w": res["img_w"]}
            elif ext == "npz":
                res = np.load(os.path.join(folder, file))
                self[image_id] = {"features": res["x"], "boxes": res["bbox"], "num_boxes": res["num_bbox"],
                                  "img_h": res["image_h"], "img_w": res["image_w"]}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(load_file, files)


    # overwriting remaining dict functions with default. might not produce intended results
    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    #def copy(self):
    #    return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    #def values(self):
    #    return self.__dict__.values()

    #def items(self):
    #    return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

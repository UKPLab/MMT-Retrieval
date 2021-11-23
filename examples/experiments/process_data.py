import base64
import csv
import json
import os
import logging
import ctypes
from collections import defaultdict
import numpy as np
import torch
from mmt_retrieval.data.datasets import ImageSentenceTripletDataset, \
    MultidatasetImageSentenceTripletDataset, MultidatasetImageSentenceTripletSampler, \
    JointImageSentenceDataset
import concurrent.futures


def _load_obj_tsv(fname, data, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    logging.info("Start to load Faster-RCNN detected objects from %s" % fname)
    FIELDNAMES = ["img_id", "img_w", "img_h", "num_boxes", "boxes",
                  "features"]
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
    with open(fname, encoding="utf-8") as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            if topk is not None and i == topk:
                break
            if item['img_id'] in data:
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
            data[item['img_id']] = item

    return data

def _load_features_folder(folder, data, extension="pt", image_ids=None, min_len=0):
    logging.info("Start to load Faster-RCNN detected objects from %s" % folder)
    if image_ids is None:
        files = os.listdir(folder)
    else:
        files = [f"{iid}.{extension}" for iid in image_ids]
    def load_file(file):
        image_id, ext = file.split(".")
        if len(image_id) < min_len:
            image_id = "9" * (min_len-1-len(image_id)) + "0" + image_id
        if extension != ext:
            return
        if image_id in data:
            return
        if ext == "pt":
            res = torch.load(os.path.join(folder, file))
            data[image_id] = {"features": res["features"], "boxes": res["boxes"], "num_boxes": res["num_boxes"],
                              "img_h": res["img_h"], "img_w": res["img_w"]}
        elif ext == "npz":
            res = np.load(os.path.join(folder, file))
            data[image_id] = {"features": res["x"], "boxes": res["bbox"], "num_boxes": res["num_bbox"],
                              "img_h": res["image_h"], "img_w": res["image_w"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*5)) as executor:
        executor.map(load_file, files)
    return data

def _flickr_image_feature_split(flickr_split_folder, flickr30k_original_folder, image_feature_folder, split):
    """
    :param flickr_split_folder: Output folder for processed split
    :param flickr30k_original_folder: https://github.com/BryanPlummer/flickr30k_entities. Unzip annotations.zip
    :param image_feature_folder: Folder with downloaded .tsv feature files
    https://github.com/jnhwkim/ban-vqa/blob/master/tools/download_flickr.sh
    https://public.ukp.informatik.tu-darmstadt.de/geigle/mmt-retrieval/flickr30k_features.zip
    :param split: test/ train/ dev
    :return:
    """
    if os.path.exists(os.path.join(flickr_split_folder, 'train_flickr30k_resnet101_faster_rcnn_genome.tsv')):
        logging.info(f"Flickr30k Image Feature Split exists. Loading from {flickr_split_folder}")
        return os.path.join(flickr_split_folder, f'{split}_flickr30k_resnet101_faster_rcnn_genome.tsv')
    logging.info(f"Flickr30k Image Feature Split does not exist. Creating in {flickr_split_folder}")
    train_images, dev_images, test_images = set(), set(), set()
    for image_set, name in zip([train_images, dev_images, test_images], ["train", "val", "test"]):
        image_set.update([l.strip() for l in open(os.path.join(flickr30k_original_folder, f"{name}.txt")).readlines()])

    tsv_file_names = [
        os.path.join(image_feature_folder, 'test_flickr30k_resnet101_faster_rcnn_genome.tsv.3'),
        os.path.join(image_feature_folder, 'train_flickr30k_resnet101_faster_rcnn_genome.tsv.1'),
        os.path.join(image_feature_folder, 'train_flickr30k_resnet101_faster_rcnn_genome.tsv.2'),
        os.path.join(image_feature_folder, 'val_flickr30k_resnet101_faster_rcnn_genome.tsv.3'),
    ]
    with open(os.path.join(flickr_split_folder, 'train_flickr30k_resnet101_faster_rcnn_genome.tsv'), 'w') as train_f:
        with open(os.path.join(flickr_split_folder, 'dev_flickr30k_resnet101_faster_rcnn_genome.tsv'), 'w') as dev_f:
            with open(os.path.join(flickr_split_folder, 'test_flickr30k_resnet101_faster_rcnn_genome.tsv'), 'w') as test_f:
                for tsv_file_name in tsv_file_names:
                    with open(tsv_file_name, 'r', encoding="utf-8") as f:
                        print('reading ',tsv_file_name )
                        for i, line in enumerate(f):
                            if i%500 == 0:
                                print(f"line {i}")
                            elems = line.split()
                            image_id = elems[0]
                            if image_id in dev_images:
                                dev_f.write(line)
                            elif image_id in train_images:
                                train_f.write(line)
                            elif image_id in test_images:
                                test_f.write(line)
    return os.path.join(flickr_split_folder, f'{split}_flickr30k_resnet101_faster_rcnn_genome.tsv')


def _flickr_split(flickr_split_folder, flickr30k_original_folder, tiny, split, image_dict):
    """

    :param flickr_split_folder: Output folder for processed split
    :param flickr30k_original_folder: https://github.com/BryanPlummer/flickr30k_entities. Unzip annotations.zip
    :param tiny:
    :param split:
    :param image_dict:
    :return:
    """
    if not os.path.exists(os.path.join(flickr_split_folder, f"flickr30k_{split}.tsv")):
        train_images, dev_images, test_images = set(), set(), set()
        for image_set, name in zip([train_images, dev_images, test_images], ["train", "val", "test"]):
            image_set.update([l.strip() for l in open(os.path.join(flickr30k_original_folder, f"{name}.txt")).readlines()])
        def read_captions(file):
            with open(file, 'r', encoding="utf-8") as f:
                sentences = f.read().split('\n')
            captions = []
            for sentence in sentences:
                if not sentence:
                    continue

                words = []
                for token in sentence.split():
                    if token[-1] == ']':
                        token = token[:-1]
                        words.append(token)
                    elif token[0] != '[':
                        words.append(token)
                captions.append(' '.join(words))
            return captions
        train_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"flickr30k_train.tsv"), "w", newline="\n", encoding="utf-8"),
                                      fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        dev_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"flickr30k_dev.tsv"), "w", newline="\n", encoding="utf-8"),
                                    fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        test_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"flickr30k_test.tsv"), "w", newline="\n", encoding="utf-8"),
                                     fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        train_writer.writeheader()
        dev_writer.writeheader()
        test_writer.writeheader()
        for image in train_images:
            captions = read_captions(os.path.join(flickr30k_original_folder, "Sentences", f"{image}.txt"))
            for i, c in enumerate(captions, start=1):
                train_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
        for image in dev_images:
            captions = read_captions(os.path.join(flickr30k_original_folder, "Sentences", f"{image}.txt"))
            for i, c in enumerate(captions, start=1):
                dev_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
        for image in test_images:
            captions = read_captions(os.path.join(flickr30k_original_folder, "Sentences", f"{image}.txt"))
            for i, c in enumerate(captions, start=1):
                test_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
    if not tiny:
        return os.path.join(flickr_split_folder, f"flickr30k_{split}.tsv")
    else:
        if os.path.exists(os.path.join(flickr_split_folder, f"flickr30k_{split}_tiny.tsv")):
            return os.path.join(flickr_split_folder, f"flickr30k_{split}_tiny.tsv")
        else:
            reader = csv.DictReader(open(os.path.join(flickr_split_folder, f"flickr30k_{split}.tsv"), "r", encoding="utf-8"), delimiter="\t")
            writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"flickr30k_{split}_tiny.tsv"), "w", newline="\n", encoding="utf-8"),
                                    fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            for row in reader:
                if row["imageid"].replace(".jpg", "") in image_dict:
                    writer.writerow(row)
        return os.path.join(flickr_split_folder, f"flickr30k_{split}_tiny.tsv")


def _multi30k_split(flickr_split_folder, multi30k_original_folder, tiny, split, image_dict, language="en"):
    """

    :param flickr_split_folder: Output folder for processed split
    :param multi30k_original_folder: https://github.com/multi30k/dataset/tree/master/data. Unzip all files in */raw
    :param tiny:
    :param split:
    :param image_dict:
    :param language:
    :return:
    """
    if language in ["en", "de"]:
        task = "task2"
    else:
        task = "task1"
    lang_flag = f"_{language}"

    if not os.path.exists(os.path.join(flickr_split_folder, f"multi30k_{split}{lang_flag}.tsv")):
        def read_ids(file):
            with open(file, "r", ) as f:
                return [l.strip().replace(".jpg", "") for l in f.readlines()]
        def read_captions(file):
            captions = []
            if task == "task2":
                for i in range(1, 6):
                    with open(f"{file}.{i}.{language}", "r", encoding="utf-8") as f:
                        captions.append([l.strip() for l in f.readlines()])
            else:
                with open(f"{file}.{language}", "r", encoding="utf-8") as f:
                    captions.append([l.strip() for l in f.readlines()])
            return list(zip(*captions))

        train_captions = read_captions(os.path.join(multi30k_original_folder, task, "raw", "train"))
        dev_captions = read_captions(os.path.join(multi30k_original_folder, task, "raw", "val"))
        if task == "task2":
            test_captions = read_captions(os.path.join(multi30k_original_folder, task, "raw", "test_2016"))
            train_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "train_images.txt"))
            dev_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "val_images.txt"))
            test_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "test_2016_images.txt"))
        else:
            test_captions = read_captions(os.path.join(multi30k_original_folder, task, "raw", "test_2016_flickr"))
            train_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "train.txt"))
            dev_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "val.txt"))
            test_images = read_ids(os.path.join(multi30k_original_folder, task, "image_splits", "test_2016_flickr.txt"))
        #original_images = original_train_images+original_dev_images+original_test_images
        #original_captions = train_captions+dev_captions+test_captions
        train_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"multi30k_train{lang_flag}.tsv"), "w", newline="\n", encoding="utf-8"),
                                      fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        dev_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"multi30k_dev{lang_flag}.tsv"), "w", newline="\n", encoding="utf-8"),
                                    fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        test_writer = csv.DictWriter(open(os.path.join(flickr_split_folder, f"multi30k_test{lang_flag}.tsv"), "w", newline="\n", encoding="utf-8"),
                                     fieldnames=["imageid", "caption", "caption_idx"], delimiter="\t")
        train_writer.writeheader()
        dev_writer.writeheader()
        test_writer.writeheader()
        for image, captions in zip(train_images, train_captions):
            for i, c in enumerate(captions, start=1):
                train_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
        for image, captions in zip(dev_images, dev_captions):
            for i, c in enumerate(captions, start=1):
                dev_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
        for image, captions in zip(test_images, test_captions):
            for i, c in enumerate(captions, start=1):
                test_writer.writerow({"imageid": image, "caption": c, "caption_idx": i})
        return os.path.join(flickr_split_folder, f"multi30k_{split}{lang_flag}.tsv")


def get_hard_file(path, do_hard):
    if not os.path.exists(path) and do_hard:
        raise RuntimeError("Hard file does not exist")
    if os.path.exists(path) and do_hard:
        return torch.load(path)
    else:
        return None

def get_flickr30k(config, split, tiny, image_dict, **args):
    image_feature_folder = config["image_feature_folder"]
    flickr_split_folder = config["flickr_split_folder"]
    flickr30k_original_folder = config["flickr30k_original_folder"]
    hard_file = config.get("flickr30k_hard_file", "")
    topk = None
    if tiny:
        if split == "train":
            topk = 1000
        else:
            topk = 256
    if not args.get("skip_image_features", False):
        if args.get("jit_loading", False):
            folder = os.path.join(flickr_split_folder, f"features_{split}")
            image_dict.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder)})
        elif args.get("feature_source", "caffe") == "caffe":
            image_dict = _load_obj_tsv(_flickr_image_feature_split(flickr_split_folder, flickr30k_original_folder, image_feature_folder, split),
                                       image_dict, topk)
        elif args.get("feature_source", "caffe") == "pytorch_caffe":
            image_dict = _load_features_folder(os.path.join(flickr_split_folder, f"features_{split}"), image_dict, extension="npz")
    file = _flickr_split(flickr_split_folder, flickr30k_original_folder, tiny, split, image_dict)

    imageids = set()
    captions = {}
    imageid2captions = defaultdict(set)
    for row in csv.DictReader(open(file, "r", encoding="utf-8"), delimiter="\t"):
        if not "captions_per_image" in args or len(imageid2captions[row["imageid"]]) < args["captions_per_image"]:
            imageids.add(row["imageid"])
            caption_id = f"{row['imageid']}_{row['caption_idx']}"
            captions[caption_id] = row["caption"]
            imageid2captions[row["imageid"]].add(caption_id)
    imageids = list(imageids)

    tags = None
    if args.get("tags", False):
        tag_dict = torch.load(os.path.join(flickr_split_folder, f"{split}_vg_tags.pt"))
        tags = [tag_dict[i] for i in imageids]

    if "additional_datasets" in args and split != "train":
        datasets = args.pop("additional_datasets")
        for dataset in datasets:
            data = DATA_LOADER[dataset](config, split, tiny, image_dict, **args)
            imageids.extend(data["images"])
            captions.update(data["captions"])
            if tags:
                tags.extend(data["tags"])

    if split == "train" and not args.get("return_dict", False):
        if args.get("joint", False):
            return JointImageSentenceDataset(imageids, captions, imageid2captions, tags=tags,
                                             hard_examples=get_hard_file(hard_file, args.get("hard", False)), hard_p=args.get("hard_p", None))
        return ImageSentenceTripletDataset(imageids, captions, imageid2captions, tags=tags,
                                           hard_examples=get_hard_file(hard_file, args.get("hard", False)),
                                           negative_examples=args.get("negative_examples", 0))
    else:
        return {"images": imageids, "captions": captions, "imageid2captions": imageid2captions, "tags": tags}

def get_multi30k(config, split, tiny, image_dict, **args):
    flickr_split_folder = config["flickr_split_folder"]
    multi30k_original_folder = config["multi30k_original_folder"]
    hard_file = config.get("flickr30k_hard_file", "")
    topk = None
    if tiny:
        if split == "train":
            topk = 1000
        else:
            topk = 256
    languages = args["languages"]
    if not args.get("skip_image_features", False):
        if args.get("jit_loading", False):
            folder = os.path.join(flickr_split_folder, f"features_{split}")
            image_dict.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder)})
        else:
            image_dict = _load_features_folder(os.path.join(flickr_split_folder, f"features_{split}"), image_dict, extension="pt")

    imageids = set()
    captions = {}
    imageid2captions = defaultdict(set)

    for language in languages:
        file = _multi30k_split(flickr_split_folder, multi30k_original_folder, tiny, split, image_dict, language)
        added = 0 # unlike Flickr30k, we have no tiny tsv so we "manually" stop
        for row in csv.DictReader(open(file, "r", encoding="utf-8"), delimiter="\t"):
            if not "captions_per_image" in args or len(imageid2captions[row["imageid"]]) < args["captions_per_image"]:
                added += 1
                if topk and added == topk:
                    break
                imageids.add(row["imageid"])
                caption_id = f"{row['imageid']}_{row['caption_idx']}_{language}"
                captions[caption_id] = row["caption"]
                imageid2captions[row["imageid"]].add(caption_id)

    if split == "train" and not args.get("return_dict", False):
        if args.get("joint", False):
            return JointImageSentenceDataset(list(imageids), captions, imageid2captions)
        if args.get("merge", True):
            return ImageSentenceTripletDataset(list(imageids), captions, imageid2captions, get_hard_file(hard_file, args.get("hard", False)))
        else:
            imageidss = [list(imageids) for _ in range(len(languages))]
            captionss = [{} for _ in range(len(languages))]
            imageid2captionss = [imageid2captions for _ in range(len(languages))]
            lan2id = {lan:i for i, lan in enumerate(languages)}
            for capt in captions.keys():
                lan = capt.split("_")[-1]
                captionss[lan2id[lan]][capt] = captions[capt]
            return MultidatasetImageSentenceTripletDataset(imageidss, captionss, imageid2captionss)
    else:
        return {"images": list(imageids), "captions": captions, "imageid2captions": imageid2captions}

def _load_oscar_format_image_features(fname, data, topk=None, select=None):
    logging.info("Start to load Faster-RCNN detected objects from %s" % fname)
    feature_dict = torch.load(fname)
    for i, (key, item) in enumerate(feature_dict.items()):
        key = str(key)
        if select is not None and key not in select:
            continue
        if topk is not None and i == topk:
            break
        if key in data:
            continue
        res = {'img_h': 1.0, 'img_w': 1.0, 'num_boxes': item.shape[0],
               "features": item[:, 0:2048].numpy(), "boxes": item[:, 2048:2052].numpy()} # width and height are already normalized
        data[key] = res
    return data

def _tiny_mscoco(mscoco_folder, k, split):
    if os.path.exists(os.path.join(mscoco_folder, f"{split}_img_keys_{k}.tsv")):
        return
    np.random.seed(2020)
    reader = csv.DictReader(open(os.path.join(mscoco_folder, f"{split}_img_keys.tsv"), "r", encoding="utf-8"), delimiter="\t")
    rows = list(reader)
    chosen = np.random.choice(len(rows), k, replace=False)
    writer = csv.DictWriter(open(os.path.join(mscoco_folder, f"{split}_img_keys_{k}.tsv"), "w", newline="\n", encoding="utf-8"),
                            fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for idx in chosen:
        writer.writerow(rows[idx])

def get_mscoco(config, split, tiny, image_dict, **args):
    mscoco_folder = config["mscoco_folder"] # via https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md
    hard_file = config.get("mscoco_hard_file", "")
    if split == "dev":
        split = "minival"
    if tiny:
        if split == "test" and "tiny_k" not in args:
            img_keys = f"{split}_img_keys_1k.tsv"
        else:
            _tiny_mscoco(mscoco_folder, args["tiny_k"], split)
            img_keys = f"{split}_img_keys_{args['tiny_k']}.tsv"
    else:
        img_keys = f"{split}_img_keys.tsv"
    imageids = [l.strip() for l in open(os.path.join(mscoco_folder, img_keys)).readlines()]
    if not args.get("skip_image_features", False):
        if args.get("jit_loading", False):
            folder = os.path.join(mscoco_folder, f"features_{split}")
            image_dict.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder)})
        else:
            image_dict = _load_oscar_format_image_features(os.path.join(mscoco_folder, f"{split}_img_frcnn_feats.pt"), image_dict, select=set(imageids))
    captions = {}
    imageid2captions = defaultdict(set)
    captions_dict = torch.load(os.path.join(mscoco_folder, f"{split}_captions.pt"))
    if not type(captions_dict[int(imageids[0])]) == list:
        captions_dict = {k: json.loads(captions_dict[int(k)]) for k in imageids}
    else:
        captions_dict = {k: captions_dict[int(k)] for k in imageids}
    for iid in imageids:
        caption = captions_dict[iid][:args.get("captions_per_image", len(captions_dict[iid]))]
        for i, c in enumerate(caption):
            caption_id = f"{iid}_{i}"
            captions[caption_id] = c
            imageid2captions[iid].add(caption_id)

    tags = None
    if args.get("tags", False):
        tag_dict = torch.load(os.path.join(mscoco_folder, f"{split}_vg_labels.pt"))
        tags = [tag_dict[int(i)] for i in imageids]

    if "additional_datasets" in args and split != "train":
        datasets = args.pop("additional_datasets")
        for dataset in datasets:
            data = DATA_LOADER[dataset](config, split, tiny, image_dict, **args)
            imageids.extend(data["images"])
            captions.update(data["captions"])
            if tags:
                tags.extend(data["tags"])

    if split == "train" and not args.get("return_dict", False):
        if args.get("joint", False):
            return JointImageSentenceDataset(list(imageids), captions, imageid2captions, tags, hard_examples=get_hard_file(hard_file, args.get("hard", False)))
        return ImageSentenceTripletDataset(imageids, captions, imageid2captions, tags,
                             hard_examples=get_hard_file(hard_file, args.get("hard", False)),
                             negative_examples=args.get("negative_examples", 0))
    else:
        return {"images": imageids, "captions": captions, "imageid2captions": imageid2captions, "tags": tags}



def _tiny_cc(cc_original_folder, k, split):
    if os.path.exists(os.path.join(cc_original_folder, f"cc_{split}_{k}.tsv")):
        return
    np.random.seed(2020)
    reader = csv.DictReader(open(os.path.join(cc_original_folder, f"cc_{split}.tsv"), "r", encoding="utf-8"), delimiter="\t")
    rows = list(reader)
    chosen = np.random.choice(len(rows), k, replace=False)
    writer = csv.DictWriter(open(os.path.join(cc_original_folder, f"cc_{split}_{k}.tsv"), "w", newline="\n", encoding="utf-8"),
                            fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for idx in chosen:
        writer.writerow(rows[idx])


def get_cc(config, split, tiny, image_dict, **args):
    cc_original_folder = config["cc_original_folder"]
    cc_feature_folder = config["cc_feature_folder"]

    if tiny:
        k = args.get("tiny_k", 1000)
        _tiny_cc(cc_original_folder, k, split)
        tiny_postfix = f"_{k}"
    else:
        tiny_postfix = ""

    imageids = []
    not_padded_imageids = []
    captions = {}
    imageid2captions = defaultdict(set)
    for row in csv.DictReader(open(os.path.join(cc_original_folder, f"cc_{split}{tiny_postfix}.tsv"), "r", encoding="utf-8"), delimiter="\t"):
        image_id = row["image_id"]
        not_padded_imageids.append(image_id)
        image_id = "9" * (11-len(image_id)) + "0" + image_id  # pad just in case when combined with other datasets
        imageids.append(image_id)
        caption_id = image_id
        captions[caption_id] = row["caption"]
        imageid2captions[image_id].add(caption_id)

    if not args.get("skip_image_features", False):
        if args.get("jit_loading", False):
            logging.warning("NO PADDING ADDED")
            folder = os.path.join(cc_feature_folder, f"features_{split}")
            image_dict.update({key.split(".")[0]: os.path.join(folder, key) for key in os.listdir(folder)})
        else:
            image_dict = _load_features_folder(os.path.join(cc_feature_folder, f"features_{split}"), image_dict, extension="npz",
                                       image_ids=not_padded_imageids, min_len=12)

    tags = None
    if args.get("tags", False):
        tag_dict = torch.load(os.path.join(cc_feature_folder, f"{split}_vg_tags.pt"))
        tags = [tag_dict[i] for i in not_padded_imageids]

    if "additional_datasets" in args and split != "train":
        datasets = args.pop("additional_datasets")
        for dataset in datasets:
            data = DATA_LOADER[dataset](config, split, tiny, image_dict, **args)
            imageids.extend(data["images"])
            captions.update(data["captions"])
            if tags:
                tags.extend(data["tags"])
    if split == "train" and not args.get("return_dict", False):
        if args.get("joint", False):
            return JointImageSentenceDataset(imageids, captions, imageid2captions, tags=tags)
        return ImageSentenceTripletDataset(imageids, captions, imageid2captions, tags, None, args.get("negative_examples", 0))
    else:
        return {"images": imageids, "captions": captions, "imageid2captions": imageid2captions, "tags": tags}

def get_sampler(name, dataset, batchsize, **args):
    if not args.get("merge", True):
        return None, MultidatasetImageSentenceTripletSampler(dataset, batchsize), False, 1
    else:
        return None, None, True, batchsize


DATA_LOADER = {
    "flickr30k": get_flickr30k,
    "mscoco": get_mscoco,
    "multi30k": get_multi30k,
    "cc": get_cc,
}



# def get_mscoco_and_flickr30k(config, split, tiny, image_dict, **args):
#     flickr30k = get_flickr30k(config, split, tiny, image_dict, return_dict=True, **args)
#     mscoco = get_mscoco(config, split, tiny, image_dict, return_dict=True, **args)
#
#     if split != "train":
#         raise ValueError("Only train split is supported for mscoco+flickr30k")
#
#     if args.get("merge", False):
#         return ImageSentenceTripletDataset(flickr30k["images"]+mscoco["images"], {**flickr30k["captions"], **mscoco["captions"]},
#                                            {**flickr30k["imageid2captions"], **mscoco["imageid2captions"]})
#     else:
#         return MultidatasetImageSentenceTripletDataset([flickr30k["images"], mscoco["images"]], [flickr30k["captions"], mscoco["captions"]],
#                                                        [flickr30k["imageid2captions"], mscoco["imageid2captions"]])
#
# def get_mscoco_and_flickr30k_and_cc(config, split, tiny, image_dict, **args):
#     flickr30k = get_flickr30k(config, split, tiny, image_dict, return_dict=True, **args)
#     mscoco = get_mscoco(config, split, True, image_dict, return_dict=True, tiny_k=30000, **args)
#     cc = get_cc(config, split, split, image_dict, return_dict=True, **args)
#     if split != "train":
#         raise ValueError("Only train split is supported for mscoco+flickr30k")
#
#     return MultidatasetImageSentenceTripletDataset([flickr30k["images"], mscoco["images"], cc["images"]],
#                                                    [flickr30k["captions"], mscoco["captions"], cc["captions"]],
#                                                        [flickr30k["imageid2captions"], mscoco["imageid2captions"], cc["imageid2captions"]])



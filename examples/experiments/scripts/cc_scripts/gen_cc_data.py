import csv
import os
import numpy as np
from tqdm import tqdm
np.random.seed(2020)

def gen_splits():
    # train_writer = csv.DictWriter(open(f"cc_train.tsv", "w", newline="\n", encoding="utf-8"),
    #                               fieldnames=["caption", "image_id"], delimiter="\t")
    # dev_writer = csv.DictWriter(open(f"cc_dev.tsv", "w", newline="\n", encoding="utf-8"),
    #                             fieldnames=["caption", "image_id"], delimiter="\t")
    test_writer = csv.DictWriter(open(f"cc_test.tsv", "w", newline="\n", encoding="utf-8"),
                                 fieldnames=["caption", "image_id"], delimiter="\t")
    # train_writer.writeheader()
    # dev_writer.writeheader()
    test_writer.writeheader()
    val_valid = []
    for f in os.listdir("features_test"):
        val_valid.append(f.split(".")[0])
    validation_tsv = [l.strip().split("\t")[0] for l in open("validation.tsv").readlines()]
    for val in tqdm(val_valid):
        caption = validation_tsv[int(val)]
        test_writer.writerow({"caption": caption, "image_id": val})

    # train_valid = []# [l.strip().split(".")[0] for l in open("train_valid.txt").readlines()]
    # for f in os.listdir("features_train_val"):
    #     train_valid.append(f.split(".")[0])
    # t_v_int = [int(v) for v in train_valid]
    # train_tsv = [l.strip().split("\t")[0] for l in open("training.tsv").readlines()]
    # dev_ids = np.random.choice(t_v_int, 1000, replace=False)
    # for image_id in tqdm(train_valid):
    #     caption = train_tsv[int(image_id)]
    #     if int(image_id) in dev_ids:
    #         dev_writer.writerow({"caption": caption, "image_id": image_id})
    #     else:
    #         train_writer.writerow({"caption": caption, "image_id": image_id})

def move_feature_split():
    image_dev = []
    for row in csv.DictReader(open(f"cc_dev.tsv", "r", encoding="utf-8"), delimiter="\t"):
        image_dev.append(row["image_id"])
    image_train = []
    for row in csv.DictReader(open(f"cc_train.tsv", "r", encoding="utf-8"), delimiter="\t"):
        image_train.append(row["image_id"])
    for img in tqdm(image_dev):
        os.rename(f"features_train_val/{img}.npz",
                  f"features_dev/{img}.npz")
    for img in tqdm(image_train):
        os.rename(f"features_train_val/{img}.npz",
                  f"features_train/{img}.npz")


if __name__ == "__main__":
    gen_splits()
    #move_feature_split()
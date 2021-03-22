# Training and Evaluating Multimodal Transformers 

We provide all necessary steps to 1) reproduce our experiments or 2) train and evaluate your own models

## Using our Image-Text Retrieval Training & Evaluation Setup


### Prepare Data
We use [process_data.py](process_data.py) to load MSCOCO, Flickr30k, Conceptual Captions and Multi30k for our experiments.
Depending on which dataset you use, you need to download the necessary data beforehand.

**Pre-Trained Transformer:** Download the pre-trained Transformer from the respective repository 
([OSCAR](https://github.com/microsoft/Oscar), [UNITER/ VILLA](https://github.com/zhegan27/VILLA), [M3P](https://github.com/microsoft/M3P)).
In the configs (see below), set `model.pretrained_model_path` to the destination folder.


**MSCOCO:** Download the data with captions and image features [here from OSCAR](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md).
In the configs (see below), set `data.mscoco_folder` to the destination folder.

**Flickr30k:** Download https://github.com/BryanPlummer/flickr30k_entities. Unzip annotations.zip to get the captions. 
Download the image features from https://drive.google.com/uc?export=download&id=11OD_qq7ITBarJwWZfi0bWIRw3HPEaHwE (source: https://github.com/jnhwkim/ban-vqa/blob/master/tools/download_flickr.sh).
Our code will create .tsv files for the splits for both the captions and the features.
In the configs (see below), 
set `data.image_feature_folder` to the destination folder for the image features, 
`data.flickr30k_original_folder` to the folder for the data repository, and
`data.flickr_split_folder` to the desired output folder for the processed split data.


**Conceptual Captions: (Dev Set only)** 
We use the Conceptual Captions dev set as a test set for different evaluation setups in our publication.

You can download the pre-processed features for easy use [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/datasets/cc_devtest.zip).

Alternatively, if you want to recreate the dataset:
Download the captions with the image URLs [for the Validation split here](https://ai.google.com/research/ConceptualCaptions/download).
Use [download.sh](scripts/cc_scripts/download.sh) to scrap the images and then generate the features yourself
(see [our guide](../../documentation/image_features.md)).
Once you have the features, use [gen_cc_data.py](scripts/cc_scripts/gen_cc_data.py) to generate the .tsv file for [process_data.py](process_data.py).

In the configs (see below), 
set `data.cc_original_folder` to the folder where the generated .tsv is, and
`data.cc_feature_folder` to the root folder for the features (i.e., the features are placed in cc_feature_folder/features_$split).


**Multi30k:** Download https://github.com/multi30k/dataset/tree/master/data. Unzip all files in */raw for the captions.
If you use M3P, you need to generate the image features yourself (see [our guide](../../documentation/image_features.md), 
images available, e.g., [here](https://www.kaggle.com/hsankesara/flickr-image-dataset).).
If you use OSCAR (or similar models), you can use the features downloaded for Flickr30k.
Note that Multi30k and Flickr30k do *not* have the same images in each split.
`data.multi30k_original_folder` to the data folder of the downloaded data,
and `data.flickr_split_folder` to the desired output folder for the processed split data.

### Training

Run `python run_train_test.py $config.yaml` with the config file of your choice to start training (and testing) your model.
We list configs in [train_configs](train_configs) which covers the training for our
embedding model, cross encoder model, and the model jointly trained for both.
In the configs, you only need to set the paths as described above and decide on your dataset.
See [the super_config.yaml](super_config.yaml) for all options in one config for more details.

### Evaluation
The results of the evaluation will be saved in .csv files in the output folder.

Run `python run_train_test.py $config.yaml` with the config file of your choice to start testing your model.
We list configs in [eval_configs](eval_configs) which covers the evaluation for our
embedding model, cross encoder model, and the model jointly trained for both.
In the configs, you only need to set the paths as described above and decide on your dataset.
See [the super_config.yaml](super_config.yaml) for all options in one config for more details.

Run `python run_test_retrieve_rerank.py eval_separate_retrieve_rerank.yaml` to evaluate specifically the retrieve-and-rerank approach with separate 
embedding and cross-encoder model (with a jointly trained model, you can simply use run_train_test.py).
In the configs, you only need to set the paths as described above and decide on your dataset.
See [the super_config.yaml](super_config.yaml) for all options in one config for more details.

## Training Your Models: Using Different Data, Losses or Evaluations

**Initializing Different Models:**
You can easily initialize your own models.
First, initialize the respective Transformer (e.g., OSCAR) with the pre-trained weights.
Then, create additional Pooling and/ or ClassificationHeads which should be placed on top.
You can also create your own modules, if necessary.
See below for an example:
```python
from mmt_retrieval.model.models import OSCAR, UNITER, M3P, ClassificationHead, Pooling
from mmt_retrieval import MultimodalTransformer

pretrained_model = OSCAR/UNITER/M3P(...)
classifier = ClassificationHead(...)
model = MultimodalTransformer(modules=[pretrained_model, classifier])
```

**Different Data, reusing Datasets:** Follow the examples in [process_data.py](process_data.py) to see how to prepare the data
for our `mmt_retrieval.data.datasets.ImageSentenceTripletDataset` and `mmt_retrieval.data.datasets.JointImageSentenceTripletDataset`.
Especially the latter dataset should cover your needs for various multimodal tasks (e.g., GQA, VQA).

**Different Datasets:**
Your dataset has to be compatible with `MultimodalTransformer.smart_batching_collate` (if you use our fit() function).

**Different Losses:** If your required loss is not among the already implemented losses (Triplet Loss, Cross Entropy), you can
easily implement your own loss.
Reference our losses `mmt_retrieval.losses.losses.py` `MultimodalTransformer.smart_batching_collate` to understand the expected data format.

**Different Evaluations:**
Reference our implemented evaluations `mmt_retrieval.evaluation.evaluations.py` and then implement your
own Evaluator by inheriting from `mmt_retrieval.evaluation.evaluations.Evaluator`.

For further questions, feel free to open an issue (preferred) or write me an e-mail.

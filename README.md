# MMT-Retrieval: Image Retrieval and more using Multimodal Transformers (OSCAR, UNITER, M3P & Co)

This project provides an easy way to use the recent pre-trained multimodal Transformers 
like [OSCAR](https://github.com/microsoft/Oscar), [UNITER/ VILLA](https://github.com/zhegan27/VILLA) or [M3P (multilingual!)](https://github.com/microsoft/M3P)
for image search and more.

The code is primarily written for image-text retrieval.
Still, many other Vision+Language tasks, beside image-text retrieval, should work out of the box using our code or require just small changes.

There is currently no unified approach for how the visual input is handled and each model uses their own slightly different approach.
We provide a common interface for all models and support for multiple feature file formats.
This greatly simplifies the process of running the models.

Our project allows you to run a model in a few lines of code and offers easy fine-tuning of your own custom models.

We also provide our fine-tuned image-text-retrieval models for download, so you can get directly started.
Check out [our example for Image Search on MSCOCO using our fine-tuned models here](examples/applications/Image_Search.ipynb).

## Citing & Authors
If you find this repository helpful, feel free to cite our publication [Retrieve Fast, Rerank Smart: Cooperative and Joint Approaches for Improved Cross-Modal Retrieval](ARXIV URL):
```
@article{geigle:2021:arxiv,
  author    = {Gregor Geigle and 
                Jonas Pfeiffer and 
                Nils Reimers and 
                Ivan Vuli\'{c} and 
                Iryna Gurevych},
  title     = {Retrieve Fast, Rerank Smart: Cooperative and Joint Approaches for Improved Cross-Modal Retrieval},
  journal   = {arXiv preprint},
  volume    = {abs/2103.TODO},
  year      = {2021},
  url       = {http://arxiv.org/abs/2103.TODO},
  archivePrefix = {arXiv},
  eprint    = {2103.TODO}
}
```

> **Abstract:** 
> Current state-of-the-art approaches to cross-modal retrieval process text and 
> visual input jointly, relying on Transformer-based architectures with 
> cross-attention mechanisms that attend over all words and objects in an image. 
> While offering unmatched retrieval performance, such models: \textbf{1)} 
> are typically pretrained from scratch and thus less scalable, \textbf{2)} 
> suffer from huge retrieval latency and inefficiency issues, which makes 
> them impractical in realistic applications. To address these crucial gaps 
> towards both improved and efficient cross-modal retrieval, we propose a novel 
> fine-tuning framework which turns any pretrained text-image multi-modal model
> into an efficient retrieval model. The framework is based on a cooperative 
> retrieve-and-rerank approach which combines: \textbf{1)} twin networks to
> separately encode all items of a corpus, enabling efficient initial 
> retrieval, and \textbf{2)} a cross-encoder component for a more nuanced
> (i.e., smarter) ranking of the retrieved small set of items. 
> We also propose to jointly fine-tune the two components with shared weights, 
> yielding a more parameter-efficient model. Our experiments on a series of 
> standard cross-modal retrieval benchmarks in monolingual, multilingual, 
> and zero-shot setups, demonstrate improved accuracy and huge efficiency 
> benefits over the state-of-the-art cross-encoders.


Don't hesitate to send me an e-mail or report an issue, if something is broken or if you have further questions or feedback.



Contact person: Gregor Geigle, [gregor.geigle@gmail.com](mailto:gregor.geigle@gmail.com)

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

>This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Installation
We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher,
**[transformers v4.1.1](https://github.com/huggingface/transformers)** or higher,
and **[sentence-transformer 0.4.1](https://github.com/UKPLab/sentence-transformers)** or higher.


**Install with pip (COMING SOON)**

Install `mmt-retrieval` with `pip`: 
```
pip install mmt-retrieval
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/MMT-Retrieval) and install it directly from the source code:
````
pip install -e .
```` 

**PyTorch with CUDA**
If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.



## Getting Started
With our repository, you can get started using the multimodal Transformers in a few lines of code.
Check out [our example for Image Search on MSCOCO using our fine-tuned models here](examples/applications/Image_Search.ipynb).
Or go along with the following steps to get started with your own project.


### Select the Model
We provide our fine-tuned Image-Text Retrieval models for download.
We also provide links to where to download the pre-trained models and models that are fine-tuned for other tasks.

Alternatively, you can fine-tune your own model, too. See [here](#training) for more.
#### Our Fine-Tuned Image-Text Retrieval Models
We publish our jointly trained fine-tuned models.
They can be used both to encode images and text in a multimodal embedding space 
and to cross-encode pairs for a pairwise similarity.

| Model | URL |
|-------|-----|
| OSCAR (Flickr30k) | https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/models/v1/oscar_join_flickr30k.zip |
| OSCAR (MSCOCO) | https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/models/v1/oscar_join_mscoco.zip |
| M3P (Multi30k - en, de fr, cs) | https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/models/v1/m3p_join_multi30k.zip |



#### Other Pre-Trained or Fine-Tuned Transformer
We currently do not directly support downloading of the different pre-trained Transformer models.
Please manually download them using the links in the respective repositories:
[OSCAR](https://github.com/microsoft/Oscar), [UNITER/ VILLA](https://github.com/zhegan27/VILLA), [M3P](https://github.com/microsoft/M3P).
We present [here](#training) examples on how to initialize your own models with the pre-trained Transformers.

OSCAR provides many already fine-tuned models for different tasks for download (see their MODEL_ZOO.md).
We provide the ability to convert those models to our framework so you can quickly  start using them.
````python
from mmt_retrieval.util import convert_finetuned_oscar

downloaded_folder_path = ".../oscar-base-ir-finetune/checkpoint-29-132780"
converted_model = convert_finetuned_oscar(downloaded_folder_path)
converted_model.save("new_save_location_for_converted_model")
````


### Step 0: Image Feature Pre-Processing
All currently supported models require a pre-processing step
where we extract the regions of interest (which serve as image input analog to tokens for the language input) from the images using a Faster R-CNN object detection model.

Which detection model is needed, depends on the model that you are using.
Check out [our guide](image_features_doc.md) where we have gathered all needed information to get startet.

If available, we also point to already pre-processed image features that can be downloaded for a quicker start.

#### Loading Features and Image Input
We load image features in a dictionary-like object (`model.image_dict`) at the start.
We support various different storage formats for the features (see the guide above).
Each image is uniquely identified by its image id in this dictionary.

The advantage of the dictionary approach is that we can designate the image input by its id which is then internally
resolved to the features.


#### Loading Features Just-In-Time (RAM Constraints)
The image features require a lot of additional memory.
For this reason, we support just-in-time loading of the features from disc.
This requires one feature file for each image. 
Many of the downloadable features are saved in a single file.
We provide code to split those big files in separate files, one for each image.

````python
from mmt_retrieval.util import split_oscar_image_feature_file_to_npz, split_tsv_features_to_npz
````


### Step 1: Getting Started
The following is an example showcasing all steps needed to get started encoding multimodal inputs with our code.

````python
from mmt_retrieval import MultimodalTransformer

# Loading a jointly trained model that can both embed and cross-encode multimodal input
model_path = "https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/models/v1/oscar_join_flickr30k.zip"
model = MultimodalTransformer(model_name_or_path=model_path)

# Image ids are the unique identifier number (as string) of each image. If you save the image features separately for each image, this would be the file name
image_ids = ["0", "1", "5"]
# We must load the image features in some way before we can use the model
# Refer to Step 0 on more details for how to generate the features
feature_folder = "path/to/processed/features"
# Directly load the features from disc. Requires more memory. 
# Increase max_workers for more concurrent threads for faster loading with many features
# Remove select to load the entire folder
model.image_dict.load_features_folder(feature_folder, max_workers=1, select=image_ids)
## OR
# Only load the file paths so that features are loaded later just-in-time when there are required.
# Recommended with restricted memory and/ or a lot of images
# Remove select to load the entire folder
model.image_dict.load_file_names(feature_folder, select=image_ids)

sentences = ["The red brown fox jumped over the fence", "A dog being good"]

# Get Embeddings (as a list of numpy arrays)
sentence_embeddings = model.encode(sentences=sentences, convert_to_numpy=True) # convert_to_numpy=True is default
image_embeddings = model.encode(images=image_ids, convert_to_numpy=True)

# Get Pairwise Similarity Matrix (as a tensor)
similarities = model.encode(sentences=sentences, images=image_ids, output_value="logits", convert_to_tensor=True, cross_product_input=True)
similarities = similarities[:,-1].reshape(len(image_ids), len(sentences))
````


## Experiments and Training
<a name="training"></a>

See [our examples](examples/experiments/README.md) to learn how to fine-tune and evaluate the multimodal Transformers.
We provide instructions for fine-tuning your own models with our image-text retrieval setup, show how to replicate our experiments,
and give pointers on how to train your own models, potentially beyond image-text retrieval.


### Expected Results with our Fine-Tuned Models
We report the JOIN+CO (,i.e., retrieve & re-rank with a jointly trained model) results of our published models
Refer to our publications for more detailed results.

Image Retrieval for MSCOCO/ Flickr30k:

| Model                | Dataset  |      |      |      |
|----------------------|----------|------|------|------|
|                      |           | R@1  | R@5  | R@10 |
| oscar-join-mscoco    |    MSCOCO (5k images) | 54.7 | 81.3 | 88.9 |
| oscar-join-flickr30k | Flickr30k (1k images) | 76.4 | 93.6 | 96.2 |

Multilingual Image Retrieval for Multi30k (in mR):

| Model                | en        | de   | fr   | cs   |
|----------------------|-----------|------|------|------|
| m3p-join-multi30k    |        83.0 | 79.2 | 75.9 |   74 |
from collections import OrderedDict
import random

from sentence_transformers.util import pytorch_cos_sim
from torch.utils.data import Dataset, Sampler
from typing import List, Optional
import torch
import numpy as np
from tqdm import tqdm

class EncodeDataset(Dataset):
    def __init__(self,
                 sentences,
                 images,
                 model,
                 is_tokenized: bool = False,
                 cross_product=False):
        """
        EncodeDataset is used by SentenceTransformer.encode method. It just stores
        the input texts and returns a tokenized version of it.
        """
        self.model = model
        if not sentences:
            self.sentences = None
        elif is_tokenized:
            self.sentences = sentences
            self.len = len(sentences)
        else:
            self.sentences = [self.model.tokenize(sent) for sent in sentences]
            self.len = len(sentences)
        if not images:
            self.images = None
        else:
            self.images = images
            self.len = len(images)
        self.is_tokenized = is_tokenized
        self.cross_product = False
        if sentences and images and cross_product:
            self.cross_product = True
            self.len = len(images) * len(sentences)


    def __getitem__(self, item):
        if self.cross_product:
            return (self.sentences[item%len(self.sentences)], self.images[item//len(self.sentences)])
        else:
            return (self.sentences[item] if self.sentences else None, self.images[item] if self.images else None)


    def __len__(self):
        return self.len

class JointImageSentenceDataset(Dataset):
    def __init__(self, imageids, captions, imageid2captions, tags=None, hard_examples=None, hard_p=None, return_both_neg=False):
        self.imageids = imageids
        self.hard_p = hard_p
        self.captions = captions
        self.imageid2captions = imageid2captions
        self.tags = tags
        if tags:
            self.img2tag = {imageids[i]: tags[i] for i in range(len(imageids))}
        self.hard_examples = hard_examples
        self.caption2image = {key: iid for iid in imageids for key in imageid2captions[iid]}
        self.caption_keys = []
        self.caption_borders = {}
        self.return_both_neg = return_both_neg
        for iid, caps in imageid2captions.items():
            current_len = len(self.caption_keys)
            self.caption_borders.update({c: (current_len, current_len+len(caps)) for c in caps})
            self.caption_keys.extend(list(caps))

    def __getitem__(self, item):
        pos_caption = self.caption_keys[item]
        pos_image = self.caption2image[pos_caption]

        if self.hard_examples is None or (self.hard_p is not None and random.random() > self.hard_p):
            sb, eb = self.caption_borders[pos_caption]
            neg_idx = np.random.choice(list(range(0, sb))+list(range(eb+1, len(self.caption_keys))))
            neg_caption = self.caption_keys[neg_idx]
            neg_image = self.caption2image[neg_caption]
        else:
            hard = self.hard_examples[pos_caption]
            idxs_image = np.random.choice(len(hard["images"]), 1, replace=False, p=hard["distribution_images"])[0]
            idxs_captions = np.random.choice(len(hard["captions"]), 1, replace=False, p=hard["distribution_captions"])[0]
            neg_caption = hard["captions"][idxs_captions]
            neg_image = hard["images"][idxs_image]

        pos_caption = self.captions[pos_caption]
        neg_caption = self.captions[neg_caption]
        if not self.return_both_neg:
            if random.random() <= 0.5:
                neg_caption = pos_caption
            else:
                neg_image = pos_image
            return [[self._merge(pos_caption, pos_image), self._merge(neg_caption, neg_image)]], [torch.LongTensor([1]), torch.LongTensor([0])]
        else:
            return [[self._merge(pos_caption, pos_image), self._merge(pos_caption, neg_image), self._merge(neg_caption, pos_image)]], [torch.LongTensor([1]), torch.LongTensor([0]),  torch.LongTensor([0])]

    def _merge(self, caption, image):
        if self.tags:
            return ([caption, self.img2tag[image]], image)
        else:
            return (caption, image)

    def __len__(self):
        return len(self.caption_keys)

class ImageSentenceTripletDataset(Dataset):
    def __init__(self, imageids, captions, imageid2captions, tags=None, hard_examples=None, negative_examples=0):
        self.hard_examples = hard_examples
        self.negative_examples = negative_examples
        self.imageid2captions = imageid2captions
        self.imageids = imageids
        self.captions = captions
        self.tags = tags
        self.caption2image = {key: iid for iid in imageids for key in imageid2captions[iid]}
        if tags:
            self.img2tag = {imageids[i]: tags[i] for i in range(len(imageids))}
        self.caption_keys = []
        #self.caption_borders = {}
        for iid, caps in imageid2captions.items():
            #current_len = len(self.caption_keys)
            #self.caption_borders.update({c: (current_len, current_len+len(caps)) for c in caps})
            self.caption_keys.extend(list(caps))

    def __getitem__(self, item):
        pos_caption = self.caption_keys[item]
        pos_image = self.caption2image[pos_caption]
        if self.hard_examples is None:
            label = torch.LongTensor([int(pos_image)])#torch.LongTensor([self.caption_borders[pos_caption][0]])
            pos_caption = (self.captions[pos_caption], None)
            tag = self.img2tag[pos_image] if self.tags else None
            pos_image = (tag, pos_image)
            return [[pos_caption]], [[pos_image]], label

        hard = self.hard_examples[pos_caption]
        if self.negative_examples > 0:
            idxs_image = np.random.choice(len(hard["images"]), self.negative_examples, replace=False, p=hard["distribution_images"])
            idxs_captions = np.random.choice(len(hard["captions"]), self.negative_examples, replace=False, p=hard["distribution_captions"])
            hard = {"captions": [hard["captions"][idx] for idx in idxs_captions], "images": [hard["images"][idx] for idx in idxs_image]}
        neg_image = hard["images"]
        neg_caption_labels = [int(self.caption2image[neg_caption]) for neg_caption in hard["captions"]]
        neg_caption = [(self.captions[neg_caption], None) for neg_caption in hard["captions"]]
        pos_caption = (self.captions[pos_caption], None)
        labels = torch.LongTensor([[int(pos_image)]+neg_caption_labels, [int(pos_image)]+[int(i) for i in neg_image]])
        pos_image = (self.img2tag[pos_image] if self.tags else None, pos_image)
        neg_image = [(self.img2tag[i] if self.tags else None, i) for i in neg_image]
        return [[c] for c in [pos_caption]+neg_caption], [[i] for i in [pos_image]+neg_image], labels

    def __len__(self):
        return len(self.caption_keys)


class HardExamples:
    """
    For our implementation of ANCE (Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval)
    https://arxiv.org/abs/2007.00808

    Callback during training to recompute the hard examples
    """
    def __init__(self, model, dataset, batchsize, topk=5, sim_batchsize=0, num_workers = 0):
        self.model = model
        self.dataset = dataset
        self.topk = topk
        self.batchsize = batchsize
        self.sim_batchsize = sim_batchsize
        self.num_workers = num_workers
        if dataset.hard_examples is None:
            dataset.hard_examples = {}

    def __call__(self, *args, **kwargs):
        dataset = self.dataset
        topk = self.topk
        images = dataset.imageids

        captions_ids = list(dataset.captions.keys())
        captions = [dataset.captions[cid] for cid in captions_ids]

        imageid2captions = dataset.imageid2captions
        for imageid in list(imageid2captions.keys()):
            for cid in imageid2captions[imageid]:
                imageid2captions[cid] = {imageid}

        image_embeddings = self.model.encode(sentences=None, images=images, batch_size=self.batchsize, convert_to_tensor=True, num_workers=self.num_workers)
        #Compute embedding for the captions
        caption_embeddings = self.model.encode(sentences=captions, batch_size=self.batchsize, convert_to_tensor=True, num_workers=self.num_workers)

        i2t_result_list = [[] for _ in range(0, len(image_embeddings))]
        t2i_result_list = [[] for _ in range(0, len(caption_embeddings))]

        hardest_t = {}

        #Compute cosine similarites
        if self.sim_batchsize == 0:
            cos_scores = pytorch_cos_sim(image_embeddings, caption_embeddings).cpu()

            # Image to Text
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, topk+5, dim=1, largest=True, sorted=False)
            i2t_cos_scores_top_k_values = cos_scores_top_k_values.tolist()
            i2t_cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

            for query_itr in range(len(i2t_cos_scores_top_k_values)):
                for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                    captions_id = captions_ids[captions_id]
                    i2t_result_list[query_itr].append({'captions_id': captions_id, 'score': score})

        else:
            for start_idx in tqdm(range(0, len(image_embeddings), self.sim_batchsize), desc="Similarity"):
                cos_scores = pytorch_cos_sim(image_embeddings[start_idx:start_idx+self.sim_batchsize], caption_embeddings).cpu()

                # Image to Text
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, topk+5, dim=1, largest=True, sorted=False)
                i2t_cos_scores_top_k_values = cos_scores_top_k_values.tolist()
                i2t_cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

                for query_itr in range(len(i2t_cos_scores_top_k_values)):
                    for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                        captions_id = captions_ids[captions_id]
                        i2t_result_list[start_idx+query_itr].append({'captions_id': captions_id, 'score': score})

        for query_itr in range(len(i2t_result_list)):
            query_id = images[query_itr]
            hardest = []
            #Sort scores
            top_hits = sorted(i2t_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_imageid2captions = imageid2captions[query_id]

            for hit in top_hits:
                if hit['captions_id'] not in query_imageid2captions and len(hardest)<topk:
                    hardest.append(hit)
            hardest_t[query_id] = hardest

        hardest_i = {}

        if self.sim_batchsize == 0:
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, topk+1, dim=0, largest=True, sorted=False)
            t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0).tolist()
            t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0).tolist()

            for query_itr in range(len(t2i_cos_scores_top_k_values)):
                for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                    image_id = images[image_id]
                    t2i_result_list[query_itr].append({'image_id': image_id, 'score': score})

        else:
            for start_idx in tqdm(range(0, len(caption_embeddings), self.sim_batchsize), desc="Similarity"):
                cos_scores = pytorch_cos_sim(image_embeddings, caption_embeddings[start_idx:start_idx+self.sim_batchsize]).cpu()

                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, topk+1, dim=0, largest=True, sorted=False)
                t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0).tolist()
                t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0).tolist()

                for query_itr in range(len(t2i_cos_scores_top_k_values)):
                    for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                        image_id = images[image_id]
                        t2i_result_list[start_idx+query_itr].append({'image_id': image_id, 'score': score})


        for query_itr in range(len(t2i_result_list)):
            query_id = captions_ids[query_itr]
            hardest = []
            #Sort scores
            top_hits = sorted(t2i_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_imageid2captions = imageid2captions[query_id]

            for hit in top_hits:
                if hit['image_id'] not in query_imageid2captions and len(hardest)<topk:
                    hardest.append(hit)
            hardest_i[query_id] = hardest
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        for caption_id in captions_ids:
            image_id = list(imageid2captions[caption_id])[0]
            hard_t, t_scores = zip(*[v.values() for v in hardest_t[image_id]])
            hard_i, i_scores = zip(*[v.values() for v in hardest_i[caption_id]])
            t_dist = softmax(t_scores)
            i_dist = softmax(i_scores)
            dataset.hard_examples[caption_id] = {"images": hard_i, "captions": hard_t, "distribution_images": i_dist, "distribution_captions": t_dist}

class MultidatasetImageSentenceTripletDataset(Dataset):
    def __init__(self, imageids, captions, imageid2captions):
        self.imageids = imageids
        self.captions = captions
        self.caption2image = [{key: iid for iid in imageid for key in rd[iid]} for rd, imageid in zip(imageid2captions, imageids)]
        self.caption_keys = [[] for _ in range(len(captions))]
        self.caption_borders = [{} for _ in range(len(captions))]
        for keys, borders, rd in zip(self.caption_keys, self.caption_borders, imageid2captions):
            for iid, caps in rd.items():
                current_len = len(keys)
                borders.update({c: (current_len, current_len+len(caps)) for c in caps})
                keys.extend(list(caps))

    def __getitem__(self, item):
        dataset_idx = 0
        start = 0
        for i, keys in enumerate(self.caption_keys):
            if item < start+len(keys):
                dataset_idx = i
                item -= start
                break
            else:
                start += len(keys)

        pos_caption = self.caption_keys[dataset_idx][item]
        pos_image = self.caption2image[dataset_idx][pos_caption]
        label = torch.LongTensor([start+self.caption_borders[dataset_idx][pos_caption][0]])
        pos_caption = self.captions[dataset_idx][pos_caption]
        return [[pos_caption]], [[pos_image]], label

        #sb, eb = self.caption_borders[pos_caption]
        #neg_idx = np.random.choice(list(range(0, sb))+list(range(eb+1, len(self.caption_keys))))
        #neg_caption = self.caption_keys[neg_idx]
        #neg_image = self.caption_keys[neg_caption]

        #labels = torch.LongTensor([self.caption_borders[pos_caption][0], self.caption_borders[neg_caption][0]])
        #pos_caption = self.captions[pos_caption]
        #neg_caption = self.captions[neg_caption]
        #return [pos_caption, neg_caption], [pos_image, neg_image], labels

    def __len__(self):
        return sum([len(keys) for keys in self.caption_keys])

class MultidatasetImageSentenceTripletSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.num_datasets = len(dataset.caption_keys)
        self.lens = [len(keys) for keys in dataset.caption_keys]
        self.batch_size = batch_size

    def __iter__(self):
        all_batches = []
        offset = 0
        for length in self.lens:
            idxs = np.random.permutation(range(offset, offset+length))
            batches = [idxs[i:i+self.batch_size] for i in range(0, length, self.batch_size)]
            all_batches.extend(batches)
            offset += length
        batch_idxs = np.random.permutation(range(len(all_batches)))
        for idx in batch_idxs:
            batch = all_batches[idx]
            if len(batch) < self.batch_size:
                continue
            else:
                yield batch

    def __len__(self):
        return sum([length // self.batch_size for length in self.lens])
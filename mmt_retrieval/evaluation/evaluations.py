from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import SentenceEvaluator
import torch
import time
import logging
from tqdm import tqdm
from sentence_transformers.util import pytorch_cos_sim
import os
import csv
import numpy as np
from typing import List, Tuple, Dict, Set

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

class Evaluator(SentenceEvaluator):
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        pass


class ImageTextRetrievalEvaluator(Evaluator):
    """
    This class is the base class for Image-Text Retrieval.
    It measures Mean Reciprocal Rank (MRR), Recall@k, mean Recall (mR) [Average over all R@k for image and text retrieval]
    and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 images: List[str],  #imageids
                 captions: Dict[str, str],  #cid => caption
                 imageid2captions: Dict[str, Set[str]],  #imageid => Set[cid]
                 tags = None,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 recall_at_k: List[int] = [1, 5, 10],
                 map_at_k: List[int] = [10],
                 mR_ks = [1, 5, 10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = ''):

        self.images = images
        self.tags = tags
        self.captions_ids = list(captions.keys())
        self.captions = [captions[cid] for cid in self.captions_ids]
        self.imageid2captions = imageid2captions
        for imageid in list(imageid2captions.keys()):
            for cid in imageid2captions[imageid]:
                self.imageid2captions[cid] = {imageid}

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.recall_at_k = recall_at_k
        self.map_at_k = map_at_k
        self.mR_ks = mR_ks

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        if name:
            name = "_" + name

        self.i2t_csv_file: str = "i2t_retrieval_evaluation" + name + "_results.csv"
        self.t2i_csv_file: str = "t2i_retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for k in recall_at_k:
            self.csv_headers.append("Recall@{}".format(k))

        for k in mrr_at_k:
            self.csv_headers.append("MRR@{}".format(k))

        for k in ndcg_at_k:
            self.csv_headers.append("NDCG@{}".format(k))

        for k in map_at_k:
            self.csv_headers.append("MAP@{}".format(k))
        self.csv_headers.append("mR")

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        raise NotImplementedError("")

    def _eval_i2t(self, i2t_result_list):
        for query_itr in range(len(i2t_result_list)):
            query_id = self.images[query_itr]

            #Sort scores
            if query_id in self.imageid2captions:
                top_hits = sorted(i2t_result_list[query_itr], key=lambda x: x['score'], reverse=True)
                query_imageid2captions = self.imageid2captions[query_id]
            else:
                continue
            if len(query_imageid2captions) == 0:
                continue

            #Recall@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.recall_at_k:
                self.i2t_num_hits_at_k[k_val].append(0)
                for hit in top_hits[0:k_val]:
                    if hit['captions_id'] in query_imageid2captions:
                        self.i2t_num_hits_at_k[k_val][-1] = 1
                        break

            #MRR@k
            for k_val in self.mrr_at_k:
                self.i2t_MRR[k_val].append(0)
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['captions_id'] in query_imageid2captions:
                        self.i2t_MRR[k_val][-1] = 1.0 / (rank + 1)
                        break

            #NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['captions_id'] in query_imageid2captions else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_imageid2captions)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                self.i2t_ndcg[k_val].append(ndcg_value)

            #MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['captions_id'] in query_imageid2captions:
                        num_correct += 1
                        sum_precisions += num_correct / (rank+1)

                avg_precision = sum_precisions / min(k_val, len(query_imageid2captions))
                self.i2t_AveP_at_k[k_val].append(avg_precision)

    def _eval_t2i(self, t2i_result_list):
        for query_itr in range(len(t2i_result_list)):
            query_id = self.captions_ids[query_itr]

            #Sort scores
            if query_id in self.imageid2captions:
                top_hits = sorted(t2i_result_list[query_itr], key=lambda x: x['score'], reverse=True)
                query_imageid2captions = self.imageid2captions[query_id]
            else:
                continue
            if len(query_imageid2captions) == 0:
                continue

            #Recall@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.recall_at_k:
                self.t2i_num_hits_at_k[k_val].append(0)
                for hit in top_hits[0:k_val]:
                    if hit['image_id'] in query_imageid2captions:
                        self.t2i_num_hits_at_k[k_val][-1] = 1
                        break

            #MRR@k
            for k_val in self.mrr_at_k:
                self.t2i_MRR[k_val].append(0)
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['image_id'] in query_imageid2captions:
                        self.t2i_MRR[k_val][-1] = 1.0 / (rank + 1)
                        break

            #NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['image_id'] in query_imageid2captions else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_imageid2captions)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                self.t2i_ndcg[k_val].append(ndcg_value)

            #MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['image_id'] in query_imageid2captions:
                        num_correct += 1
                        sum_precisions += num_correct / (rank+1)

                avg_precision = sum_precisions / min(k_val, len(query_imageid2captions))
                self.t2i_AveP_at_k[k_val].append(avg_precision)

    def _average_results(self):
            #Compute averages
            for k in self.i2t_num_hits_at_k:
                self.i2t_num_hits_at_k[k] = np.mean(self.i2t_num_hits_at_k[k])
    
            for k in self.i2t_ndcg:
                self.i2t_ndcg[k] = np.mean(self.i2t_ndcg[k])
    
            for k in self.i2t_MRR:
                self.i2t_MRR[k] = np.mean(self.i2t_MRR[k])
    
            for k in self.i2t_AveP_at_k:
                self.i2t_AveP_at_k[k] = np.mean(self.i2t_AveP_at_k[k])
    
            for k in self.t2i_num_hits_at_k:
                self.t2i_num_hits_at_k[k] = np.mean(self.t2i_num_hits_at_k[k])
    
            for k in self.t2i_ndcg:
                self.t2i_ndcg[k] = np.mean(self.t2i_ndcg[k])
    
            for k in self.t2i_MRR:
                self.t2i_MRR[k] = np.mean(self.t2i_MRR[k])
    
            for k in self.t2i_AveP_at_k:
                self.t2i_AveP_at_k[k] = np.mean(self.t2i_AveP_at_k[k])

            self.mR = (sum(self.t2i_num_hits_at_k[k] for k in self.mR_ks) + sum(self.i2t_num_hits_at_k[k] for k in self.mR_ks)) / (2*len(self.mR_ks))

    def _log_results(self, delta_time):
            # Output
            logging.info("IMAGE TO TEXT")
            for k in self.i2t_num_hits_at_k:
                logging.info("Recall@{}: {:.2f}%".format(k, self.i2t_num_hits_at_k[k]*100))

            for k in self.i2t_MRR:
                logging.info("MRR@{}: {:.4f}".format(k, self.i2t_MRR[k]))

            for k in self.i2t_ndcg:
                logging.info("NDCG@{}: {:.4f}".format(k, self.i2t_ndcg[k]))

            for k in self.i2t_AveP_at_k:
                logging.info("MAP@{}: {:.4f}".format(k, self.i2t_AveP_at_k[k]))

            logging.info("TEXT TO IMAGE")
            for k in self.t2i_num_hits_at_k:
                logging.info("Recall@{}: {:.2f}%".format(k, self.t2i_num_hits_at_k[k]*100))

            for k in self.t2i_MRR:
                logging.info("MRR@{}: {:.4f}".format(k, self.t2i_MRR[k]))

            for k in self.t2i_ndcg:
                logging.info("NDCG@{}: {:.4f}".format(k, self.t2i_ndcg[k]))

            for k in self.t2i_AveP_at_k:
                logging.info("MAP@{}: {:.4f}".format(k, self.t2i_AveP_at_k[k]))

            logging.info("mR: {:.4f}".format(self.mR))

            logging.info("Images: {}".format(len(self.images)))
            logging.info("Queries: {}\n".format(len(self.captions)))
            logging.info(f"Time: {delta_time:.2f}")

    def _write_output(self, output_path, epoch, steps):
        if output_path is not None:
            csv_path = os.path.join(output_path, self.i2t_csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for k in self.recall_at_k:
                output_data.append(self.i2t_num_hits_at_k[k])

            for k in self.mrr_at_k:
                output_data.append(self.i2t_MRR[k])

            for k in self.ndcg_at_k:
                output_data.append(self.i2t_ndcg[k])

            for k in self.map_at_k:
                output_data.append(self.i2t_AveP_at_k[k])

            output_data.append(self.mR)

            fOut.write(",".join(map(str,output_data)))
            fOut.write("\n")
            fOut.close()

            csv_path = os.path.join(output_path, self.t2i_csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for k in self.recall_at_k:
                output_data.append(self.t2i_num_hits_at_k[k])

            for k in self.mrr_at_k:
                output_data.append(self.t2i_MRR[k])

            for k in self.ndcg_at_k:
                output_data.append(self.t2i_ndcg[k])

            for k in self.map_at_k:
                output_data.append(self.t2i_AveP_at_k[k])

            output_data.append(self.mR)

            fOut.write(",".join(map(str,output_data)))
            fOut.write("\n")
            fOut.close()

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg


class EmbeddingImageTextRetrievalEvaluator(ImageTextRetrievalEvaluator):
    """
    Image-Text Retrieval with Embeddings
    It measures Mean Reciprocal Rank (MRR), Recall@k, mean Recall (mR) [Average over all R@k for image and text retrieval]
    and Normalized Discounted Cumulative Gain (NDCG)
    :param batched_sim: If set to a value >0, the cosine similarity will be computed with this as batch size.
    Useful for very large datasets.
    """

    def __init__(self,
                 images: List[str],  #imageids
                 captions: Dict[str, str],  #cid => caption
                 imageid2captions: Dict[str, Set[str]],  #imageid => Set[cid]
                 tags = None,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10],
                 mR_ks = [1, 5, 10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 batched_sim=0):

        super(EmbeddingImageTextRetrievalEvaluator, self).__init__(images, captions, imageid2captions, tags, mrr_at_k, 
                                                                   ndcg_at_k, recall_at_k, map_at_k, mR_ks, show_progress_bar, batch_size, name)
        self.batched_sim = batched_sim

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Image Text Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        self.i2t_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.i2t_MRR = {k: [] for k in self.mrr_at_k}
        self.i2t_ndcg = {k: [] for k in self.ndcg_at_k}
        self.i2t_AveP_at_k = {k: [] for k in self.map_at_k}

        self.t2i_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.t2i_MRR = {k: [] for k in self.mrr_at_k}
        self.t2i_ndcg = {k: [] for k in self.ndcg_at_k}
        self.t2i_AveP_at_k = {k: [] for k in self.map_at_k}

        self.mR = 0
        
        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.recall_at_k), max(self.map_at_k))

        time_start = time.time()

        image_embeddings = model.encode(sentences=self.tags, images=self.images, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        caption_embeddings = model.encode(sentences=self.captions, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        delta_time = time.time() - time_start

        i2t_result_list = [[] for _ in range(0, len(image_embeddings))]
        t2i_result_list = [[] for _ in range(0, len(caption_embeddings))]

        time_start = time.time()

        if self.batched_sim > 0:
            for start_idx in tqdm(range(0, len(image_embeddings), self.batched_sim), desc="Similarity"):
                cos_scores = pytorch_cos_sim(image_embeddings[start_idx:start_idx+self.batched_sim], caption_embeddings)

                # Image to Text
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0]) - 1), dim=1, largest=True, sorted=False)
                i2t_cos_scores_top_k_values = cos_scores_top_k_values.tolist()
                i2t_cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

                for query_itr in range(len(i2t_cos_scores_top_k_values)):
                    for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                        captions_id = self.captions_ids[captions_id]
                        i2t_result_list[start_idx+query_itr].append({'captions_id': captions_id, 'score': score})
        else:
            #Compute cosine similarites
            cos_scores = pytorch_cos_sim(image_embeddings, caption_embeddings)

            # Image to Text
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0]) - 1), dim=1, largest=True, sorted=False)
            i2t_cos_scores_top_k_values = cos_scores_top_k_values.tolist()
            i2t_cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

            for query_itr in range(len(i2t_cos_scores_top_k_values)):
                for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                    captions_id = self.captions_ids[captions_id]
                    i2t_result_list[query_itr].append({'captions_id': captions_id, 'score': score})

        delta_time = time.time() - time_start + delta_time

        self._eval_i2t(i2t_result_list)

        time_start = time.time()

        if self.batched_sim > 0:
            for start_idx in tqdm(range(0, len(caption_embeddings), self.batched_sim), desc="Similarity"):
                cos_scores = pytorch_cos_sim(image_embeddings, caption_embeddings[start_idx:start_idx+self.batched_sim])

                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[1]) - 1), dim=0, largest=True, sorted=False)
                t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0).tolist()
                t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0).tolist()

                for query_itr in range(len(t2i_cos_scores_top_k_values)):
                    for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                        image_id = self.images[image_id]
                        t2i_result_list[start_idx+query_itr].append({'image_id': image_id, 'score': score})
        else:
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[1]) - 1), dim=0, largest=True, sorted=False)
            t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0).tolist()
            t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0).tolist()

            for query_itr in range(len(t2i_cos_scores_top_k_values)):
                for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                    image_id = self.images[image_id]
                    t2i_result_list[query_itr].append({'image_id': image_id, 'score': score})

        delta_time = time.time() - time_start + delta_time

        self._eval_t2i(t2i_result_list)
    
        self._average_results()
        self._log_results(delta_time)
        self._write_output(output_path, epoch, steps)

        return self.mR

class CrossEncoderImageTextRetrievalEvaluator(ImageTextRetrievalEvaluator):
    """
    Image-Text Retrieval with Cross Encoder
    It measures Mean Reciprocal Rank (MRR), Recall@k, mean Recall (mR) [Average over all R@k for image and text retrieval]
    and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 images: List[str],  #imageids
                 captions: Dict[str, str],  #cid => caption
                 imageid2captions: Dict[str, Set[str]],  #imageid => Set[cid]
                 tags = None,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10],
                 mR_ks = [1, 5, 10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = ''):

        super(CrossEncoderImageTextRetrievalEvaluator, self).__init__(images, captions, imageid2captions, tags, mrr_at_k,
                                                                   ndcg_at_k, recall_at_k, map_at_k, mR_ks, show_progress_bar, batch_size, name)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Image Text Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.recall_at_k), max(self.map_at_k))

        #Init score computation values
        self.i2t_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.i2t_MRR = {k: [] for k in self.mrr_at_k}
        self.i2t_ndcg = {k: [] for k in self.ndcg_at_k}
        self.i2t_AveP_at_k = {k: [] for k in self.map_at_k}

        self.t2i_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.t2i_MRR = {k: [] for k in self.mrr_at_k}
        self.t2i_ndcg = {k: [] for k in self.ndcg_at_k}
        self.t2i_AveP_at_k = {k: [] for k in self.map_at_k}

        self.mR = 0

        i2t_result_list = [[] for _ in range(0, len(self.images))]
        t2i_result_list = [[] for _ in range(0, len(self.captions))]

        time_start = time.time()

        #Compute cosine similarites
        cos_scores = torch.zeros((len(self.images), len(self.captions)))
        for i in range(len(self.images)):
            images = []
            captions = []
            for c in range(len(self.captions)):
                images.append(self.images[i])
                if self.tags:
                    captions.append([self.captions[c], self.tags[i]])
                else:
                    captions.append(self.captions[c])
            sims = model.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True)[:,1]
            cos_scores[i] = sims

        # Image to Text
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(self.captions) - 1), dim=1, largest=True, sorted=False)
        i2t_cos_scores_top_k_values = cos_scores_top_k_values.tolist()
        i2t_cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

        delta_time = time.time() - time_start

        for query_itr in range(len(i2t_cos_scores_top_k_values)):
            for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                captions_id = self.captions_ids[captions_id]
                i2t_result_list[query_itr].append({'captions_id': captions_id, 'score': score})

        self._eval_i2t(i2t_result_list)
        
        time_start = time.time()

        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(self.images) - 1), dim=0, largest=True, sorted=False)
        t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0).tolist()
        t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0).tolist()

        delta_time = time.time() - time_start + delta_time

        for query_itr in range(len(t2i_cos_scores_top_k_values)):
            for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                image_id = self.images[image_id]
                t2i_result_list[query_itr].append({'image_id': image_id, 'score': score})

        self._eval_t2i(t2i_result_list)
        self._average_results()
        self._log_results(delta_time)
        self._write_output(output_path, epoch, steps)
        
        return self.mR


class RetrieveRerankImageTextRetrievalEvaluator(ImageTextRetrievalEvaluator):
    """
    Image-Text Retrieval with Retrieve & Re-Ranking
    It measures Mean Reciprocal Rank (MRR), Recall@k, mean Recall (mR) [Average over all R@k for image and text retrieval]
    and Normalized Discounted Cumulative Gain (NDCG)
    :param retrieve: Decide how many candidates will be retrieved for re-ranking
    :param scoring: Use another re-ranking approach besides purely using the cross-encoder scores
    :param scoring_factor: weight factor between the embedding score and cross encoder score (if scoring is not 'standard')
    """

    def __init__(self,
                 images: List[str],  #imageids
                 captions: Dict[str, str],  #cid => caption
                 imageid2captions: Dict[str, Set[str]],  #imageid => Set[cid]
                 tags = None,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10],
                 mR_ks = [1, 5, 10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 retrieve=10,
                 scoring="standard",
                 scoring_factor=0.5):
        super(RetrieveRerankImageTextRetrievalEvaluator, self).__init__(images, captions, imageid2captions, tags, mrr_at_k,
         ndcg_at_k, recall_at_k, map_at_k, mR_ks, show_progress_bar, batch_size, name)
        self.retrieve = retrieve
        self.scoring = scoring
        self.scoring_factor = scoring_factor
        

    def __call__(self, models, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        
        if isinstance(models, list):
            biencoder, crossencoder = models
        else:
            biencoder = models
            crossencoder = models
        logging.info("Image Text Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        max_k = max(self.retrieve, max(self.mrr_at_k), max(self.ndcg_at_k), max(self.recall_at_k), max(self.map_at_k))

        time_start = time.time()

        # Compute embedding for the images
        image_embeddings = biencoder.encode(sentences=self.tags, images=self.images, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        #Compute embedding for the captions
        caption_embeddings = biencoder.encode(sentences=self.captions, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        #Init score computation values
        self.i2t_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.i2t_MRR = {k: [] for k in self.mrr_at_k}
        self.i2t_ndcg = {k: [] for k in self.ndcg_at_k}
        self.i2t_AveP_at_k = {k: [] for k in self.map_at_k}

        self.t2i_num_hits_at_k = {k: [] for k in self.recall_at_k}
        self.t2i_MRR = {k: [] for k in self.mrr_at_k}
        self.t2i_ndcg = {k: [] for k in self.ndcg_at_k}
        self.t2i_AveP_at_k = {k: [] for k in self.map_at_k}

        self.mR = 0

        i2t_result_list = [[] for _ in range(0, len(image_embeddings))]
        t2i_result_list = [[] for _ in range(0, len(caption_embeddings))]

        #Compute cosine similarites
        cos_scores = pytorch_cos_sim(image_embeddings, caption_embeddings).cpu()

        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, max_k, dim=1, largest=True, sorted=False)
        i2t_cos_scores_top_k_values = cos_scores_top_k_values
        i2t_cos_scores_top_k_idx = cos_scores_top_k_idx

        images = []
        captions = []
        assign = []
        for i in range(len(self.images)):
            if self.images[i] in self.imageid2captions:
                for c, c_idx in enumerate(i2t_cos_scores_top_k_idx[i]):
                    images.append(self.images[i])
                    if self.tags:
                        captions.append([self.captions[c_idx], self.tags[i]])
                    else:
                        captions.append(self.captions[c_idx])
                    assign.append((i,c))

        #mb_sims = i2t_cos_scores_top_k_values.detach().clone()
        if self.scoring == "combination":
            sims = torch.softmax(crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True), 1)[:,1]
            for s, (i,c) in zip(sims, assign):
                emb_score = i2t_cos_scores_top_k_values[i][c]
                i2t_cos_scores_top_k_values[i][c] = (1-self.scoring_factor)*s + self.scoring_factor*emb_score
        elif self.scoring == "combination_normalized":
            sims = torch.softmax(crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True), 1)[:,1]
            sims = sims.reshape(len(self.images), -1)
            for i in range(len(self.images)):
                emb_min_abs = torch.abs(torch.min(i2t_cos_scores_top_k_values[i]))
                emb_score = (i2t_cos_scores_top_k_values[i]-emb_min_abs)/(torch.max(i2t_cos_scores_top_k_values[i])-emb_min_abs)
                i2t_cos_scores_top_k_values[i] = (1-self.scoring_factor)*(sims[i]-torch.min(sims[i]))/(torch.max(sims[i])-torch.min(sims[i])) + \
                                                 self.scoring_factor*emb_score
        elif self.scoring == "rank":
            sims = crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True)[:,1]
            sims = sims.reshape(len(self.images), -1)
            for i in range(len(self.images)):
                emb_rank = torch.argsort(i2t_cos_scores_top_k_values[i]).float()
                sim_rank = torch.argsort(sims[i]).float()
                i2t_cos_scores_top_k_values[i] = (1-self.scoring_factor)*emb_rank + self.scoring_factor*sim_rank
        else:
            sims = crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True)[:,1]
            for s, (i,c) in zip(sims, assign):
                i2t_cos_scores_top_k_values[i][c] = s

        delta_time = time.time() - time_start

        for query_itr in range(len(i2t_cos_scores_top_k_values)):
            for captions_id, score in zip(i2t_cos_scores_top_k_idx[query_itr], i2t_cos_scores_top_k_values[query_itr]):
                captions_id = self.captions_ids[captions_id]
                i2t_result_list[query_itr].append({'captions_id': captions_id, 'score': score})

        self._eval_i2t(i2t_result_list)

        time_start = time.time()

        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, max_k, dim=0, largest=True, sorted=False)
        t2i_cos_scores_top_k_values = cos_scores_top_k_values.permute(1, 0)
        t2i_cos_scores_top_k_idx = cos_scores_top_k_idx.permute(1, 0)

        images = []
        captions = []
        assign = []
        for c in range(len(self.captions)):
            if self.captions_ids[c] in self.imageid2captions:
                for i, i_idx in enumerate(t2i_cos_scores_top_k_idx[c]):
                    images.append(self.images[i_idx])
                    if self.tags:
                        captions.append([self.captions[c], self.tags[i_idx]])
                    else:
                        captions.append(self.captions[c])
                    assign.append((c,i))
        if self.scoring == "combination":
            sims = torch.softmax(crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True), 1)[:,1]
            for s, (i,c) in zip(sims, assign):
                emb_score = t2i_cos_scores_top_k_values[i][c]
                t2i_cos_scores_top_k_values[i][c] = (1-self.scoring_factor)*s + self.scoring_factor*emb_score
        elif self.scoring == "combination_normalized":
            sims = torch.softmax(crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True), 1)[:,1]
            sims = sims.reshape(len(self.captions), -1)
            for i in range(len(self.images)):
                emb_min_abs = torch.abs(torch.min(t2i_cos_scores_top_k_values[i]))
                emb_score = (t2i_cos_scores_top_k_values[i]-emb_min_abs)/(torch.max(t2i_cos_scores_top_k_values[i])-emb_min_abs)
                sims_score = (sims[i]-torch.min(sims[i]))/(torch.max(sims[i])-torch.min(sims[i]))
                t2i_cos_scores_top_k_values[i] = (1-self.scoring_factor)*sims_score + self.scoring_factor*emb_score
        elif self.scoring == "rank":
            sims = crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True)[:,1]
            sims = sims.reshape(len(self.captions), -1)
            for i in range(len(self.images)):
                emb_rank = torch.argsort(t2i_cos_scores_top_k_values[i]).float()
                sim_rank = torch.argsort(sims[i]).float()
                t2i_cos_scores_top_k_values[i] = (1-self.scoring_factor)*emb_rank + self.scoring_factor*sim_rank
        else:
            sims = crossencoder.encode(captions, images, batch_size=self.batch_size, output_value="logits", show_progress_bar=True, convert_to_tensor=True)[:,1]
            for s, (i,c) in zip(sims, assign):
                t2i_cos_scores_top_k_values[i][c] = s

        delta_time = time.time() - time_start + delta_time

        for query_itr in range(len(t2i_cos_scores_top_k_values)):
            for image_id, score in zip(t2i_cos_scores_top_k_idx[query_itr], t2i_cos_scores_top_k_values[query_itr]):
                image_id = self.images[image_id]
                t2i_result_list[query_itr].append({'image_id': image_id, 'score': score})

        self._eval_t2i(t2i_result_list)


        self._average_results()
        self._log_results(delta_time)
        self._write_output(output_path, epoch, steps)

        return self.mR


class ImageQuestionClassification(Evaluator):
    """
    Evaluation for image question datasets like VQA, GQA
    """
    def __init__(self,
                 images,  #imageids
                 questions,  #cid => caption
                 labels,
                 batch_size: int = 32,
                 name: str = ''):

        self.batch_size = batch_size
        self.name = name
        self.questions = questions
        self.labels = labels
        self.images = images
        if name:
            name = "_" + name

        self.file: str = "class" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "acc"]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Accuracy on " + self.name + " dataset" + out_txt)

        logits = model.encode(self.questions, self.images, batch_size=self.batch_size, show_progress_bar=True, output_value="logits")

        acc = (self.labels.squeeze() == torch.argmax(logits, dim=1)).squeeze().long().cpu().tolist()

        acc = np.mean(acc)
        logging.info("Questions: {}\n".format(len(self.questions)))
        logging.info(f"Accuracy: {acc:.2f}")
        if output_path is not None:
            csv_path = os.path.join(output_path, self.file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps, acc]

            fOut.write(",".join(map(str,output_data)))
            fOut.write("\n")
            fOut.close()
        return acc

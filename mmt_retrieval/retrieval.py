from sentence_transformers import LoggingHandler
import torch
import logging
from tqdm import tqdm
from sentence_transformers.util import pytorch_cos_sim
import os
from typing import List, Tuple, Dict, Set, Optional

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

class ImageTextRetrieval:
    def __init__(self,
                 embedding_file: Optional[str]=None,
                 images: Optional[List[str]]=None,
                 texts: Optional[List[str]]=None,
                 embedder=None,
                 cross_encoder=None,
                 embedding_batchsize=512,
                 cross_encoder_batchsize=512):
        self.cross_encoder = cross_encoder
        self.embedder = embedder
        assert cross_encoder is not None and embedder is not None
        #self.device = cross_encoder.device() if cross_encoder is not None else embedder.device()
        self.texts = texts
        self.images = images
        self.text_embeddings = None
        self.image_embeddings = None
        self.embedding_batchsize = embedding_batchsize
        self.cross_encoder_batchsize = cross_encoder_batchsize

        if embedding_file:
            embeddings = torch.load(embedding_file)
            self.texts = embeddings.get("texts", None)
            self.images = embeddings.get("images", None)
            self.text_embeddings = embeddings.get("text_embeddings", None)
            self.image_embeddings = embeddings.get("image_embeddings", None)

    def search(self, text_queries=None, image_queries=None, topk=10, store_embeddings_path=None, show_progress_bar=False):
        self.create_embeddings(store_embeddings_path)
        image_results, text_results = None, None
        if text_queries:
            image_results = self._search(text_queries, "image", topk, show_progress_bar)
        if image_queries:
            text_results = self._search(image_queries, "text", topk, show_progress_bar)

        if text_queries and image_queries:
            return image_results, text_results
        elif image_queries:
            return text_results
        elif text_queries:
            return image_results

    def create_embeddings(self, store_embeddings_path=None, show_progress_bar=False):
        if self.embedder:
            created_embeddings = False
            if self.texts and self.text_embeddings is None:
                logging.info("Creating embeddings for the texts")
                self.text_embeddings = self.embedder.encode(sentences=self.texts,
                                                                  show_progress_bar=show_progress_bar,
                                                                  batch_size=self.embedding_batchsize, convert_to_tensor=True)
                created_embeddings = True
            if self.images and self.image_embeddings is None:
                logging.info("Creating embeddings for the images")
                self.image_embeddings = self.embedder.encode(images=self.images,
                                                                   show_progress_bar=show_progress_bar,
                                                                   batch_size=self.embedding_batchsize, convert_to_tensor=True)
                created_embeddings = True
            if created_embeddings:
                results = {"texts": self.texts, "images": self.images, "text_embeddings": self.text_embeddings, "image_embeddings": self.image_embeddings}
                if store_embeddings_path:
                    path, file = os.path.split(store_embeddings_path)
                    os.makedirs(path, exist_ok=True)
                    torch.save(results, store_embeddings_path)
                return results


    def _search(self, queries, target_modality, topk=10, show_progress_bar=False):
        if not isinstance(queries, list):
            queries = [queries]
        # initialize ranking with all images if we do not reduce with embeddings
        if target_modality == "image":
            rankings = [range(len(self.images)) for _ in range(len(queries))]
        else:
            rankings = [range(len(self.texts)) for _ in range(len(queries))]
        scores = [[] for _ in range(len(queries))]

        # exact cosine-based nearest neighbor retrieval
        if self.embedder:
            target_embeddings = self.image_embeddings if target_modality == "image" else self.text_embeddings
            if target_modality == "image":
                query_embeddings = self.embedder.encode(sentences=queries,
                                                        show_progress_bar=show_progress_bar,
                                                        batch_size=self.embedding_batchsize, convert_to_tensor=True)
            else:
                query_embeddings = self.embedder.encode(images=queries,
                                                        show_progress_bar=show_progress_bar,
                                                        batch_size=self.embedding_batchsize, convert_to_tensor=True)
            cos_scores = pytorch_cos_sim(query_embeddings, target_embeddings)
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, topk, dim=1, largest=True, sorted=True)
            rankings = cos_scores_top_k_idx.tolist()
            scores = cos_scores_top_k_values.tolist()
        # pairwise similarity of all query-image pairs. If we use embeddings, then only with the candidate images
        if self.cross_encoder:
            for q in tqdm(range(len(queries)), desc="Queries"):
                images = []
                captions = []
                for d in rankings[q]:
                    if target_modality == "image":
                        images.append(self.images[d])
                        captions.append(queries[q])
                    else:
                        images.append(queries[q])
                        captions.append(self.texts[d])
                sims = self.cross_encoder.encode(captions, images, batch_size=self.cross_encoder_batchsize,
                                                 output_value="logits", show_progress_bar=show_progress_bar, convert_to_tensor=True)[:,-1]
                values, idxs = torch.sort(sims)
                scores[q] = values.tolist()
                rankings[q] = list(rankings[q]) #if ranking is still a range generator, fix that
                rankings[q] = [rankings[q][i] for i in idxs]

        result = {
            "indices": rankings,
            "scores": scores
        }
        if target_modality == "image":
            result["images"] = [[self.images[i] for i in ranks] for ranks in rankings]
        else:
            result["texts"] = [[self.texts[i] for i in ranks] for ranks in rankings]
        return result
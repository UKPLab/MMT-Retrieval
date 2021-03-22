import json
import logging
import os
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from numpy import ndarray
import transformers
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

try:
    import wandb
except ImportError:
    pass

from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import __version__
from sentence_transformers import SentenceTransformer

from mmt_retrieval.data.image_dict import ImageDict
from mmt_retrieval.evaluation.evaluations import Evaluator
from mmt_retrieval.data.datasets import EncodeDataset

class MultimodalTransformer(SentenceTransformer):
    """
    Loads or create a MultimodalTransformer model, that can be used to encode multimodal data.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path.
    If it is not a path, it first tries to download a pre-trained MultimodalTransformer model.
    If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    """
    def __init__(self, model_name_or_path=None, modules: Iterable[nn.Module] = None, device: str = None):
        super().__init__(model_name_or_path, modules, device)
        self.image_dict = ImageDict()
        self.use_wandb = False

    def encode(self, sentences: Union[str, List[str], List[int]] = None,
               images: Union[str, List[str]] = None,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'multimodal_embedding',
               cross_product_input: bool = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               is_pretokenized: bool = False,
               device: str = None,
               num_workers: int = 0) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Encode multimodal input
        :param sentences: the sentences
        :param images: imageids
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encoding
        :param output_value:  Default 'multimodal_embedding', to get multimodal embeddings.
        Can be set to 'token_embeddings' to get embeddings for each input or set it to 'logits' to get the logits from the classification head.
        :param cross_product_input: If true, will evaluate the cross-product of the sentences and images.
        Output ordering will be decided by 'for each image for each sentence', i.e. each image with all sentences first.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param is_pretokenized: If is_pretokenized=True, sentences must be a list of integers, containing the tokenized sentences with each token convert to the respective int.
        :param device: Which torch.device to use for the computation
        :param num_workers: Number of background-workers to prepare data. Set to positive number to increase speed
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        if device is None:
            device = self._target_device
        self.to(device)
        all_embeddings = []
        if images and not isinstance(images, list):
            images = [images]
        if sentences and not isinstance(sentences, list):
            sentences = [sentences]
        inp_dataset = EncodeDataset(sentences, images, self, is_tokenized=is_pretokenized, cross_product=cross_product_input)
        inp_dataloader = DataLoader(inp_dataset, batch_size=batch_size, collate_fn=self.smart_batching_collate_eval, num_workers=num_workers, shuffle=False)
        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        for features in iterator:
            for feature_name in features:
                features[feature_name] = features[feature_name].to(device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    #input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    #embeddings = embeddings * input_mask_expanded
                    embeddings = [emb[:i] for emb, i in zip(embeddings, input_mask.sum(dim=1))]
                all_embeddings.extend(embeddings)

        if convert_to_tensor:
            try:
                all_embeddings = torch.stack(all_embeddings)
            except RuntimeError:
                return all_embeddings
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.cpu().detach().numpy() for emb in all_embeddings])

        return all_embeddings


    def get_max_seq_length(self):
        """
        Returns the maximal sequence length for input the model accepts. Longer inputs will be truncated
        """
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, text: str):
        """
        Tokenizes the text
        """
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_image_features(self, *features):
        return self._first_module().get_image_features(*features)

    def get_features(self, *features):
        return self._first_module().get_features(*features)

    # from sentence-transformers
    def get_sentence_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def get_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path, optimizer=None, scheduler=None):
        """
        Saves all elements for this seq. model into different sub-folders
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        if optimizer:
            torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        if scheduler:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path),
                                      'type': type(module).__module__ + '.' + type(module).__qualname__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        labels = []
        num_blocks = len(batch[0]) - 1
        num_pairs = [len(batch[0][i]) for i in range(num_blocks)]
        paired_blocks = [[[] for _ in range(num_pairs[i])] for i in range(num_blocks)]
        max_seq_lens = [[[0, 0] for _ in range(num_pairs[i])] for i in range(num_blocks)]
        for item in batch:
            blocks, label = item[:-1], item[-1]
            for i, block in enumerate(blocks):
                for j, pair in enumerate(block):
                    for p in pair:
                        tokens = None
                        image_features = None
                        if p[0] is not None:
                            tokens = self.tokenize(p[0])
                            max_seq_lens[i][j][0] = max(max_seq_lens[i][j][0], self._text_length(tokens))
                        if p[1] is not None:
                            image_features = self.image_dict[p[1]]
                            max_seq_lens[i][j][1] = max(max_seq_lens[i][j][1], len(image_features['features']))
                        paired_blocks[i][j].append((tokens, image_features))
            if isinstance(label, list):
                labels.extend(label)
            else:
                labels.append(label)

        features = []
        for i in range(num_blocks):
            for j in range(num_pairs[i]):
                feature_lists = {}
                for pair in paired_blocks[i][j]:
                    pair_features = self.get_features(pair[0], pair[1], max_seq_lens[i][j][0], max_seq_lens[i][j][1])

                    for feature_name in pair_features:
                        if feature_name not in feature_lists:
                            feature_lists[feature_name] = []

                        feature_lists[feature_name].append(pair_features[feature_name])
                for feature_name in feature_lists:
                    feature_lists[feature_name] = torch.cat(feature_lists[feature_name])
                features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}

    def smart_batching_collate_eval(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model.
        Here, batch is a list of texts
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        feature_lists = {}
        if batch[0][1] is not None:
            batch = [[pair[0], self.image_dict[pair[1]]] for pair in batch]

        max_text_len = max([self._text_length(pair[0]) for pair in batch]) if batch[0][0] is not None else 0
        max_img_len = max([len(pair[1]['features']) for pair in batch]) if batch[0][1] is not None else 0

        for pair in batch:
            pair_features = self.get_features(pair[0], pair[1], max_text_len, max_img_len)
            for feature_name in pair_features:
                if feature_name not in feature_lists:
                    feature_lists[feature_name] = []

                feature_lists[feature_name].append(pair_features[feature_name])

        for feature_name in feature_lists:
            feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

        return feature_lists

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            optimizer_model,
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            steps_per_epoch = None,
            gradient_accumulation = 1,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            load_optimizer_scheduler_path: Optional[str] = None,
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            logging_steps: int = 100,
            use_wandb = False,
            output_path: str = None,
            output_path_ignore_not_empty: bool = False,
            save_best_model: bool = True,
            save_latest_model: bool = False,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            dataset_callback = None,
            objectives_sequence = None
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param load_optimizer_scheduler_path: If set, will load 'optimizer.pt' and 'scheduler.pt' from this directory
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param output_path_ignore_not_empty: By default, training will stop if output_path is not empty. If set to true, this error will be ignored and training proceeds.
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param save_latest_model: If true, save the latest model with optimizer and scheduler in ouput_path/latest_checkpoint.
        Combine with load_optimizer_scheduler_path to continue training after a stop.
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        self.use_wandb = use_wandb
        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if not output_path_ignore_not_empty and len(os.listdir(output_path)) > 0:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))
            if save_latest_model:
                os.makedirs(os.path.join(output_path, "latest_checkpoint"), exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self._target_device

        for loss_model in loss_models:
            loss_model.to(device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        param_optimizer = list(optimizer_model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps*num_train_steps)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
        self._load_optimizer_scheduler(load_optimizer_scheduler_path, optimizer, scheduler)
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)
        if objectives_sequence is None:
            objectives_sequence = range(num_train_objectives)
        logging_loss_scaling = [len([j for j in objectives_sequence if j==i]) for i in range(num_train_objectives)]
        skip_scheduler = False

        start_epoch = 0
        start_step = 0
        if load_optimizer_scheduler_path:
            done_steps = scheduler._step_count + 1
            start_epoch = done_steps // steps_per_epoch
            start_step = done_steps % steps_per_epoch
            global_step = done_steps
        tr_losses = [torch.tensor(0.0).to(self._target_device) for _ in range(len(loss_models))]
        logging_losses_scalar = [0.0]*len(loss_models)
        for epoch in trange(start_epoch, epochs, desc="Epoch"):
            training_steps = start_step

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(start_step, steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in objectives_sequence:
                    loss_model = loss_models[train_idx]
                    data_iterator = data_iterators[train_idx]
                    tr_loss = tr_losses[train_idx]
                    for acc_idx in range(gradient_accumulation):
                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            #logging.info("Restart data_iterator")
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features = data['features']
                        for paired_sentence_idx in range(len(features)):
                            for feature_name in features[paired_sentence_idx]:
                                features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(self._target_device)

                        labels = data['labels'].to(self._target_device)

                        if use_amp:
                            with autocast():
                                loss_value = loss_model(features, labels)
                            scale_before_step = scaler.get_scale()
                            scaler.scale(loss_value).backward()
                        else:
                            loss_value = loss_model(features, labels)
                            loss_value.backward()

                        if steps_per_epoch==training_steps+acc_idx:
                            break
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    tr_loss += loss_value.detach()
                    optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    logs: Dict[str, float] = {}
                    for loss_idx, (tr_loss, logging_loss_scalar, scaling) in enumerate(zip(tr_losses, logging_losses_scalar, logging_loss_scaling)):
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"+str(loss_idx)] = (tr_loss_scalar - logging_loss_scalar) / (logging_steps*scaling)
                        logging_losses_scalar[loss_idx] = tr_loss_scalar
                    # backward compatibility for pytorch schedulers
                    logs["learning_rate"] = scheduler.get_last_lr()[0]
                    self._log(logs, global_step)

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                               training_steps, global_step, callback)
                    if save_latest_model:
                        self.save(os.path.join(output_path, "latest_checkpoint"), optimizer, scheduler)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            start_step = 0
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, global_step, callback)
            if save_latest_model:
                self.save(os.path.join(output_path, "latest_checkpoint"), optimizer, scheduler)
            if dataset_callback is not None:
                dataset_callback()

    def evaluate(self, evaluator: Evaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _log(self, logs, steps):
        if self.use_wandb:
            wandb.log(logs, step=steps)
        logging.info(logs)

    def _load_optimizer_scheduler(self, path, optimizer, scheduler):
        if path is not None:
            optimizer.load_state_dict(
                torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, global_step, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            self._log({"main_eval_score": score}, global_step)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)


    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    @property
    def device(self) -> device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def tokenizer(self):
        """
        Property to get the tokenizer that is used by this model
        """
        return self._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Property to set the tokenizer that is should used by this model
        """
        self._first_module().tokenizer = value

    @property
    def max_seq_length(self):
        """
        Property to get the maximal text input sequence length for the model. Longer inputs will be truncated.
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        """
        Property to set the maximal text input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value

    @property
    def max_image_seq_length(self):
        """
        Property to get the maximal input image sequence length for the model. Longer inputs will be truncated.
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_image_seq_length(self, value):
        """
        Property to set the maximal input image sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value
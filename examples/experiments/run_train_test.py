import random
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import transformers
import yaml
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import SequentialEvaluator
import logging
from datetime import datetime
import os
import argparse
import wandb
import numpy as np
from mmt_retrieval.data.datasets import HardExamples
from mmt_retrieval.evaluation.evaluations import EmbeddingImageTextRetrievalEvaluator, CrossEncoderImageTextRetrievalEvaluator, \
    RetrieveRerankImageTextRetrievalEvaluator
from mmt_retrieval.mmt import MultimodalTransformer
from mmt_retrieval.model.models import OSCAR, M3P, DoubleOSCAR, UNITER, JointOSCAR, ClassificationHead, Pooling
from process_data import get_sampler, DATA_LOADER
from mmt_retrieval.losses.losses import BatchHardTripletLoss, NTXentLossTriplet, SampledHardTripletLoss, CrossEntropyLoss

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def run(config):
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
    model_config = config["model"]
    model_folder_name = f"{model_config['name']}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    model_save_path = os.path.join(config["model_path"], model_folder_name)
    os.makedirs(model_save_path, exist_ok=True)

    if config.get("use_wandb", False):
        wandb_config = config["wandb"]
        run = wandb.init(config=config, project=wandb_config.get("wandb_project", "mmt-retrieval"), name=wandb_config.get("wandb_name", model_folder_name), reinit=True)
    logging.info(config)
    if config.get("do_train", True):
        logging.info("### Training ###")
        logging.info("Creating a new model in ", model_save_path)

        model = build_model(model_config)

        if config.get("use_wandb", False):
            wandb.watch(model)

        train_config = config["training"]
        training_tasks = []
        dataset_callback = None
        for task in train_config["tasks"]:
            dataset = DATA_LOADER[task["name"]](get_data(config["data"], task["name"]), "train", task.get("tiny", False), model.image_dict,
                                                joint=task["loss"]["name"] == "joint", **task.get("data_args", {}))
            sampler, batch_sampler, shuffle, batch_size = get_sampler(task["name"], dataset,
                                                                      task.get("batchsize", train_config["batchsize"]), **task.get("data_args", {}))
            dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, sampler=sampler, batch_sampler=batch_sampler, num_workers=train_config.get("num_workers", 0))
            if task["loss"]["name"] == "triplet":
                loss = BatchHardTripletLoss(model=model, margin=task["loss"].get("margin", 0.2))
            elif task["loss"]["name"] == "ance":
                loss = SampledHardTripletLoss(model=model, margin=task["loss"].get("margin", 0.2))
                dataset_callback = add_callback(model, dataset, train_config["dev"]["batchsize"],
                                                task.get("data_args", {}).get("topk", 32),
                                                task.get("data_args", {}).get("sim_batchsize", 0),
                                                train_config.get("num_workers", 0), dataset_callback)
            elif task["loss"]["name"] == "ntxent":
                loss = NTXentLossTriplet(model=model, scale=task["loss"].get("scale", 20.0))
            elif task["loss"]["name"] == "joint":
                loss = CrossEntropyLoss(model)
                if task.get("do_hard", False) and not task.get("data_args", {}).get("hard", False):
                    dataset_callback = add_callback(model, dataset, train_config["dev"]["batchsize"],
                                                         task.get("data_args", {}).get("topk", 32),
                                                         task.get("data_args", {}).get("sim_batchsize", 0),
                                                         train_config.get("num_workers", 0), dataset_callback)
            training_tasks.append((dataloader, loss))

        dev_config = train_config["dev"]
        dev_evaluator = get_evaluator(config["data"], dev_config, model, "dev")
        optimizer_class = transformers.AdamW
        optimizer_params={"lr": train_config.get("lr", 2e-5), "eps": train_config.get("eps", 1e-6)}
        if dataset_callback is not None:
            dataset_callback()
        model.fit(train_objectives=training_tasks,
                  optimizer_model=model,
                  evaluator=dev_evaluator,
                  epochs=train_config.get("epochs", 1),
                  steps_per_epoch=train_config.get("steps_per_epoch", None),
                  gradient_accumulation=train_config.get("gradient_accumulation", 1),
                  evaluation_steps=train_config.get("evaluation_steps", 1000),
                  warmup_steps=train_config.get("warmup_steps", 0.1),
                  output_path=model_save_path,
                  use_amp=train_config.get("use_amp", True),
                  optimizer_params=optimizer_params,
                  optimizer_class=optimizer_class,
                  load_optimizer_scheduler_path=train_config.get("load_optimizer_scheduler_path", None),
                  save_latest_model=train_config.get("save_latest_model", False),
                  weight_decay=train_config.get("weight_decay", 0.01),
                  max_grad_norm=train_config.get("max_grad_norm", 1),
                  use_wandb=config.get("use_wandb", False),
                  logging_steps=config.get("wandb", {}).get("logging_steps", 0),
                  dataset_callback=dataset_callback,
                  objectives_sequence=train_config.get("task_sequence", None)
                  )

    if config.get("do_dev_test", False):
        logging.info("### Testing on Dev Set ###")
        dev_config = config["test"]
        if "model_path" in model_config or config.get("do_train", True):
            logging.info("Loading model from ", model_config.get("model_path", model_save_path))
            model = MultimodalTransformer(model_name_or_path=model_config.get("model_path", model_save_path))
        else:
            logging.info("Creating a new model in ", model_save_path)
            model = build_model(model_config)
        if config.get("use_wandb", False):
            wandb.watch(model)
        dev_evaluator = get_evaluator(config["data"], dev_config, model, "dev")
        dev_evaluator(model, output_path=model_save_path)
        
    if config.get("do_test", True):
        logging.info("### Testing ###")
        test_config = config["test"]
        if "model_path" in model_config or config.get("do_train", True):
            logging.info("Loading model from ", model_config.get("model_path", model_save_path))
            model = MultimodalTransformer(model_name_or_path=model_config.get("model_path", model_save_path))
        else:
            logging.info("Creating a new model in ", model_save_path)
            model = build_model(model_config)
        if config.get("use_wandb", False):
            wandb.watch(model)
        test_evaluator = get_evaluator(config["data"], test_config, model, "test")
        test_evaluator(model, output_path=model_save_path)

    with open(os.path.join(model_save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if config.get("use_wandb", False):
        wandb.save(os.path.join(model_save_path, "*.csv"))
        wandb.join()
        run.finish()


def add_callback(model, dataset, batchsize, topk, sim_batchsize, num_workers, callback=None, ):
    if callback is None:
        callback = HardExamples(model, dataset, batchsize, topk, sim_batchsize, num_workers)
    else:
        dataset.hard_examples = callback.dataset.hard_examples
    return callback

def build_model(model_config):
    if "legacy" not in model_config["name"]:
        if "oscar" in model_config["name"]:
            embedding_model = OSCAR(model_config["pretrained_model_path"],
                                    max_seq_length=model_config.get("max_seq_length", 70),
                                    max_image_seq_len=model_config.get("max_image_seq_len", 50))
        elif "m3p" in model_config["name"]:
            embedding_model = M3P(model_config["pretrained_model_path"],
                                  max_seq_length=model_config.get("max_seq_length", 128),
                                  max_image_seq_len=model_config.get("max_image_seq_len", 50))
        elif "villa" in model_config["name"] or "uniter" in model_config["name"]:
            embedding_model = UNITER(model_config["pretrained_model_path"],
                                     max_seq_length=model_config.get("max_seq_length", 70),
                                     max_image_seq_len=model_config.get("max_image_seq_len", 50))
        if model_config.get("half_layers", False):
            module_list = torch.nn.ModuleList()
            for i, layer in enumerate(embedding_model.auto_model.encoder.layer):
                if i % 2 == 0:
                    module_list.append(layer)
            embedding_model.auto_model.encoder.layer = module_list

        class_head = ClassificationHead(2, model_config.get("input_key", "pooled_cls_token_embeddings"), 768,
                                        model_config.get("classifier_type", "linear"),
                                        model_config.get("scaling_factor", 2))
        pooling_model = Pooling(768,
                                       pooling_mode_mean_tokens=model_config.get("mean", True),
                                       pooling_mode_cls_token=model_config.get("cls", False),
                                       pooling_mode_max_tokens=model_config.get("max", False))
        return MultimodalTransformer(modules=[embedding_model, class_head, pooling_model])

    # Legacy models
    elif "joint" in model_config["name"]:
        if model_config["name"] == "joint_oscar":
            model = JointOSCAR(model_config["pretrained_model_path"],
                               max_seq_length=model_config.get("max_seq_length", 70),
                               max_image_seq_len=model_config.get("max_image_seq_len", 50))
        if model_config.get("half_layers", False):
            module_list = torch.nn.ModuleList()
            for i, layer in enumerate(model.auto_model.bert.encoder.layer):
                if i % 2 == 0:
                    module_list.append(layer)
            model.auto_model.bert.encoder.layer = module_list
        return MultimodalTransformer(modules=[model])
    else:
        if model_config["name"] == "oscar":
            embedding_model = OSCAR(model_config["pretrained_model_path"],
                                    max_seq_length=model_config.get("max_seq_length", 70),
                                    max_image_seq_len=model_config.get("max_image_seq_len", 50),
                                    token_dropout_prob=model_config.get("token_dropout_prob", 0.0))
        elif model_config["name"] == "double_oscar":
            embedding_model = DoubleOSCAR(model_config["pretrained_model_path"],
                                          max_seq_length=model_config.get("max_seq_length", 70),
                                          max_image_seq_len=model_config.get("max_image_seq_len", 50),
                                          token_dropout_prob=model_config.get("token_dropout_prob", 0.0))
        elif model_config["name"] == "m3p":
            embedding_model = M3P(model_config["pretrained_model_path"],
                                  max_seq_length=model_config.get("max_seq_length", 128),
                                  max_image_seq_len=model_config.get("max_image_seq_len", 50))
        elif model_config["name"] == "villa" or model_config["name"] == "uniter":
            embedding_model = UNITER(model_config["pretrained_model_path"],
                                    max_seq_length=model_config.get("max_seq_length", 70),
                                    max_image_seq_len=model_config.get("max_image_seq_len", 50))
        if model_config.get("half_layers", False):
            module_list = torch.nn.ModuleList()
            for i, layer in enumerate(embedding_model.auto_model.encoder.layer):
                if i % 2 == 0:
                    module_list.append(layer)
            embedding_model.auto_model.encoder.layer = module_list
        pooling_model = Pooling(768,
                                       pooling_mode_mean_tokens=model_config.get("mean", True),
                                       pooling_mode_cls_token=model_config.get("cls", False),
                                       pooling_mode_max_tokens=model_config.get("max", False))
        return MultimodalTransformer(modules=[embedding_model, pooling_model])

def get_data(data_config, name):
    """
    for backwards compatibility of old configs
    :param data_config:
    :param name:
    :return:
    """
    if "all" in data_config:
        return data_config["all"]
    elif name in data_config:
        return data_config[name]
    else:
        return data_config

def get_evaluator(data_config, config, model, split):
    split_evaluators = []
    for task in config["tasks"]:
        load_split = task.get("overwrite_split", split)
        split_dataset = DATA_LOADER[task["name"]](get_data(data_config, task["name"]), load_split, task.get("tiny", False),
                                                  model.image_dict, return_dict=True, **task.get("data_args", {}))
        split_eval_name = task["evaluator"]["name"]
        file_name = f"{split}-{task['name']}-{split_eval_name}"
        if split_eval_name == "embedding_itr" or split_eval_name == "itr":
            evaluator = EmbeddingImageTextRetrievalEvaluator(split_dataset["images"], split_dataset["captions"], split_dataset["imageid2captions"], split_dataset.get("tags", None),
                                                             name=file_name, show_progress_bar=True,
                                                             batched_sim=task.get("batched_sim", 0))
        elif split_eval_name == "ce_itr" or split_eval_name == "joint_itr":
            evaluator = CrossEncoderImageTextRetrievalEvaluator(split_dataset["images"], split_dataset["captions"], split_dataset["imageid2captions"], split_dataset.get("tags", None),
                                                                name=file_name, batch_size=config["batchsize"], show_progress_bar=True)
        elif split_eval_name == "rr_itr":
            evaluator = RetrieveRerankImageTextRetrievalEvaluator(split_dataset["images"], split_dataset["captions"], split_dataset["imageid2captions"], split_dataset.get("tags", None),
                                                                  name=file_name, batch_size=config["batchsize"], show_progress_bar=True,
                                                                  retrieve=task.get("retrieve", 10), scoring=task.get("scoring", "standard"),
                                                                  scoring_factor=task.get("scoring_factor", 0.5))
        split_evaluators.append(evaluator)
    if config.get("main_score_function", "normal") == "mean":
        msf = lambda scores: np.mean(scores)
    else:
        msf = lambda scores: scores[-1]
    return SequentialEvaluator(split_evaluators, main_score_function=msf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    run(config)
import random
import torch
import yaml
from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import SequentialEvaluator
import logging
from datetime import datetime
import os
import argparse
import wandb
import numpy as np
from mmt_retrieval.evaluation.evaluations import RetrieveRerankImageTextRetrievalEvaluator
from mmt_retrieval.mmt import MultimodalTransformer
from process_data import DATA_LOADER

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
        run = wandb.init(config=config, project=wandb_config.get("wandb_project", "mmemb"), name=wandb_config.get("wandb_name", model_folder_name), reinit=True)
    logging.info(config)

    if config.get("do_dev_test", False):
        logging.info("### Testing on Dev ###")
        logging.info("Loading model from ", model_config.get("model_path_embedding", model_save_path))
        emb_model = MultimodalTransformer(model_name_or_path=model_config.get("model_path_embedding", model_save_path))
        logging.info("Loading model from ", model_config.get("model_path_cross", model_save_path))
        cross_model = MultimodalTransformer(model_name_or_path=model_config.get("model_path_cross", model_save_path))
        cross_model.image_dict = emb_model.image_dict
        dev_config = config["test"]
        dev_evaluator = get_evaluator(config["data"], dev_config, emb_model, "dev")
        dev_evaluator([emb_model, cross_model], output_path=model_save_path)

    if config.get("do_test", True):
        logging.info("### Testing ###")
        test_config = config["test"]
        logging.info("Loading model from ", model_config.get("model_path_embedding", model_save_path))
        emb_model = MultimodalTransformer(model_name_or_path=model_config.get("model_path_embedding", model_save_path))
        logging.info("Loading model from ", model_config.get("model_path_cross", model_save_path))
        cross_model = MultimodalTransformer(model_name_or_path=model_config.get("model_path_cross", model_save_path))
        cross_model.image_dict = emb_model.image_dict
        test_evaluator = get_evaluator(config["data"], test_config, emb_model, "test")
        test_evaluator([emb_model, cross_model], output_path=model_save_path)

    with open(os.path.join(model_save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if config.get("use_wandb", False):
        wandb.save(os.path.join(model_save_path, "*.csv"))
        wandb.join()
        run.finish()


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
        split_dataset = DATA_LOADER[task["name"]](get_data(data_config, task["name"]), split, task.get("tiny", False),
                                                  model.image_dict, **task.get("data_args", {}))
        split_eval_name = task["evaluator"]["name"]
        file_name = f"{split}-{task['name']}-{split_eval_name}"
        evaluator = RetrieveRerankImageTextRetrievalEvaluator(split_dataset["images"], split_dataset["captions"], split_dataset["imageid2captions"], split_dataset.get("tags", None),
                                                         name=file_name, batch_size=config["batchsize"], show_progress_bar=True,
                                                                  retrieve=task.get("retrieve", 10),
                                                                  scoring=task.get("scoring", "standard"), scoring_factor=task.get("scoring_factor", 0.5))
        split_evaluators.append(evaluator)
    return SequentialEvaluator(split_evaluators)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    run(config)
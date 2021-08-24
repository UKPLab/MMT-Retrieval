import random
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import transformers
import yaml
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
import logging
from datetime import datetime
import os
import argparse
import wandb
import numpy as np
from mmt_retrieval.evaluation.evaluations import ImageQuestionClassification
from mmt_retrieval.mmt import MultimodalTransformer
from mmt_retrieval.model.models import OSCAR, M3P, UNITER, ClassificationHead
from mmt_retrieval.losses.losses import CrossEntropyLoss

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

### Implement yourself. See ../experiments/process_data.py for an example.
### Take care to load the image features in the image_dict using the provided methods. See documentation/image_features.md for more
def load_data(split, image_dict):
    return None

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
        run = wandb.init(config=config, project=wandb_config.get("wandb_project", "mmt-classification"), name=wandb_config.get("wandb_name", model_folder_name), reinit=True)
    logging.info(config)
    if config.get("do_train", True):
        logging.info("### Training ###")
        logging.info("Creating a new model in ", model_save_path)

        model = build_model(model_config)

        if config.get("use_wandb", False):
            wandb.watch(model)

        train_config = config["training"]
        training_tasks = []
        dataset = load_data("train", model.image_dict)
        dataloader = DataLoader(dataset, batch_size=train_config["batchsize"], num_workers=train_config.get("num_workers", 0))

        loss = CrossEntropyLoss(model)
        training_tasks.append((dataloader, loss))

        dev_config = train_config["dev"]
        split_dataset = load_data("dev", model.image_dict) # Implement yourself
        file_name = f"dev-acc"
        dev_evaluator = ImageQuestionClassification(split_dataset["images"], split_dataset["captions"], split_dataset["labels"],
                                                name=file_name, batch_size=dev_config["batchsize"])
        optimizer_class = transformers.AdamW
        optimizer_params={"lr": train_config.get("lr", 2e-5), "eps": train_config.get("eps", 1e-6)}
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
        split_dataset = load_data("dev", model.image_dict) # Implement yourself
        file_name = f"dev-acc"
        evaluator = ImageQuestionClassification(split_dataset["images"], split_dataset["captions"], split_dataset["labels"],
                                                name=file_name, batch_size=dev_config["batchsize"])
        evaluator(model, output_path=model_save_path)
        
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

        split_dataset = load_data("test", model.image_dict) # Implement yourself
        file_name = f"test-acc"
        evaluator = ImageQuestionClassification(split_dataset["images"], split_dataset["captions"], split_dataset["labels"],
                                            name=file_name, batch_size=test_config["batchsize"])
        evaluator(model, output_path=model_save_path)

    with open(os.path.join(model_save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if config.get("use_wandb", False):
        wandb.save(os.path.join(model_save_path, "*.csv"))
        wandb.join()
        run.finish()



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

        class_head = ClassificationHead(model_config.get("num_labels", 2),
                                        model_config.get("input_key", "pooled_cls_token_embeddings"), 768,
                                        model_config.get("classifier_type", "linear"),
                                        model_config.get("scaling_factor", 2))
        return MultimodalTransformer(modules=[embedding_model, class_head])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    run(config)
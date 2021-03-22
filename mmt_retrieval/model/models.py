import logging
from collections import namedtuple
import torch
from torch import nn
from torch import Tensor
from mmt_retrieval.multimodal_transformers.m3p.model import TransformerModel
from mmt_retrieval.multimodal_transformers.oscar.modeling.modeling_bert import BertImgModel, ImageBertForSequenceClassification
from mmt_retrieval.multimodal_transformers.villa.model import UniterModel
from transformers import AutoTokenizer, BertConfig, BertTokenizer
import json
from typing import List, Dict, Optional, Union
import os
import numpy as np

class ClassificationHead(nn.Module):
    """
    A classification head on top a transformer.
    The logits will be saved under the 'logits' key in the features.
    
    :param num_labels: The number of labels for the classifier
    :param input_key: The input key for the features that is used for the classifier.
    Either 'cls_token_embeddings' for the CLS token output
    or 'pooled_cls_token_embeddings' for the further processed output
    :param input_dim: Input dimension. The hidden size of the model
    :param classifier_type: 'linear' for a simple 1-layer classifier, 'mlp' for two layers with ReLU activation inbetween
    :param scaling_factor: For classifier_type='mlp', scaling factor applied to input_dim for the first linear layer
    :param dropout: Dropout rate. Applied before each linear layer.
    """
    def __init__(self, num_labels: int, input_key: str="cls_token_embeddings", input_dim: int=768,
                 classifier_type: str="linear", scaling_factor: int=2, dropout: int=0.1):
        super(ClassificationHead, self).__init__()
        self.config_keys = ["num_labels", "input_dim", "classifier_type", "scaling_factor", "input_key", "dropout"]
        self.num_labels = num_labels
        self.input_key = input_key
        self.input_dim = input_dim
        self.classifier_type = classifier_type
        self.scaling_factor = scaling_factor
        self.dropout = dropout
        self._dropout = nn.Dropout(dropout)
        if classifier_type == 'linear':
            self.classifier = nn.Linear(input_dim, self.num_labels)
        elif classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim * scaling_factor),
                nn.ReLU(),
                nn.Linear(input_dim * scaling_factor, self.num_labels)
            )

    def forward(self, features):
        input = features[self.input_key]
        input = self._dropout(input)
        logits = self.classifier(input)

        features.update({"logits": logits})

        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, "class_head.pt"))
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        model = ClassificationHead(**config)
        state_dict = torch.load(os.path.join(input_path, "class_head.pt"))
        model.load_state_dict(state_dict)
        return model


class Pooling(nn.Module):
    """
    This class is identical to sentence_transformers.model.Pooling except we change the key in the features from
    'sentence_embedding' to 'multimodal_embedding' (with the option of changing this back for sentence-transformer compatibility)

    Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized input a fixed sized multimodal embedding. This layer also allows
    to use the CLS token if it is returned by the underlying embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    :param pooling_key: Key name for the pooled embedding in the returned features. Default is 'multimodal_embedding';
    sentence-transformers expects 'sentence_embedding' for its losses etc.
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_key: str = "multimodal_embedding"
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens', 'pooling_key']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_key = pooling_key

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({self.pooling_key: output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)


class OSCAR(nn.Module):
    """
    Wrapper around pre-trained OSCAR.
    Handles correct processing of image features and text and passes it to the model.
    
    :param model_path: Path to pre-trained OSCAR model folder
    :param max_seq_length: Maximum text sequence length. Longer sequences are truncates
    :param max_image_seq_len: Maximum number of image regions. Longer sequences are truncated
    :param model_args: Overwrite config values, e.g., to increase hidden_dropout_prob
    :param tokenizer_args: Overwrite tokenizer config values
    :param do_lower_case: Overwrite tokenizer config
    :param cache_dir: Change the cache directory for huggingface, only for tokenizer
    """
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(OSCAR, self).__init__()
        self.config_keys = ['max_seq_length', 'max_image_seq_len']
        self.max_seq_length = max_seq_length
        self.max_image_seq_len = max_image_seq_len
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        config = BertConfig.from_pretrained(model_path, **model_args, cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=cache_dir, **tokenizer_args)
        if not hasattr(config, 'img_feature_dim') or not hasattr(config, 'img_feature_type'):
            config.img_feature_dim = 2054
            config.img_feature_type = "faster_r-cnn"
        self.auto_model = BertImgModel.from_pretrained(model_path, config=config)

    def forward(self, features):
        output_states = self.auto_model(**features)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens,
                         'pooled_cls_token_embeddings': output_states[1], 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        pad_seq_length = min(pad_seq_length, self.max_seq_length, self.auto_model.config.max_position_embeddings-2) + 2 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens), is_split_into_words=True, max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")
        else:
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens[0]), self.tokenizer.convert_ids_to_tokens(tokens[1]), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, self.auto_model.config.max_position_embeddings-1)
        vision_features = image_features['features'][:pad_seq_length]

        boxes = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])
        image_height, image_width = image_features['img_h'], image_features['img_w']
        # https://github.com/microsoft/Oscar/issues/33
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)
        full_features = np.concatenate((vision_features, spatial_features), axis=1)
        attention = [1]*full_features.shape[0]
        full_features = np.concatenate((full_features, np.zeros((pad_seq_length - len(full_features), full_features.shape[1]))))
        attention = attention+[0]*(pad_seq_length - len(attention))
        full_features = torch.FloatTensor(full_features)
        attention = torch.LongTensor(attention)
        features = {} #self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        features["img_feats"] = full_features.unsqueeze(dim=0)
        features["attention_mask"] = attention.unsqueeze(dim=0) #torch.cat((features["attention_mask"], attention.unsqueeze(dim=0)), dim=1)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return OSCAR(model_path=input_path, **config)

# Legacy Code
class JointOSCAR(nn.Module):
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(JointOSCAR, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length
        self.config_keys = ['max_image_seq_len']
        self.max_image_seq_len = max_image_seq_len
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        config = BertConfig.from_pretrained(model_path, **model_args, cache_dir=cache_dir)
        if not hasattr(config, "loss_type"):
            config.loss_type = "sfmx"
        self.auto_model = ImageBertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=cache_dir, **tokenizer_args)
        self.image_dict = {}

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        outputs = self.auto_model(**features)
        logits = outputs[0]
        features.update({"logits": logits, "cross_similarity": logits[:, 1]})
        return features

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, self.auto_model.config.max_position_embeddings-2) + 2 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")
        else:
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens[0]), self.tokenizer.convert_ids_to_tokens(tokens[1]), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, self.auto_model.config.max_position_embeddings-1)
        vision_features = image_features['features'][:pad_seq_length]

        boxes = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])
        image_height, image_width = image_features['img_h'], image_features['img_w']
        # https://github.com/microsoft/Oscar/issues/33
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)
        full_features = np.concatenate((vision_features, spatial_features), axis=1)
        attention = [1]*full_features.shape[0]
        full_features = np.concatenate((full_features, np.zeros((pad_seq_length - len(full_features), full_features.shape[1]))))
        attention = attention+[0]*(pad_seq_length - len(attention))
        full_features = torch.FloatTensor(full_features)
        attention = torch.LongTensor(attention)
        features = {} #self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        features["img_feats"] = full_features.unsqueeze(dim=0)
        features["attention_mask"] = attention.unsqueeze(dim=0) #torch.cat((features["attention_mask"], attention.unsqueeze(dim=0)), dim=1)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return JointOSCAR(model_path=input_path, **config)

# Legacy Code
class DoubleOSCAR(nn.Module):
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 model_args: Dict = {}, cache_dir: Optional[str] = None, token_dropout_prob = 0.0,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(DoubleOSCAR, self).__init__()
        self.config_keys = ['max_seq_length', 'max_image_seq_len', "token_dropout_prob"]
        self.max_seq_length = max_seq_length
        self.max_image_seq_len = max_image_seq_len
        self.token_dropout_prob = token_dropout_prob
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        #try:
        #    config = AutoConfig.from_pretrained(model_path, **model_args, cache_dir=cache_dir)
        #    self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, **tokenizer_args)
        #except ValueError:
        config = BertConfig.from_pretrained(model_path, **model_args, cache_dir=cache_dir)
        config.hidden_dropout_prob = token_dropout_prob
        self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=cache_dir, **tokenizer_args)
        if not hasattr(config, 'img_feature_dim') or not hasattr(config, 'img_feature_type'):
            config.img_feature_dim = 2054
            config.img_feature_type = "faster_r-cnn"
        self.auto_model = BertImgModel.from_pretrained(model_path, config=config, cache_dir=cache_dir)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        if os.path.exists(os.path.join(model_path, "class_head.pt")):
            logging.info("Loading Classification Head from {}".format(os.path.join(model_path, "class_head.pt")))
            ch_state_dict = torch.load(os.path.join(model_path, "class_head.pt"))
            self.classifier.load_state_dict(ch_state_dict)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.auto_model(**features)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token

        pooled_output = output_states[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        features.update({"logits": logits, "cross_similarity": logits[:, 1], 'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, self.auto_model.config.max_position_embeddings-2) + 2 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")
        else:
            return self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens[0]), self.tokenizer.convert_ids_to_tokens(tokens[1]), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt")

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, self.auto_model.config.max_position_embeddings-1)
        vision_features = image_features['features'][:pad_seq_length]

        boxes = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])
        image_height, image_width = image_features['img_h'], image_features['img_w']
        # https://github.com/microsoft/Oscar/issues/33
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)
        full_features = np.concatenate((vision_features, spatial_features), axis=1)
        attention = [1]*full_features.shape[0]
        full_features = np.concatenate((full_features, np.zeros((pad_seq_length - len(full_features), full_features.shape[1]))))
        attention = attention+[0]*(pad_seq_length - len(attention))
        full_features = torch.FloatTensor(full_features)
        attention = torch.LongTensor(attention)
        features = {} #self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        features["img_feats"] = full_features.unsqueeze(dim=0)
        features["attention_mask"] = attention.unsqueeze(dim=0) #torch.cat((features["attention_mask"], attention.unsqueeze(dim=0)), dim=1)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        torch.save(self.classifier.state_dict(), os.path.join(output_path, "class_head.pt"))
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return DoubleOSCAR(model_path=input_path, **config)


class M3P(nn.Module):
    """
    Wrapper around pre-trained M3P.
    Handles correct processing of image features and text and passes it to the model.
    
    :param model_path: Path to pre-trained M3P model folder
    :param max_seq_length: Maximum text sequence length. Longer sequences are truncates
    :param max_image_seq_len: Maximum number of image regions. Longer sequences are truncated
    :param model_args: Overwrite model config values
    :param tokenizer_args: Overwrite tokenizer config values
    :param do_lower_case: Overwrite tokenizer config
    :param cache_dir: Change the cache directory for huggingface, only for tokenizer
    """
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 cache_dir: Optional[str] = None, model_args={},
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(M3P, self).__init__()
        self.config_keys = ['max_seq_length', 'max_image_seq_len']
        self.max_seq_length = max_seq_length
        self.max_image_seq_len = max_image_seq_len
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir=cache_dir)
        params = {"emb_dim": 768, "n_layers": 12, "n_dec_layers": -1, "n_heads": 12, "dropout": 0.1, "attention_dropout": 0.1,
                  "gelu_activation": True, "share_inout_emb": True, "sinusoidal_embeddings": False, "attention_setting": "v1",
                  "refine_layers": 6, "use_externel_att": False, "eos_index" : self.tokenizer.eos_token_id,
                  "pad_index": self.tokenizer.pad_token_id, "mask_index": self.tokenizer.mask_token_id,
                  "n_words": self.tokenizer.vocab_size, "n_langs": 1, "id2lang": {0:"en"}, "lang2id": {"en":0}}
        params.update(model_args)
        params = namedtuple("Params", params.keys())(*params.values())
        self.auto_model = TransformerModel(params, is_encoder=True, with_output=False, is_crossModal=True)
        state_dict = torch.load(os.path.join(model_path, "m3p_under_weights.pth"), map_location=lambda storage, location: storage)["model"]
        if all([k.startswith('module.') for k in state_dict.keys()]):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        self.auto_model.load_state_dict(state_dict, strict=False)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        alen = torch.arange(features["x"].size()[1], dtype=torch.long, device=features["x"].device)
        attention_mask = (alen < features["lengths"][:, None]).long()
        if "x_img" in features:
            alen = torch.arange(features["x_img"].size()[1], dtype=torch.long, device=features["x"].device)
            attention_mask = torch.cat((attention_mask, (alen < features["lengths_img"][:, None]).long()), dim=1)
        features["x"] = features["x"].transpose(0,1)
        if "x_img" in features:
            features["x_img"] = features["x_img"].transpose(0,1)
            features["image_loc"] = features["image_loc"].transpose(0,1)
            output_tokens = self.auto_model("jointfwd", **features)
        else:
            features["stream_"] = "text"
            features["causal"] = False
            output_tokens = self.auto_model("crossfwd", **features)
        output_tokens = output_tokens.transpose(0, 1)
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        pooled = self.auto_model.pooled_layer(output_tokens)
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'pooled_cls_token_embeddings': pooled, 'attention_mask': attention_mask})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.dim

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = {"x": torch.LongTensor([self.tokenizer.cls_token_id]).unsqueeze(dim=0), "lengths": torch.LongTensor([1])}
            #features = self.get_sentence_features([], text_pad_seq_length)
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, 512) + 2 #Add space for special tokens

        complete_tokens = self.tokenizer.build_inputs_with_special_tokens(tokens)[:pad_seq_length]
        length = len(complete_tokens)
        complete_tokens = complete_tokens + [0] * (pad_seq_length - len(complete_tokens))
        features = {"x": torch.LongTensor(complete_tokens).unsqueeze(dim=0), "lengths": torch.LongTensor([length])}
        return features

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, 512)
        vision_features = image_features['features'][:pad_seq_length]

        boxes_orig = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])
        image_height, image_width = image_features['img_h'], image_features['img_w']

        image_height, image_width = image_features['img_h'], image_features['img_w']
        box_width = boxes_orig[:, 2] - boxes_orig[:, 0]
        box_height = boxes_orig[:, 3] - boxes_orig[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes_orig[:, 0] / image_width
        scaled_y = boxes_orig[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width*scaled_height), axis=1)

        vision_features = np.concatenate(
            (vision_features, np.zeros((pad_seq_length - len(vision_features), vision_features[0].shape[0])))).copy()
        spatial_features = np.concatenate(
            (spatial_features, np.zeros((pad_seq_length - len(spatial_features), 5)))).copy()

        features = dict() #self.get_sentence_features([], 0)
        features["image_loc"] = torch.FloatTensor(spatial_features).unsqueeze(dim=0)
        features["lengths_img"] = torch.LongTensor([len(boxes_orig)])
        features["x_img"] = torch.FloatTensor(vision_features).unsqueeze(dim=0)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        torch.save({"model" :self.auto_model.state_dict()}, os.path.join(output_path, "m3p_under_weights.pth"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return M3P(model_path=input_path, **config)

# Legacy Code
class JointM3P(nn.Module):
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 model_args: Dict = {}, cache_dir: Optional[str] = None, num_labels=2,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(JointM3P, self).__init__()
        self.config_keys = ['max_seq_length', 'max_image_seq_len', "num_labels", "model_args"]
        self.max_seq_length = max_seq_length
        self.max_image_seq_len = max_image_seq_len
        self.num_labels = num_labels
        self.model_args = model_args
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        params = {"emb_dim": 768, "n_layers": 12, "n_dec_layers": -1, "n_heads": 12, "dropout": 0.1, "attention_dropout": 0.1,
                  "gelu_activation": True, "share_inout_emb": True, "sinusoidal_embeddings": False, "attention_setting": "v1",
                  "refine_layers": 6, "use_externel_att": False, "eos_index" : self.tokenizer.eos_token_id,
                  "pad_index": self.tokenizer.pad_token_id, "mask_index": self.tokenizer.mask_token_id,
                  "n_words": self.tokenizer.vocab_size, "n_langs": 1, "id2lang": {0:"en"}, "lang2id": {"en":0}}

        params = namedtuple("Params", params.keys())(*params.values())
        self.auto_model = TransformerModel(params, is_encoder=True, with_output=False, is_crossModal=True)
        state_dict = torch.load(os.path.join(model_path, "m3p_under_weights.pth"), map_location=lambda storage, location: storage)["model"]
        if all([k.startswith('module.') for k in state_dict.keys()]):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        self.auto_model.load_state_dict(state_dict, strict=False)

        if 'classifier' in model_args:
            if 'cls_hidden_scale' not in model_args:
                model_args["cls_hidden_scale"] = 2

            if model_args["classifier"] == 'linear':
                self.classifier = nn.Linear(768,
                                            self.num_labels)
            elif model_args["classifier"] == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(768, 768 * model_args["cls_hidden_scale"]),
                    nn.ReLU(),
                    nn.Linear(768 * model_args["cls_hidden_scale"], self.num_labels)
                )
        else:
            self.classifier = nn.Linear(768, self.num_labels)  # original

        if os.path.exists(os.path.join(model_path, "class_head.pt")):
            logging.info("Loading Classification Head from {}".format(os.path.join(model_path, "class_head.pt")))
            ch_state_dict = torch.load(os.path.join(model_path, "class_head.pt"))
            self.classifier.load_state_dict(ch_state_dict)


    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        alen = torch.arange(features["x"].size()[1], dtype=torch.long, device=features["x"].device)
        attention_mask = (alen < features["lengths"][:, None]).long()
        if "x_img" in features:
            alen = torch.arange(features["x_img"].size()[1], dtype=torch.long, device=features["x"].device)
            attention_mask = torch.cat((attention_mask, (alen < features["lengths_img"][:, None]).long()), dim=1)
        features["x"] = features["x"].transpose(0,1) # dont ask why they do this. they immediatly transpose it back anyway...
        if "x_img" in features:
            features["x_img"] = features["x_img"].transpose(0,1)
            features["image_loc"] = features["image_loc"].transpose(0,1)
            output_tokens = self.auto_model("jointfwd", **features)
        else:
            features["stream_"] = "text"
            features["causal"] = False
            output_tokens = self.auto_model("crossfwd", **features)
        output_tokens = output_tokens.transpose(0, 1) #whyyy
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': attention_mask})

        logits = self.classifier(cls_tokens)

        features.update({"logits": logits, 'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        return features

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = self.tokenizer.encode_plus([self.tokenizer.cls_token], add_special_tokens=False, return_tensors='pt')
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, 512) + 2 #Add space for special tokens

        complete_tokens = self.tokenizer.build_inputs_with_special_tokens(tokens)[:pad_seq_length]
        length = len(complete_tokens)
        complete_tokens = complete_tokens + [0] * (pad_seq_length - len(complete_tokens))
        features = {"x": torch.LongTensor(complete_tokens).unsqueeze(dim=0), "lengths": torch.LongTensor([length])}
        return features

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, 512)
        vision_features = image_features['features'][:pad_seq_length]

        boxes_orig = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])

        image_height, image_width = image_features['img_h'], image_features['img_w']
        box_width = boxes_orig[:, 2] - boxes_orig[:, 0]
        box_height = boxes_orig[:, 3] - boxes_orig[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes_orig[:, 0] / image_width
        scaled_y = boxes_orig[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width*scaled_height), axis=1)

        vision_features = np.concatenate(
            (vision_features, np.zeros((pad_seq_length - len(vision_features), vision_features[0].shape[0])))).copy()
        spatial_features = np.concatenate(
            (spatial_features, np.zeros((pad_seq_length - len(spatial_features), 5)))).copy()

        features = dict() #self.get_sentence_features([], 0)
        features["image_loc"] = torch.FloatTensor(spatial_features).unsqueeze(dim=0)
        features["lengths_img"] = torch.LongTensor([len(boxes_orig)])
        features["x_img"] = torch.FloatTensor(vision_features).unsqueeze(dim=0)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        torch.save({"model" :self.auto_model.state_dict()}, os.path.join(output_path, "m3p_under_weights.pth"))
        torch.save(self.classifier.state_dict(), os.path.join(output_path, "class_head.pt"))
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return JointM3P(model_path=input_path, **config)


class UNITER(nn.Module):
    """
    Wrapper around pre-trained UNITER or VILLA.
    Handles correct processing of image features and text and passes it to the model.

    :param model_path: Path to pre-trained UNITER/VILLA model file
    :param max_seq_length: Maximum text sequence length. Longer sequences are truncates
    :param max_image_seq_len: Maximum number of image regions. Longer sequences are truncated
    :param config_file: Overwrite default config file 'uniter-base.json' in model_path folder with different config.
    :param tokenizer_args: Overwrite tokenizer config values
    :param do_lower_case: Overwrite tokenizer config
    :param cache_dir: Change the cache directory for huggingface, only for tokenizer
    """
    def __init__(self, model_path: str, max_seq_length: int = 128, max_image_seq_len=50,
                 config_file=None, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(UNITER, self).__init__()
        self.config_keys = ['max_seq_length', 'max_image_seq_len', "config_file"]
        self.max_seq_length = max_seq_length
        self.max_image_seq_len = max_image_seq_len
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        self.config_file = os.path.join(os.path.split(model_path)[0], "uniter-base.json") if config_file is None else config_file
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir=cache_dir, **tokenizer_args)
        self.auto_model = UniterModel.from_pretrained(state_dict=torch.load(model_path), config_file=self.config_file, img_dim=2048)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.auto_model(output_all_encoded_layers=False, **features)
        output_tokens = output_states

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens,
                         'pooled_cls_token_embeddings': output_states[1], 'attention_mask': features['attention_mask']})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_features(self, tokens, image_features, text_pad_seq_length, img_pad_seq_length, image_dict=None):
        if tokens:
            features = self.get_sentence_features(tokens, text_pad_seq_length)
        else:
            features = self.get_sentence_features([self.tokenizer.cls_token], pad_seq_length=-1, add_special_tokens=False)
        if image_features:
            img_features = self.get_image_features(image_features, image_dict, img_pad_seq_length)
            for feature in img_features:
                if feature in features:
                    features[feature] = torch.cat((features[feature], img_features[feature]), dim=1)
                else:
                    features[feature] = img_features[feature]
            #UNITER/VILLA keeps images and text after embedding compact, but we instead keep each padded. it does not actually matter
            # since we adapt the attention mask accordingly (their is like (1, 1, 1, 1, 0, 0) ours is (1, 1, 1, 0, 1, 0)).
            gather_index = torch.arange(0, features["attention_mask"].shape[1], dtype=torch.long,).unsqueeze(0)
            features["gather_index"] = gather_index
        return features

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int, add_special_tokens=True):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, self.auto_model.config.max_position_embeddings-2) + 2 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            features = self.tokenizer.encode_plus(self.tokenizer.convert_ids_to_tokens(tokens), max_length=pad_seq_length, pad_to_max_length=True, return_tensors="pt", add_special_tokens=add_special_tokens)
            features.pop("token_type_ids")
            features["position_ids"] = torch.arange(0, features["input_ids"].size(1), dtype=torch.long).unsqueeze(0)
            return features

    def get_image_features(self, image_features, image_dict, pad_seq_length):
        pad_seq_length = min(pad_seq_length, self.max_image_seq_len, self.auto_model.config.max_position_embeddings-1)
        vision_features = image_features['features'][:pad_seq_length]

        boxes = image_features['boxes'][:self.max_image_seq_len]
        # assert obj_num == len(boxes_orig) == len(inputs['vision_feats'])
        image_height, image_width = image_features['img_h'], image_features['img_w']
        # https://github.com/microsoft/Oscar/issues/33
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height, scaled_width*scaled_height), axis=1)

        attention = [1]*vision_features.shape[0]
        attention = attention+[0]*(pad_seq_length - len(attention))

        spatial_features = np.concatenate((spatial_features, np.zeros((pad_seq_length - len(spatial_features), spatial_features.shape[1]))))
        vision_features = np.concatenate((vision_features, np.zeros((pad_seq_length - len(vision_features), vision_features.shape[1]))))

        spatial_features = torch.FloatTensor(spatial_features)
        vision_features = torch.FloatTensor(vision_features)
        attention = torch.LongTensor(attention)
        features = dict() #self.get_sentence_features([self.tokenizer.cls_token], pad_seq_length=-1, add_special_tokens=False)
        features["img_feat"] = vision_features.unsqueeze(dim=0)
        features["img_pos_feat"] = spatial_features.unsqueeze(dim=0)
        features["attention_mask"] = attention.unsqueeze(dim=0) #torch.cat((features["attention_mask"], attention.unsqueeze(dim=0)), dim=1)
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        torch.save(self.auto_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        #self.auto_model.save_pretrained(output_path)
        #self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return UNITER(model_path=os.path.join(input_path, "pytorch_model.bin"), **config)

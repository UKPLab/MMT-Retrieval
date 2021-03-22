# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MAX_POSITIONS = 514  # maximum input sequence length

logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal, k=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=params.n_words,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout, n_langs=None):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.n_langs = n_langs
        if n_langs is None:
            self.out_lin = Linear(dim, dim)
        else:
            self.out_lin = nn.ModuleList()
            for i in range(n_langs):
                self.out_lin.append(Linear(dim, dim))

    def forward(self, input, mask, kv=None, cache=None, segment_label=None,id_plus=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        mask for q_len: calc q,k score, how many q words take into account
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        if id_plus:
            cur_layer_id = self.layer_id+1
        else:
            cur_layer_id = self.layer_id
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or cur_layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if cur_layer_id in cache:
                if kv is None:
                    k_, v_ = cache[cur_layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[cur_layer_id]
            cache[cur_layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        if self.n_langs is None:
            return self.out_lin(context)
        else:
            return self.out_lin[segment_label](context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# for image
class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, hidden_size=1024, type_vocab_size=2, hidden_dropout_prob=0.1):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(2048, hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.image_distbution_embeddings = nn.Linear(1600, hidden_size)
        self.image_location_embeddings = nn.Linear(5, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, input_loc=None, input_dist=None,token_type_ids=None):
        seq_length = input_ids.size(1)
        bs = input_ids.size(0)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if input_loc is None:
            input_loc = torch.arange(np.zeros(bs, seq_length, 5), dtype=torch.long, device=input_ids.device)

        image_embeddings = self.image_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        if input_dist is not None:
            dist_embeddings = self.image_distbution_embeddings(input_dist)
            embeddings = image_embeddings + loc_embeddings + dist_embeddings
        else:
            embeddings = image_embeddings + loc_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Moudle for refine layer after encoder

def attention_sub(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,
                 dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = nn.LayerNorm(d_model, eps=1e-12)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention_sub(query_, key_, value_, mask=mask,
                                     dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


# refine module, aim to find objects' relationship according to features
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward)


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, dim, hidden_dim, glue_att=True, dropout=0.1, n_layers=3 ):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(num_heads, dim, project_k_v=1, scale=1, do_aoa=1, norm_q=0, dropout_aoa=dropout)
        layer = AoA_Refiner_Layer(dim, attn,
                                  TransformerFFN(dim, hidden_dim, dim, dropout, gelu_activation=glue_att), dropout)
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class CrossAlignMatrix(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.att_weight_c = Linear(in_dim, hidden_dim)
        self.att_weight_q = Linear(in_dim, hidden_dim)
        self.att_weight_cq = Linear(in_dim, hidden_dim)
        self.align_output = Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-12)
        # self.act = gelu if gelu_activation else F.relu

    def forward(self, c, q, c_mask=None, q_mask=None):
        c_len = c.size(1)
        q_len = q.size(1)

        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        threshold = torch.ones_like(s) * 15
        s = torch.max(torch.min(s, threshold), -threshold)
        # print('s', s)
        mask = (q_mask == 0).unsqueeze(1)
        #print('mask', mask)
        s.masked_fill_(mask, -float('inf'))
        #print('masked s', s)

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)

        # (batch, c_len, hidden_size * 8)
        x = c2q_att

        x = self.align_output(x)

        return x


class BertImagePooler(nn.Module):
    def __init__(self, hidden_size,gelu_activation=True):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.image_dense = nn.Linear(2048,hidden_size)
        self.out_dense = nn.Linear(2*hidden_size,hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.cls_img = nn.Linear(hidden_size, 1)
        self.activation = gelu if gelu_activation else F.relu

    def forward(self, hidden_states,org_img_hidden):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.mean(hidden_states,dim=1)
        pooled_output = self.dense(first_token_tensor)
        org_img_pooled = self.image_dense(org_img_hidden)
        concat_img = torch.cat([pooled_output,org_img_pooled],dim=1)

        pooled_output = self.out_dense(concat_img)
        pooled_output = self.LayerNorm(pooled_output)
        pooled_output = self.activation(pooled_output)

        return self.cls_img(pooled_output)

class VaeEncoder(nn.Module): #CVAE or VAE
    def __init__(self, input_dim,hidden_size):
        super(VaeEncoder, self).__init__()
        self.x_to_mu = nn.Linear(input_dim, hidden_size)
        self.x_to_logvar = nn.Linear(input_dim, hidden_size)
        self.out_dense = nn.Linear(2 * hidden_size, hidden_size)

    def reparameterize(self, x,c):
        mu = self.x_to_mu(x)
        if self.training:
            logvar = self.x_to_logvar(x)
            z = torch.randn(mu.size())
            z = z.cuda()
            z = mu + z * torch.exp(0.5 * logvar)
            kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))

            z = torch.cat([z,c],dim=-1)
            pooled_output = self.out_dense(z)

            return pooled_output, kld
        else:
            z = torch.cat([mu, c],dim=-1)
            pooled_output = self.out_dense(z)
            return pooled_output,None

    def forward(self, x,c):
        return self.reparameterize(x,c)

class LatentDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(LatentDecoder, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_mu = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "latent" the model by simply taking the hidden state corresponding
        # to the first token.
        latent_output = self.dense(hidden_states)
        original_output = self.dense_mu(latent_output)
        original_output = self.LayerNorm(original_output)
        original_output = self.activation(original_output)
        return original_output

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ObjPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_objs = 1600
        self.pad_index = params.pad_index
        dim = params.emb_dim

        self.proj = Linear(dim, self.n_objs, bias=True)

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        # assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_objs)
        loss = F.cross_entropy(scores, y, reduction='mean',ignore_index=-1)

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size,gelu_activation=True):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = gelu if gelu_activation else F.relu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states.to(dtype=self.LayerNorm.weight.dtype))
        return hidden_states



class TransformerModel(nn.Module):
    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers',
                  'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, is_encoder, with_output,is_crossModal=False):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.is_crossModal = is_crossModal

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        if self.n_langs>1:
            self.english_only = False
        else:
            self.english_only = True
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = params.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads  # 8 by default
        self.n_layers = params.n_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        # if params.n_langs > 1: #for mlm load
        #     self.lang_embeddings = Embedding(self.n_langs, self.dim)

        #we set token type embedding
        if params.n_langs>1:
            self.cross_lang_embeddings = Embedding(self.n_langs, self.dim)#contain image
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        self.image_embeddings = BertImageEmbeddings(self.dim, 2, self.dropout)
        self.refine_embeddings = AoA_Refiner_Core(self.n_heads, self.dim, self.hidden_dim,n_layers=params.refine_layers)

        #we set
        self.cross_alignment = CrossAlignMatrix(self.dim, 1, self.dim)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder or self.is_crossModal:
            self.layer_norm15 = nn.ModuleList() #cross_Modal not use this
            self.encoder_attn = nn.ModuleList()

        self.attention_setting = params.attention_setting

        self.use_externel_att = params.use_externel_att


        for _ in range(self.n_layers):
            self.attentions.append(
                    MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            #if self.is_decoder or self.is_crossModal:
            self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
            # if self.english_only is True:
            self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))

            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout,
                                            gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        #for understanding tasks
        self.pooled_layer = BertPooler(self.dim)
        self.seq_relationship = nn.Linear(self.dim, 1)  # visual-linguistic matching

        self.mrfr_dense = nn.Linear(self.dim, 2048, bias=True)

        #for obj prediction
        self.transformer_obj = BertPredictionHeadTransform(self.dim)

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            self.pred_obj_layer = ObjPredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode=='crossfwd':
            return self.crossfwd(**kwargs)
        elif mode=='jointfwd':
            return self.jointfwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def jointfwd(self, x, lengths, x_img,lengths_img,causal=False,positions=None, langs=None,
                image_loc=None,refine_image=False,is_latent=False):
        """
        :param x: text input
        :param lengths: text length
        :param x_img: img input
        :param lengths_img:  img length 100
        :param causal: not for generation masks
        :param positions: need to update with img length
        :param langs:  lang's ids not use for jointfwd
        :param image_loc: image location features 5-D
        :param refine_image: whether use AOA Refine module
        :param is_latent : not use now
        :return:
        """

        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        x_img = x_img.transpose(0,1)

        #image embedding
        img_tensor = self.image_embeddings(x_img, image_loc.transpose(0, 1))

        # get img masks
        img_mask, img_attn_mask = get_masks(img_tensor.size()[1], lengths_img, False)
        # if refine_image:
        #     img_tensor = self.refine_embeddings(img_tensor, img_attn_mask)


        #text and combine image and text
        tensor = self.embeddings(x)


        c_slen = img_tensor.size()[1]+slen
        cat_length = torch.add(lengths_img,lengths)

        mask, self_attn_masks = get_masks(c_slen, cat_length, causal) #[B,IMG+SENT,IMG+SENT]


        tensor = torch.cat([img_tensor, tensor], dim=1)#[B,IMG+SENT,DIM]

        # positions
        positions = x.new(c_slen).long()
        positions = torch.arange(c_slen, out=positions).unsqueeze(0)

        # combine position embedding and token type embedding
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # if langs is not None:
        #     tensor = tensor + self.cross_lang_embeddings(langs)

        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)


        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, self_attn_masks, cache=None)

            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        tensor = tensor.transpose(0, 1)
        #
        # if is_latent:
        #     return tensor,original_text,original_img,text_kld,img_kld
        return tensor

    def crossfwd(self, x, lengths, causal, stream_='text',src_enc=None, src_len=None, positions=None, langs=None, cache=None, enc_mask=None,
                image_loc=None,refine_image=False, refine_encoder=False,image_dist=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
            'refine_image'  refine module after image embeddings
            'refine_encoder' refine module after encoder

            'image_fusion': whether fusion image into text
            'image_enc': fusion image embedding
            'image_mask' fusion image mask

            "stream_' :text or img
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        assert stream_ in ['img','text']
        assert self.is_crossModal

        if stream_=='img':
            slen, bs = x.size()[0], x.size()[1]
        else:
            slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_crossModal
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_crossModal and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]
            if enc_mask is not None:
                src_mask &= enc_mask

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        original_embedding = None
        if stream_=='img':
            tensor = self.image_embeddings(x, image_loc.transpose(0, 1),image_dist)
            if langs is not None:  # currently we set langs=None
                tensor = tensor + self.cross_lang_embeddings(langs)  # [en fr de ....img]
            #tensor = self.layer_norm_emb(tensor)
            tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        else:
            tensor = self.embeddings(x)
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
            if langs is not None:  # currently we set langs=None
                tensor = tensor + self.cross_lang_embeddings(langs)  #[en fr de ....img]
            tensor = self.layer_norm_emb(tensor)
            tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        lang_id = langs.max() if langs is not None else None

        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # # wether refine embeddings as applying encoding
        # if stream_=='img' and refine_image:
        #     tensor = self.refine_embeddings(tensor, attn_mask)
        #
        # if image_fusion and image_enc is not None and image_mask is not None:
        #     tensor = self.cross_alignment(tensor,image_enc,attn_mask,image_mask)

        # transformer layers
        for i in range(self.n_layers):
            # self attention

            attn = self.attentions[i](tensor, attn_mask, cache=cache)

            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            if causal and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)


        return tensor

    def predict(self, tensor, pred_mask=None, y=None, get_scores=None,is_obj=False,is_relation=False,is_mrfr=False,
                ):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
            'is_obj" is a boolean specifying whether predict mask object regions
        """

        if is_relation:
            enc2_pooled = self.pooled_layer(tensor)
            relation_scores = self.seq_relationship(enc2_pooled)
            return relation_scores
        if is_mrfr:
            mrfr_out = self.mrfr_dense(tensor)
            return mrfr_out
        if is_obj:
            tensor = self.transformer_obj(tensor)
        else:
            masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        if is_obj:
            scores, loss = self.pred_obj_layer(tensor, y, get_scores)
        else:
            scores, loss = self.pred_layer(masked_tensor, y, get_scores)

        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None,cross_modal=True):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # language IDs
        if tgt_lang_id != None:
            langs = src_len.new(max_len).long().fill_(tgt_lang_id)
            langs = langs.unsqueeze(1).expand(max_len, bs)
        else:
            langs = None

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:
            if langs == None:
                cur_langs = None
            else:
                cur_langs = langs[:cur_len]
            # compute word scores
            if self.is_crossModal:
                tensor = self.forward(
                    'crossfwd',
                    x=generated[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    langs=cur_langs,
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    cache=cache
                )
            else:
                tensor = self.forward(
                    'fwd',
                    x=generated[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    langs=cur_langs,
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    cache=cache
                )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :]  # (bs, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
            (bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        #if tgt_lang_id != None:
        langs = positions.clone().fill_(tgt_lang_id)
        # else:
        #     langs = None

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:
            # if langs != None:
            cur_langs = langs[:cur_len]
            # else:
            #     cur_langs = None
            # compute word scores
            if self.is_crossModal:
                tensor = self.forward(
                    'crossfwd',
                    x=generated[:cur_len],
                    lengths=src_len.new(bs * beam_size).fill_(cur_len),
                    positions=positions[:cur_len],
                    langs=cur_langs,
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    cache=cache
                )
            else:
                tensor = self.forward(
                    'fwd',
                    x=generated[:cur_len],
                    lengths=src_len.new(bs * beam_size).fill_(cur_len),
                    positions=positions[:cur_len],
                    langs=cur_langs,
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    cache=cache
                )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(),
                                                    value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty

import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers import util


def cosine_distance(captions, images):
    """
    Compute the 2D matrix of cosine distances (1-cosine_similarity) between all embeddings.
    """
    return 1 - util.pytorch_cos_sim(captions, images)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels):
        logits = self.model(sentence_features[0])["logits"]
        loss = self.loss(logits, labels.view(-1))
        return loss

class BatchHardTripletLoss(nn.Module):
    """
    Our implementation of multimodal in-batch triplet loss from VSE++
    https://arxiv.org/abs/1707.05612
    """
    def __init__(self, model, margin: float = 0.2):
        super(BatchHardTripletLoss, self).__init__()
        self.model = model
        self.triplet_margin = margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
        captions, images = reps
        return self.batch_hard_triplet_loss(labels, captions, images)


    def batch_hard_triplet_loss(self, labels: Tensor, captions, images) -> Tensor:
        # Get the pairwise distance matrix
        pairwise_dist = cosine_distance(captions, images)

        hardest_positive_dist = torch.diagonal(pairwise_dist)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float().squeeze()

        anchor_negative_dist = pairwise_dist + 2 * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist0, _ = anchor_negative_dist.min(0)
        hardest_negative_dist1, _ = anchor_negative_dist.min(1)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = relu(hardest_positive_dist - hardest_negative_dist0 + self.triplet_margin) \
             + relu(hardest_positive_dist - hardest_negative_dist1 + self.triplet_margin)
        triplet_loss = tl.mean()

        return triplet_loss

    @staticmethod
    def get_anchor_negative_triplet_mask(labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


class SampledHardTripletLoss(nn.Module):
    """
    For our implementation of ANCE (Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval)
    https://arxiv.org/abs/2007.00808
    """
    def __init__(self, model, margin: float = 0.2):
        super(SampledHardTripletLoss, self).__init__()
        self.model = model
        self.triplet_margin = margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
        pos_captions, pos_images = reps[0], reps[len(reps)//2]
        neg_captions, neg_images = torch.cat(reps[1:len(reps)//2]), torch.cat(reps[1+len(reps)//2:])
        return self.batch_hard_triplet_loss(neg_captions, neg_images, pos_captions, pos_images)


    def batch_hard_triplet_loss(self, neg_captions, neg_images, pos_captions, pos_images) -> Tensor:
        pairwise_dist0 = 1-torch.cosine_similarity(pos_captions.unsqueeze(dim=1), neg_images.view(
            neg_captions.size()[0]//pos_captions.size()[0], pos_captions.size()[0], -1).permute(1, 0, 2), dim=2)
        pairwise_dist1 = 1-torch.cosine_similarity(pos_images.unsqueeze(dim=1), neg_captions.view(
            neg_captions.size()[0]//pos_captions.size()[0], pos_captions.size()[0], -1).permute(1, 0, 2), dim=2)

        hardest_positive_dist = 1 - torch.cosine_similarity(pos_images, pos_captions, dim=1)

        tl = relu(hardest_positive_dist.unsqueeze(dim=1) - pairwise_dist0 + self.triplet_margin) \
             + relu(hardest_positive_dist.unsqueeze(dim=1) - pairwise_dist1 + self.triplet_margin)
        triplet_loss = tl.mean()

        return triplet_loss

class NTXentLossTriplet(nn.Module):
    def __init__(self, model, scale: float = 20.0):
        super(NTXentLossTriplet, self).__init__()
        self.model = model
        self.scale = scale
        self.loss_fct = nn.CrossEntropyLoss()
        self.similarity_fct = util.pytorch_cos_sim

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
        captions, images = torch.cat(reps[:-1]), reps[-1]
        scores = self.similarity_fct(images, captions) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device) # Example a[i] should match with b[i]
        cross_entropy_loss = nn.CrossEntropyLoss()
        return cross_entropy_loss(scores, labels)


### Legacy experimental losses
#
# class AttractRepelLoss(nn.Module):
#     def __init__(self, model, pos_margin: float = 0.2, neg_margin: float = 0.2):
#         super(AttractRepelLoss, self).__init__()
#         self.model = model
#         self.pos_margin = pos_margin
#         self.neg_margin = neg_margin
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
#         pos_captions, neg_captions, images = reps
#         return self.pos_triplet_loss(labels, pos_captions, images) + self.neg_triplet_loss(labels, neg_captions, images)
#
#     def pos_triplet_loss(self, labels: Tensor, captions, images) -> Tensor:
#         # Get the pairwise distance matrix
#         pairwise_dist = cosine_distance(captions, images)
#
#         ## For each anchor, get the hardest positive
#         ## First, we need to get a mask for every valid positive (they should have same label)
#         #mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()
#
#         ## We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
#         #anchor_positive_dist = mask_anchor_positive * pairwise_dist
#
#         # shape (batch_size, 1)
#         #hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
#         hardest_positive_dist = torch.diagonal(pairwise_dist)
#
#         # For each anchor, get the hardest negative
#         # First, we need to get a mask for every valid negative (they should have different labels)
#         mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float().squeeze()
#
#         anchor_negative_dist = pairwise_dist + 2 * (1.0 - mask_anchor_negative)
#
#         # shape (batch_size,)
#         hardest_negative_dist0, _ = anchor_negative_dist.min(0)
#         hardest_negative_dist1, _ = anchor_negative_dist.min(1)
#
#         # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
#         tl = relu(hardest_positive_dist - hardest_negative_dist0 + self.pos_margin) \
#              + relu(hardest_positive_dist - hardest_negative_dist1 + self.pos_margin)
#         triplet_loss = tl.mean()
#
#         return triplet_loss
#
#     def neg_triplet_loss(self, labels: Tensor, captions, images) -> Tensor:
#         # Get the pairwise distance matrix
#         pairwise_dist = cosine_distance(captions, images)
#
#         hardest_positive_dist = torch.diagonal(pairwise_dist)
#
#         # For each anchor, get the hardest negative
#         # First, we need to get a mask for every valid negative (they should have different labels)
#         mask_anchor_negative = AttractRepelLoss.get_anchor_negative_triplet_mask(labels).float().squeeze()
#
#         anchor_negative_dist = pairwise_dist - 2 * (1.0 - mask_anchor_negative)
#
#         # shape (batch_size,)
#         hardest_negative_dist0, _ = anchor_negative_dist.max(0)
#         hardest_negative_dist1, _ = anchor_negative_dist.max(1)
#
#         # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
#         tl = relu(-hardest_positive_dist + hardest_negative_dist0 + self.neg_margin) \
#              + relu(-hardest_positive_dist + hardest_negative_dist1 + self.neg_margin)
#         triplet_loss = tl.mean()
#
#         return triplet_loss
#
#
#     @staticmethod
#     def get_triplet_mask(labels):
#         """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
#         A triplet (i, j, k) is valid if:
#             - i, j, k are distinct
#             - labels[i] == labels[j] and labels[i] != labels[k]
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         """
#         # Check that i, j and k are distinct
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#         i_not_equal_j = indices_not_equal.unsqueeze(2)
#         i_not_equal_k = indices_not_equal.unsqueeze(1)
#         j_not_equal_k = indices_not_equal.unsqueeze(0)
#
#         distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
#
#         label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#         i_equal_j = label_equal.unsqueeze(2)
#         i_equal_k = label_equal.unsqueeze(1)
#
#         valid_labels = ~i_equal_k & i_equal_j
#
#         return valid_labels & distinct_indices
#
#     @staticmethod
#     def get_anchor_positive_triplet_mask(labels):
#         """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check that i and j are distinct
#
#
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#
#         # Check if labels[i] == labels[j]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#         labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#
#         return labels_equal & indices_not_equal
#
#     @staticmethod
#     def get_anchor_negative_triplet_mask(labels):
#         """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check if labels[i] != labels[k]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#
#         return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
#
# class NLITripletLoss(nn.Module):
#     def __init__(self, model, inter_margin: float = 0.2, in_margin: float = 0.2, do_neutral=True):
#         super(NLITripletLoss, self).__init__()
#         self.model = model
#         self.inter_margin = inter_margin
#         self.in_margin = in_margin
#         self.do_neutral = do_neutral
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
#         return self.inter_triplet_loss(labels, reps[0], reps[-1]) + self.in_triplet_loss(reps)
#
#     def inter_triplet_loss(self, labels: Tensor, captions, images) -> Tensor:
#         pairwise_dist = cosine_distance(captions, images)
#
#         ## For each anchor, get the hardest positive
#         ## First, we need to get a mask for every valid positive (they should have same label)
#         #mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()
#
#         ## We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
#         #anchor_positive_dist = mask_anchor_positive * pairwise_dist
#
#         # shape (batch_size, 1)
#         #hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
#         hardest_positive_dist = torch.diagonal(pairwise_dist)
#
#         # For each anchor, get the hardest negative
#         # First, we need to get a mask for every valid negative (they should have different labels)
#         mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float().squeeze()
#
#         anchor_negative_dist = pairwise_dist + 2 * (1.0 - mask_anchor_negative)
#
#         # shape (batch_size,)
#         hardest_negative_dist0, _ = anchor_negative_dist.min(0)
#         hardest_negative_dist1, _ = anchor_negative_dist.min(1)
#
#         # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
#         tl = relu(hardest_positive_dist - hardest_negative_dist0 + self.inter_margin) \
#              + relu(hardest_positive_dist - hardest_negative_dist1 + self.inter_margin)
#         triplet_loss = tl.mean()
#
#         return triplet_loss
#
#     def in_triplet_loss(self, reps) -> Tensor:
#         cos_entailment = 1-torch.cosine_similarity(reps[0], reps[-1], dim=1)
#         cos_contradiction = 1-torch.cosine_similarity(reps[-2], reps[-1], dim=1)
#         if self.do_neutral:
#             cos_neutral = 1-torch.cosine_similarity(reps[1], reps[-1], dim=1)
#
#         tl = relu(cos_entailment - cos_contradiction + self.in_margin)
#         if self.do_neutral:
#             tl = tl + relu(cos_entailment - cos_neutral + self.in_margin) + relu(cos_neutral - cos_contradiction + self.in_margin)
#
#         triplet_loss = tl.mean()
#
#         return triplet_loss
#
#     @staticmethod
#     def get_triplet_mask(labels):
#         """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
#         A triplet (i, j, k) is valid if:
#             - i, j, k are distinct
#             - labels[i] == labels[j] and labels[i] != labels[k]
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         """
#         # Check that i, j and k are distinct
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#         i_not_equal_j = indices_not_equal.unsqueeze(2)
#         i_not_equal_k = indices_not_equal.unsqueeze(1)
#         j_not_equal_k = indices_not_equal.unsqueeze(0)
#
#         distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
#
#         label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#         i_equal_j = label_equal.unsqueeze(2)
#         i_equal_k = label_equal.unsqueeze(1)
#
#         valid_labels = ~i_equal_k & i_equal_j
#
#         return valid_labels & distinct_indices
#
#     @staticmethod
#     def get_anchor_positive_triplet_mask(labels):
#         """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check that i and j are distinct
#
#
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#
#         # Check if labels[i] == labels[j]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#         labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#
#         return labels_equal & indices_not_equal
#
#     @staticmethod
#     def get_anchor_negative_triplet_mask(labels):
#         """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check if labels[i] != labels[k]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#
#         return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
#
# class BatchHarderTripletLoss(nn.Module):
#     def __init__(self, model, margin: float = 0.2):
#         super(BatchHarderTripletLoss, self).__init__()
#         self.model = model
#         self.triplet_margin = margin
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         reps = [self.model(sentence_feature)['multimodal_embedding'] for sentence_feature in sentence_features]
#         captions, images = torch.cat(reps[:len(reps)//2]), torch.cat(reps[len(reps)//2:])
#         labels_capt, labels_img = torch.cat([l for l in labels[:,0].permute(1, 0)]), torch.cat([l for l in labels[:,1].permute(1, 0)])
#         return self.batch_hard_triplet_loss(labels_capt, labels_img, captions, images, len(reps[0]))
#
#
#     def batch_hard_triplet_loss(self, labels_capt, labels_img, captions, images, bs) -> Tensor:
#         # Get the pairwise distance matrix
#         pairwise_dist0 = cosine_distance(captions[:bs], images)
#         pairwise_dist1 = cosine_distance(images[:bs], captions)
#
#         hardest_positive_dist0 = torch.diagonal(pairwise_dist0)
#         hardest_positive_dist1 = torch.diagonal(pairwise_dist1)
#         # For each anchor, get the hardest negative
#         # First, we need to get a mask for every valid negative (they should have different labels)
#         mask_anchor_negative0 = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels_capt).float().squeeze()
#         anchor_negative_dist0 = pairwise_dist0 + 2 * (1.0 - mask_anchor_negative0[:bs])
#
#         mask_anchor_negative1 = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels_img).float().squeeze()
#         anchor_negative_dist1 = pairwise_dist1 + 2 * (1.0 - mask_anchor_negative1[:bs])
#
#         # shape (batch_size,)
#         hardest_negative_dist0 = anchor_negative_dist0.min(1)[0]
#         hardest_negative_dist1 = anchor_negative_dist1.min(1)[0]
#
#         # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
#         tl = relu(hardest_positive_dist0 - hardest_negative_dist0 + self.triplet_margin) \
#              + relu(hardest_positive_dist1 - hardest_negative_dist1 + self.triplet_margin)
#         triplet_loss = tl.mean()
#
#         return triplet_loss
#
#     @staticmethod
#     def get_triplet_mask(labels):
#         """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
#         A triplet (i, j, k) is valid if:
#             - i, j, k are distinct
#             - labels[i] == labels[j] and labels[i] != labels[k]
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         """
#         # Check that i, j and k are distinct
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#         i_not_equal_j = indices_not_equal.unsqueeze(2)
#         i_not_equal_k = indices_not_equal.unsqueeze(1)
#         j_not_equal_k = indices_not_equal.unsqueeze(0)
#
#         distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
#
#         label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#         i_equal_j = label_equal.unsqueeze(2)
#         i_equal_k = label_equal.unsqueeze(1)
#
#         valid_labels = ~i_equal_k & i_equal_j
#
#         return valid_labels & distinct_indices
#
#     @staticmethod
#     def get_anchor_positive_triplet_mask(labels):
#         """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check that i and j are distinct
#
#
#         indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#         indices_not_equal = ~indices_equal
#
#         # Check if labels[i] == labels[j]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#         labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#
#         return labels_equal & indices_not_equal
#
#     @staticmethod
#     def get_anchor_negative_triplet_mask(labels):
#         """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
#         Args:
#             labels: tf.int32 `Tensor` with shape [batch_size]
#         Returns:
#             mask: tf.bool `Tensor` with shape [batch_size, batch_size]
#         """
#         # Check if labels[i] != labels[k]
#         # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
#
#         return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
#
#
#

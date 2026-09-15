# -*- coding: utf-8 -*-
"""MobileNetV3-small encoder and supervised contrastive objective."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


class MegaHanLiteEncoder(nn.Module):
    """Encode 64px glyph images into normalized compact embeddings."""

    def __init__(self, embedding_size=128):
        super().__init__()
        if (
            not isinstance(embedding_size, int)
            or isinstance(embedding_size, bool)
            or embedding_size <= 0
        ):
            raise ValueError("embedding_size must be a positive integer")

        backbone = mobilenet_v3_small(weights=None)
        feature_size = backbone.classifier[0].in_features
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(
            feature_size,
            embedding_size,
            bias=False,
        )

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, 1)
        embeddings = self.projection(flattened)
        return F.normalize(embeddings, p=2, dim=1, eps=1e-12)


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.07,
):
    """Compute stable single-view supervised contrastive loss."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embedding]")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if embeddings.shape[0] < 2:
        raise ValueError("the batch has no positive pair")
    if not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must be floating point")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be positive")

    labels = labels.to(device=embeddings.device)
    normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = embeddings.shape[0]
    self_mask = torch.eye(
        batch_size,
        device=embeddings.device,
        dtype=torch.bool,
    )
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_counts = positive_mask.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not bool(valid_anchors.any().item()):
        raise ValueError("the batch has no positive pair")

    denominator_logits = logits.masked_fill(self_mask, float("-inf"))
    log_denominator = torch.logsumexp(
        denominator_logits,
        dim=1,
        keepdim=True,
    )
    log_probabilities = logits - log_denominator
    positive_log_probabilities = log_probabilities.masked_fill(
        ~positive_mask,
        0.0,
    )
    mean_positive_log_probability = (
        positive_log_probabilities.sum(dim=1)
        / positive_counts.clamp_min(1)
    )
    return -mean_positive_log_probability[valid_anchors].mean()

"""
Gating: turn regime probabilities into weights over the experts.

Each regime dimension d (e.g. day type, week type) has a classifier that gives
P_d(label | x). The joint probability of expert (l_1, ..., l_m) is the product
of its labels' probabilities, re-normalized over the experts that exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator

from moeforecast.experts import ExpertKey


@dataclass
class GatingResult:
    """Output of a gating rule for n time steps and E experts."""

    weights: np.ndarray  # (n, E), rows sum to 1
    mode: np.ndarray  # (n,), "hard" or "soft"
    confidence: np.ndarray  # (n,), min over dimensions of the top class probability


def joint_probabilities(
    probas: Sequence[np.ndarray],
    classes: Sequence[np.ndarray],
    expert_keys: Sequence[ExpertKey],
    temperature: float = 1.0,
) -> np.ndarray:
    """(n, E) product of per-dimension probabilities, re-normalized over the experts."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    n = probas[0].shape[0]
    joint = np.ones((n, len(expert_keys)))
    for d, (proba, cls) in enumerate(zip(probas, classes, strict=True)):
        column = {label: j for j, label in enumerate(cls)}
        for e, key in enumerate(expert_keys):
            j = column.get(key[d])
            joint[:, e] *= proba[:, j] if j is not None else 0.0
    total = joint.sum(axis=1, keepdims=True)
    uniform = np.full_like(joint, 1.0 / len(expert_keys))
    normalized = np.where(total > 0, joint / np.where(total > 0, total, 1), uniform)
    calibrated = np.power(normalized, 1.0 / temperature)
    calibrated_total = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.where(calibrated_total > 0, calibrated_total, 1)


def _confidence(probas: Sequence[np.ndarray]) -> np.ndarray:
    return np.min(np.column_stack([p.max(axis=1) for p in probas]), axis=1)


def _hard(joint: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(joint)
    weights[np.arange(len(joint)), joint.argmax(axis=1)] = 1.0
    return weights


def _soft(joint: np.ndarray, min_weight: float) -> np.ndarray:
    weights = np.where(joint >= min_weight, joint, 0.0)
    total = weights.sum(axis=1, keepdims=True)
    empty = total[:, 0] == 0
    weights = weights / np.where(total > 0, total, 1)
    weights[empty] = _hard(joint[empty])
    return weights


class HardGating(BaseEstimator):
    """Route every time step to its single most probable expert."""

    def route(self, probas, classes, expert_keys) -> GatingResult:
        joint = joint_probabilities(probas, classes, expert_keys)
        return GatingResult(_hard(joint), np.full(len(joint), "hard"), _confidence(probas))


class SoftGating(BaseEstimator):
    """
    Weight the experts by their joint probability.

    Parameters
    ----------
    min_weight : float
        Experts whose (re-normalized) joint probability is below this value are
        dropped and the remaining weights re-normalized.
    temperature : float
        Positive calibration temperature. Values below 1 sharpen probabilities;
        values above 1 make the soft weights more uniform.
    """

    def __init__(self, min_weight: float = 0.05, temperature: float = 1.0):
        self.min_weight = min_weight
        self.temperature = temperature

    def route(self, probas, classes, expert_keys) -> GatingResult:
        joint = joint_probabilities(probas, classes, expert_keys, self.temperature)
        weights = _soft(joint, self.min_weight)
        return GatingResult(weights, np.full(len(joint), "soft"), _confidence(probas))


class AdaptiveGating(BaseEstimator):
    """
    Hard gating when the classifiers are confident, soft gating otherwise.

    The confidence of a time step is the smallest, over regime dimensions, of
    the classifier's top-class probability.

    Parameters
    ----------
    threshold : float
        Confidence at or above which hard gating is used.
    min_weight : float
        Passed to the soft rule.
    temperature : float
        Positive calibration temperature applied to joint expert probabilities.
    """

    def __init__(self, threshold: float = 0.75, min_weight: float = 0.05, temperature: float = 1.0):
        self.threshold = threshold
        self.min_weight = min_weight
        self.temperature = temperature

    def route(self, probas, classes, expert_keys) -> GatingResult:
        joint = joint_probabilities(probas, classes, expert_keys, self.temperature)
        confidence = _confidence(probas)
        is_hard = confidence >= self.threshold
        weights = np.where(is_hard[:, None], _hard(joint), _soft(joint, self.min_weight))
        mode = np.where(is_hard, "hard", "soft")
        return GatingResult(weights, mode, confidence)

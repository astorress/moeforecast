import numpy as np

from moeforecast import AdaptiveGating, HardGating, SoftGating
from moeforecast.gating import joint_probabilities

KEYS = [(0, 0), (0, 1), (1, 0)]  # expert (1, 1) does not exist
CLASSES = [np.array([0, 1]), np.array([0, 1])]
PROBAS = [np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]]), np.array([[0.95, 0.05], [0.1, 0.9], [0.5, 0.5]])]


def test_joint_probabilities_renormalize_over_existing_experts():
    joint = joint_probabilities(PROBAS, CLASSES, KEYS)
    assert np.allclose(joint.sum(axis=1), 1)
    # Row 1: most likely combination (1, 1) has no expert; mass goes to the others
    assert joint[1].argmax() == 2 or joint[1].argmax() == 1


def test_hard_gating_is_one_hot():
    w = HardGating().route(PROBAS, CLASSES, KEYS).weights
    assert np.array_equal(w.sum(axis=1), np.ones(3)) and set(np.unique(w)) <= {0.0, 1.0}
    assert w[0, 0] == 1.0


def test_soft_and_adaptive_modes():
    soft = SoftGating(min_weight=0.05).route(PROBAS, CLASSES, KEYS)
    assert np.allclose(soft.weights.sum(axis=1), 1)
    ada = AdaptiveGating(threshold=0.85).route(PROBAS, CLASSES, KEYS)
    assert list(ada.mode) == ["hard", "soft", "soft"]
    assert np.allclose(ada.weights.sum(axis=1), 1)


def test_class_missing_from_classifier_gets_zero_weight():
    classes = [np.array([0]), np.array([0, 1])]
    probas = [np.ones((2, 1)), np.array([[0.5, 0.5], [0.2, 0.8]])]
    joint = joint_probabilities(probas, classes, KEYS)
    assert np.allclose(joint[:, 2], 0)


def test_temperature_controls_soft_probability_concentration():
    cold = SoftGating(min_weight=0, temperature=0.5).route(PROBAS, CLASSES, KEYS).weights
    warm = SoftGating(min_weight=0, temperature=2.0).route(PROBAS, CLASSES, KEYS).weights
    assert np.all(cold.max(axis=1) >= warm.max(axis=1))

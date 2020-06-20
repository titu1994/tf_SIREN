import pytest
import numpy as np
import tensorflow as tf
import tf_siren


def test_sine_activation():
    x = tf.constant([0.0, np.pi / 2.0, np.pi])
    activation = tf_siren.Sine(w0=1.0)
    y = activation(x)

    assert (tf.reduce_sum(y - [0.0, 1.0, 0.0]) ** 2) <= 1e-6


def test_sine_activation_with_w0():
    x = tf.constant([0.0, np.pi / 2.0, np.pi])
    activation = tf_siren.Sine(w0=1.5)
    y = activation(x)

    assert (tf.reduce_sum(y - [0.0, 1. / np.sqrt(2.0), -1.0]) ** 2) <= 1e-6


if __name__ == '__main__':
    pytest.main(__file__)


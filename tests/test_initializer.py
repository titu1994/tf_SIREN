import pytest
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tf_siren

normal_dist = stats.norm(0, 1)


def test_siren_initialization():
    tf.random.set_seed(0)

    c = 6.0
    samples = 10
    dim = 128
    activation = tf_siren.Sine(w0=1.0, dtype=tf.float32)
    initializer = tf_siren.SIRENInitializer(w0=1.0, c=c, seed=0)

    x = tf.random.uniform(shape=[samples, dim], minval=-1.0, maxval=1.)
    out = x

    # weight matrix
    w = initializer(shape=[dim, dim], dtype=tf.float32)

    for i in range(100):  # Simulate 100 layer SIREN forward pass
        out = tf.matmul(out, w)  # The paper suggests output should have zero mean, unit std.

        out_np = out.numpy()[0]  # select just one sample to test later
        # print(i + 1, out_np.mean(), out_np.std())

        out = activation(out)

        # Simulate new layer with new weights for next pass
        w = initializer(shape=[dim, dim], dtype=tf.float32)

    out = out_np
    check_dist = tf.random.normal(out.shape, stddev=1.0).numpy()

    # Compute 2 tailed z statistic
    means = out.mean() - check_dist.mean()
    stds = np.sqrt((np.square(out.std()) + np.square(check_dist.std())))

    z_statistic = abs(means / (stds))

    # multiply by 2.0 as it is a 2 tailed test
    probability_same_dist = 2 * (1. - normal_dist.cdf(z_statistic))

    assert probability_same_dist >= 0.95  # more than 95% confident the two distributions have same mean and std


def test_siren_initialization_with_w0():
    tf.random.set_seed(0)

    w0 = 2.0
    c = 6.0
    samples = 10
    dim = 128
    activation = tf_siren.Sine(w0=w0, dtype=tf.float32)
    initializer = tf_siren.SIRENInitializer(w0=w0, c=c, seed=0)

    x = tf.random.uniform(shape=[samples, dim], minval=-1.0, maxval=1.)
    out = x

    # weight matrix
    w = initializer(shape=[dim, dim], dtype=tf.float32)

    for i in range(100):  # Simulate 100 layer SIREN forward pass
        # for w0 > 1, we multiply w0 by weight matrix before forward pass
        out = tf.matmul(out, w0 * w)  # The paper suggests output should have zero mean, unit std.

        out_np = out.numpy()[0]  # select just one sample to test later
        # print(i + 1, out_np.mean(), out_np.std())

        out = activation(out)

        # Simulate new layer with new weights for next pass
        w = initializer(shape=[dim, dim], dtype=tf.float32)

    out = out_np
    check_dist = tf.random.normal(out.shape, stddev=1.0).numpy()

    # Compute 2 tailed z statistic
    means = out.mean() - check_dist.mean()
    stds = np.sqrt((np.square(out.std()) + np.square(check_dist.std())))

    z_statistic = abs(means / (stds))
    # multiply by 2.0 as it is a 2 tailed test
    probability_same_dist = 2 * (1. - normal_dist.cdf(z_statistic))

    assert probability_same_dist >= 0.90  # more than 90% confident the two distributions have same mean and std


if __name__ == '__main__':
    pytest.main(__file__)

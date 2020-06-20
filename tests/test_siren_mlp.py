import pytest
import numpy as np
import tensorflow as tf
import tf_siren


def test_sine_layer():
    ip = tf.keras.layers.Input(shape=[2])
    out = tf_siren.SinusodialRepresentationDense(units=32, w0=1.0)(ip)

    model = tf.keras.Model(inputs=ip, outputs=out)
    model.compile('adam', 'mse')

    x = tf.random.uniform([10, 2], -1., 1.)
    y = model(x, training=True)

    assert y.shape == (10, 32)

    y = tf.random.normal([10, 32])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=8, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


def test_sine_layer_with_w0():
    w0 = 2.0

    ip = tf.keras.layers.Input(shape=[2])
    out = tf_siren.SinusodialRepresentationDense(units=32, w0=w0)(ip)

    model = tf.keras.Model(inputs=ip, outputs=out)
    model.compile('adam', 'mse')

    x = tf.random.uniform([10, 2], -1., 1.)
    y = model(x, training=True)

    assert y.shape == (10, 32)

    y = tf.random.normal([10, 32])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=8, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


def test_SIREN_model():
    model = tf_siren.SIRENModel(32, final_units=32, final_activation='linear',
                                num_layers=2, w0=1.0, w0_initial=30.0)
    model.compile('adam', 'mse')

    x = tf.random.uniform([10, 2], -1., 1.)
    y = model(x, training=True)

    assert y.shape == (10, 32)

    y = tf.random.normal([10, 32])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=8, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


def test_SIREN_model_with_w0():
    w0 = 2.0

    model = tf_siren.SIRENModel(32, final_units=32, final_activation='linear',
                                num_layers=2, w0=w0, w0_initial=30.0)
    model.compile('adam', 'mse')

    x = tf.random.uniform([10, 2], -1., 1.)
    y = model(x, training=True)

    assert y.shape == (10, 32)

    y = tf.random.normal([10, 32])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=8, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


if __name__ == '__main__':
    pytest.main(__file__)

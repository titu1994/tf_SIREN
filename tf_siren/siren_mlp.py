import tensorflow as tf
from tf_siren import siren


class SIRENModel(tf.keras.Model):

    def __init__(self, units: int, final_units: int,
                 final_activation: str = "linear",
                 num_layers: int = 1,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 use_bias: bool = True, **kwargs):
        """
        SIREN model from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661).
        Used to create a multi-layer MLP using SinusodialRepresentationDense layers.

        Args:
            units: Number of hidden units in the intermediate layers.
            final_units: Number of hidden units in the final layer.
            final_activation: Activation function of the final layer.
            num_layers: Number of layers in the network.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            w0_initial: By default, scales `w0` of first layer to 30 (as used in the paper).
            use_bias: Boolean whether to use bias or not.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIRENModel, self).__init__(**kwargs)

        siren_layers = [siren.SinusodialRepresentationDense(units, w0=w0_initial, use_bias=use_bias)]

        for _ in range(num_layers - 1):
            siren_layers.append(siren.SinusodialRepresentationDense(units, w0=w0, use_bias=use_bias))

        self.siren_layers = tf.keras.Sequential(siren_layers)
        self.final_dense = siren.SinusodialRepresentationDense(final_units, activation=final_activation,
                                                               use_bias=use_bias)

    def call(self, inputs, training=None, mask=None):
        features = self.siren_layers(inputs)
        output = self.final_dense(features)
        return output

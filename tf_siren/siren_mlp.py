import tensorflow as tf
from tf_siren import siren


class SIRENModel(tf.keras.Model):

    def __init__(self, units: int, final_units: int,
                 final_activation: str = "linear",
                 num_layers: int = 1,
                 w0: float = 30.0,
                 w0_initial: float = 30.0,
                 initial_layer_init: str = 'siren_first_uniform',
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
            initial_layer_init: Initialization for the first SIREN layer.
                Can be any valid keras initialization object or string.
                For SIREN, use `siren_uniform` for the general initialization,
                or `siren_first_uniform` which is specific for first layer.
            use_bias: Boolean whether to use bias or not.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIRENModel, self).__init__(**kwargs)

        siren_layers = [siren.SinusodialRepresentationDense(units, w0=w0_initial, use_bias=use_bias,
                                                            kernel_initializer=initial_layer_init,
                                                            **kwargs)]

        for _ in range(num_layers - 1):
            siren_layers.append(siren.SinusodialRepresentationDense(units, w0=w0, use_bias=use_bias, **kwargs))

        self.siren_layers = tf.keras.Sequential(siren_layers)
        self.final_dense = siren.SinusodialRepresentationDense(final_units, activation=final_activation,
                                                               use_bias=use_bias, **kwargs)

    def call(self, inputs, training=None, mask=None):
        features = self.siren_layers(inputs)
        output = self.final_dense(features)
        return output


class ScaledSIRENModel(tf.keras.Model):

    def __init__(self, units: int, final_units: int,
                 final_activation: str = "linear",
                 num_layers: int = 1,
                 w0: float = 30.0,
                 w0_initial: float = 30.0,
                 scale: float = 1.0,
                 scale_initial: float = None,
                 initial_layer_init: str = 'siren_first_uniform',
                 use_bias: bool = True, **kwargs):
        """
        Scaled SIREN model from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661).
        Used to create a multi-layer MLP using ScaledSinusodialRepresentationDense layers.

        Args:
            units: Number of hidden units in the intermediate layers.
            final_units: Number of hidden units in the final layer.
            final_activation: Activation function of the final layer.
            num_layers: Number of layers in the network.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            w0_initial: By default, scales `w0` of first layer to 30 (as used in the paper).
            scale: Scale of the kernel matrix prior to matmul.
            scale_initial: Scale of the kernel matrix prior to matmul, for the first layer.
                By default, uses the `w0_initial` value if not passed a value.
            initial_layer_init: Initialization for the first SIREN layer.
                Can be any valid keras initialization object or string.
                For SIREN, use `siren_uniform` for the general initialization,
                or `siren_first_uniform` which is specific for first layer.
            use_bias: Boolean whether to use bias or not.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        super(ScaledSIRENModel, self).__init__(**kwargs)

        if scale_initial is None:
            scale_initial = w0_initial

        siren_layers = [siren.ScaledSinusodialRepresentationDense(units, scale=scale_initial, w0=w0_initial,
                                                                  use_bias=use_bias,
                                                                  kernel_initializer=initial_layer_init,
                                                                  **kwargs)]

        for _ in range(num_layers - 1):
            siren_layers.append(siren.ScaledSinusodialRepresentationDense(units, scale=scale, w0=w0, use_bias=use_bias,
                                                                          **kwargs))

        self.siren_layers = tf.keras.Sequential(siren_layers)
        self.final_dense = siren.ScaledSinusodialRepresentationDense(final_units, scale=scale,
                                                                     activation=final_activation,
                                                                     use_bias=use_bias, **kwargs)

    def call(self, inputs, training=None, mask=None):
        features = self.siren_layers(inputs)
        output = self.final_dense(features)
        return output

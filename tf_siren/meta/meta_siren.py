import tensorflow as tf
from tf_siren import siren


class HyperHeNormalInitializer(tf.keras.initializers.VarianceScaling):

    def __init__(self, scale_factor: float = 0.01, seed=None):
        self.scale_factor = scale_factor
        super().__init__(scale=2., mode="fan_in", distribution="truncated_normal", seed=seed)

    def __call__(self, shape, dtype=tf.float32):
        initialized_tensor = super().__call__(shape, dtype)
        scaled_tensor = initialized_tensor * self.scale_factor
        return scaled_tensor

    def get_config(self):
        base_config = super().get_config()
        config = {
            'scale_factor': self.scale_factor,
        }
        return dict(list(base_config.items()) + list(config.items()))


class _MetaSinusodialRepresentationDense(siren.SinusodialRepresentationDense):
    """
    A Meta wrapper over a SinusodialRepresentationDense.

    Does not have its own weights, accepts parameters during forward call via `params`.
    Unpacks these params and reshapes them for use in batched call of multiple
    kernels and biases over individual samples in the batch.
    """
    @tf.function
    def __call__(self, inputs, params=None):
        # input = [batch, input_dim]
        # kernel = [batch, input_dim, output_dim]
        # bias = [batch, output_dim]

        if self.use_bias:
            kernel, bias = params
        else:
            kernel = params

        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            bias = tf.expand_dims(bias, axis=1)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable

        return outputs


class HyperNetBlock(tf.keras.layers.Layer):

    def __init__(self, input_units: int,
                 output_units: int,
                 hyper_units: int,
                 activation: str = 'relu',
                 num_hyper_layers: int = 1,
                 hyper_final_activation: str = 'linear',
                 use_bias: bool = True,
                 kernel_initializer: str = 'he_normal',
                 bias_initializer: str = 'he_uniform',
                 kernel_regularizer: float = 0.0,
                 bias_regularizer: float = 0.0,
                 **kwargs):
        """
        A meta layer that computes the weights for a single layer.
        Represents a single layer SinusodialRepresentationDense.

        Args:
            input_units: Positive integer, dimensionality of the input space.
            output_units: Positive integer, dimensionality of the output space.
            hyper_units: Positive integer, dimensionality of the hidden space of the
                hyper feed forward network.
            activation: Activation function to use.
                If you don't specify anything, relu activation is applied
            num_hyper_layers: Number of layers in the hyper network.
            hyper_final_activation: Activation function to use for the final layer of
                the hyper network.
            use_bias: Boolean whether to use bias or not.
            kernel_initializer: String defining the initializer used for the kernel matrix.
            bias_initializer: String defining the initializer used for the bias matrix.
            kernel_regularizer: Float defining the regularization strength used for the kernel matrix.
            bias_regularizer: Float defining the regularization strength used for the kernel matrix.
        """
        super().__init__(**kwargs)

        self.input_units = input_units
        self.output_units = output_units
        self.hyper_units = hyper_units
        self.use_bias = use_bias

        if kernel_regularizer != 0.0:
            kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        else:
            kernel_regularizer = None

        if bias_regularizer != 0.0:
            bias_regularizer = tf.keras.regularizers.l2(bias_regularizer)
        else:
            bias_regularizer = None

        hyper_net = []
        hyper_net.append(tf.keras.layers.Dense(hyper_units, activation=activation, use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               **kwargs))

        for _ in range(num_hyper_layers - 1):
            hyper_net.append(tf.keras.layers.Dense(hyper_units, activation=activation, use_bias=use_bias,
                                                   kernel_initializer=kernel_initializer,
                                                   bias_initializer=bias_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer, **kwargs))

        hyper_net.append(tf.keras.layers.Dense(output_units, activation=hyper_final_activation, use_bias=use_bias,
                                               kernel_initializer='hyper_he_normal',
                                               bias_initializer='siren_first_uniform',
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer, **kwargs))

        self.hyper_net = tf.keras.Sequential(hyper_net, name='hypernet_cell')

    def call(self, context, training=None):
        parameters = self.hyper_net(context)
        return parameters


class MetaSinusodialRepresentationDense(tf.keras.layers.Layer):
    __doc__ = siren.SinusodialRepresentationDense.__doc__

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hyper_units: int,
                 num_hyper_layers: int = 1,
                 w0: float = 1.0,
                 c: float = 6.0,
                 siren_activation: str = 'sine',
                 hyper_activation: str = 'relu',
                 use_bias: bool = True,
                 kernel_initializer: str = 'he_normal',
                 bias_initializer: str = 'he_uniform',
                 kernel_regularizer: float = None,
                 bias_regularizer: float = None,
                 **kwargs):
        """
        A single meta layer in a HyperNetwork. Represents a single layer SinusodialRepresentationDense.

        It comprises of two components:
        1) The feed forward network that computes the weights for a layer.
        2) The meta layer (inner layer) that will use the above computed weights,
            and perform the actual forward pass of the meta layer.

        Args:
            input_units: Positive integer, dimensionality of the input space.
            output_units: Positive integer, dimensionality of the output space.
            hyper_units: Positive integer, dimensionality of the hidden space of the
                hyper feed forward network.
            num_hyper_layers: Number of layers in the hyper network.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            c: Recommended value to scale the distribution when initializing
                weights.
            siren_activation: Activation function for the meta SinusodialRepresentationDense
                layer.
            hyper_activation: Activation function for the hyper network which generates the
                weights for the meta SinusodialRepresentationDense layer.
            use_bias: Boolean whether to use bias or not.
            kernel_initializer: String defining the initializer used for the kernel matrix.
            bias_initializer: String defining the initializer used for the bias matrix.
            kernel_regularizer: Float defining the regularization strength used for the kernel matrix.
            bias_regularizer: Float defining the regularization strength used for the kernel matrix.
        """
        super().__init__(**kwargs)

        self.input_units = input_units
        self.output_units = output_units
        self.hyper_units = hyper_units

        total_param_count = input_units * output_units
        if use_bias:
            total_param_count += output_units

        self.total_param_count = total_param_count
        self.kernel_param_count = input_units * output_units
        self.bias_param_count = output_units

        # Model which provides parameters for inner model
        self.hyper_net = HyperNetBlock(
            input_units=input_units, output_units=total_param_count, hyper_units=hyper_units,
            activation=hyper_activation, num_hyper_layers=num_hyper_layers,
            hyper_final_activation='linear', use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

        # Weights wont be generated for this meta layer, just its forward method will be used
        self.inner_siren = _MetaSinusodialRepresentationDense(
            units=output_units, w0=w0, c=c, activation=siren_activation, use_bias=use_bias
        )

        # Don't allow to build weights
        self.inner_siren.built = True

    @tf.function
    def call(self, context, **kwargs):
        parameters = self.hyper_net(context)  # [B, total_parameter_count]

        # Unpack kernel weights from generated parameters
        kernel = tf.reshape(parameters[:, :self.kernel_param_count],
                            [-1, self.input_units, self.output_units])

        # Unpack bias parameters if available
        if self.hyper_net.use_bias:
            bias = tf.reshape(parameters[:, self.kernel_param_count:], [-1, self.output_units])
        else:
            bias = None

        if self.hyper_net.use_bias:
            return kernel, bias
        else:
            return kernel

    @tf.function
    def inner_call(self, inputs, params, **kwargs):
        """
        A convenience method to perform a forward pass over the meta layer.
        Requires the weights generated from the call() method to be passed as `params`.
        
        Args:
            inputs: Input tensors to the meta layer.
            params: Parameters of this meta layer.
        """
        outputs = self.inner_siren(inputs, params=params)
        return outputs


tf.keras.utils.get_custom_objects().update({
    'hyper_he_normal': HyperHeNormalInitializer,
})

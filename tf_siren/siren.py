import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.init_ops_v2 import _compute_fans


class SIRENFirstLayerInitializer(tf.keras.initializers.Initializer):

    def __init__(self, scale=1.0, seed=None):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, fan_out = _compute_fans(shape)
        limit = self.scale / max(1.0, float(fan_in))
        return tf.random.uniform(shape, -limit, limit, seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'scale': self.scale,
            'seed': self.seed
        }
        return dict(list(base_config.items()) + list(config.items()))


class SIRENInitializer(tf.keras.initializers.VarianceScaling):

    def __init__(self, w0: float = 1.0, c: float = 6.0, seed: int = None):
        # Uniform variance scaler multiplies by 3.0 for limits, so scale down here to compensate
        self.w0 = w0
        self.c = c
        scale = c / (3.0 * w0 * w0)
        super(SIRENInitializer, self).__init__(scale=scale, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'c': self.c
        }
        return dict(list(base_config.items()) + list(config.items()))


class Sine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        """
        Sine activation function with w0 scaling support.

        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(Sine, self).__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)
    
    def get_config(self):
        config = {'w0': self.w0}
        base_config = super(Sine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinusodialRepresentationDense(tf.keras.layers.Dense):

    def __init__(self,
                 units,
                 w0: float = 1.0,
                 c: float = 6.0,
                 activation='sine',
                 use_bias=True,
                 kernel_initializer='siren_uniform',
                 bias_initializer='he_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Sine Representation Dense layer. Extends Dense layer.

        Constructs weights which support `w0` scaling per layer along with change to `c`
        from the paper "Implicit Neural Representations with Periodic Activation Functions".

        Args:
            units: Positive integer, dimensionality of the output space.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            c: Recommended value to scale the distribution when initializing
                weights.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        self.w0 = float(w0)
        self.c = float(c)

        if activation == 'sine':
            activation = Sine(w0=w0)

        if kernel_initializer == 'siren_uniform':
            kernel_initializer = SIRENInitializer(w0=w0, c=c)

        if bias_initializer == 'siren_uniform':
            bias_initializer = SIRENInitializer(w0=w0, c=c)

        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        base_config = super(SinusodialRepresentationDense, self).get_config()
        config = {
            'w0': self.w0,
            'c': self.c
        }
        return dict(list(base_config.items()) + list(config.items()))


class ScaledSinusodialRepresentationDense(SinusodialRepresentationDense):

    def __init__(self,
                 units,
                 scale: float = 1.0,
                 w0: float = 1.0,
                 c: float = 6.0,
                 activation='sine',
                 use_bias=True,
                 kernel_initializer='siren_uniform',
                 bias_initializer='he_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Scaled Sine Representation Dense layer. Extends Dense layer.

        Constructs weights which support `w0` scaling per layer along with change to `c`
        from the paper "Implicit Neural Representations with Periodic Activation Functions".

        During forward pass, kernel matrix will be scaled by factor `scale`.
        `scale` should be set somewhere in the range [1, 2], though higher values may be possible
        with low learning rates.

        Note: Increasing `scale` will increase the norm of the gradient to the weights of this layer.
        Be careful with learning rate when changing `scale`.

        Args:
            units: Positive integer, dimensionality of the output space.
            scale: Scale of the kernel matrix during the forward pass.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            c: Recommended value to scale the distribution when initializing
                weights.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        self.scale = scale

        super().__init__(
            units=units,
            w0=w0,
            c=c,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        """
        Overriding the Dense layer call, in order to multiple `kernel` by the factor `w0`
        prior to matmul. This preserves the distribution of the activation,
        while leaving gradients wrt int input of sine neuron unchanged.
        """
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            # Broadcasting is required for the inputs.
            # [W0 multiplication here !]
            outputs = tf.tensordot(inputs, self.scale * self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = tf.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                # [W0 multiplication here !]
                outputs = tf.sparse.sparse_dense_matmul(inputs, self.scale * self.kernel)
            else:
                # [W0 multiplication here !]
                outputs = tf.matmul(inputs, self.scale * self.kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable

        return outputs


tf.keras.utils.get_custom_objects().update({
    'sine': Sine,
    'siren_uniform': SIRENInitializer,
    'siren_first_uniform': SIRENFirstLayerInitializer,
    'SinusodialRepresentationDense': SinusodialRepresentationDense,
    'ScaledSinusodialRepresentationDense': ScaledSinusodialRepresentationDense,
})

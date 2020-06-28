import tensorflow as tf

from tf_siren import siren_mlp
from tf_siren.encoder import SetEncoder
from tf_siren.meta import meta_siren_mlp


class NeuralProcessHyperNet(tf.keras.Model):

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 siren_units: int = 256,
                 hyper_units: int = 256,
                 latent_dim: int = 256,
                 num_siren_layers: int = 3,
                 num_hyper_layers: int = 1,
                 num_encoder_layers: int = 2,
                 encoder_activation: str = 'sine',
                 hyper_activation: str = 'relu',
                 final_activation: str = 'linear',
                 siren_w0: float = 30.0,
                 siren_w0_initial: float = 30.0,
                 encoder_w0: float = 30.0,
                 use_bias: bool = True,
                 lambda_embedding: float = 0.1,
                 lambda_hyper: float = 100.0,
                 lambda_mse: float = 1.0,
                 hypernet_param_count=None,
                 **kwargs):
        """
        SIREN HyperNetwork from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661).
        Constructs a Hyper Network to represent the parameter space of the entire dataset it encodes.

        Uses a SetEncoder from the above paper for encoder embeddings, and a multi layer meta network
        to generate parameters for a SIREN Network.

        Args:
            input_units: Positive integer, dimensionality of the input space.
                Generally number of coordinates.
            output_units: Positive integer, dimensionality of the output space.
                Generally number of channels of the output tensor.
            siren_units: Positive integer, dimensionality of the SIREN layer.
            hyper_units: Positive integer, dimensionality of the Hyper network layer.
            latent_dim: Positive integer, dimensionality of the latent space of the SetEncoder.
            num_siren_layers: Number of layers that each SIREN network should have,
                per hyper network.
            num_hyper_layers: Number of layers that each hyper network should have,
                to model the SIREN network.
            num_encoder_layers: Number of layers that the SetEncoder model should have.
            encoder_activation: Activation function for the SetEncoder layers.
            hyper_activation: Activation function for the Hyper Network layers.
            final_activation: Activation function for the final layer of the meta SIREN network.
            siren_w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            siren_w0_initial: By default, scales `w0` of first layer to 30 (as used in the paper).
            encoder_w0:  w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            use_bias: Boolean whether to use bias or not.
            lambda_embedding: Loss weight for the L2 regularization of the embedding.
            lambda_hyper: Loss weight for the L2 regularization of the hyper network weights.
            lambda_mse: Loss weight for the pixel wise MSE loss.
            hypernet_param_count: Optional, parameter count for the hyper network.
                Used for loss scaling of the L2 regularizatiion for the hyper network as follows:
                L_hyper = lambda_hyper / hypernet_param_count * sum(L2 norm of parameters)
                If left as None, computed later from the underlying model parameter count.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        super().__init__(**kwargs)

        self.lambda_embedding = lambda_embedding
        self.lambda_hyper = lambda_hyper
        self.lambda_mse = lambda_mse
        self.hypernet_param_count = hypernet_param_count

        self.hyper_net = meta_siren_mlp.MetaSIRENModel(
            input_units=input_units, hidden_units=siren_units, final_units=output_units, hyper_units=hyper_units,
            final_activation=final_activation, hyper_activation=hyper_activation,
            w0=siren_w0, w0_initial=siren_w0_initial,
            num_layers=num_siren_layers, num_hyper_layers=num_hyper_layers,
            use_bias=use_bias,
        )

        self.set_encoder = SetEncoder(
            output_units=latent_dim, hidden_units=latent_dim, num_hidden_layers=num_encoder_layers,
            activation=encoder_activation, w0=encoder_w0, use_bias=use_bias
        )

    @tf.function
    def call(self, inputs, training=None, mask=None):
        coords, _ = inputs

        embeddings = self.set_encoder(inputs)

        param_list = self.hyper_net(embeddings)

        decoded_images = self.hyper_net.inner_call(coords, param_list)
        return decoded_images, embeddings

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            coords, pixels = data
            decoded_images, embeddings = self.call(data)
            image_loss = self.loss(y_true=pixels, y_pred=decoded_images)
            loss = self.lambda_mse * image_loss

            # embedding loss
            embedding_loss = self.lambda_embedding / self.set_encoder.latent_dim * tf.reduce_sum(tf.square(embeddings))
            loss += embedding_loss

            # l2 regularization losses
            reg = []
            params = self.hyper_net.trainable_variables
            for param in params:
                reg.append(tf.reduce_sum(tf.square(param)))

            if self.hypernet_param_count is None:
                self.hypernet_param_count = self.hyper_net.count_params()

            reg_loss = self.lambda_hyper / self.hypernet_param_count * tf.add_n(reg)
            loss += reg_loss

        grads = tape.gradient(loss, self.set_encoder.trainable_variables + self.hyper_net.trainable_variables)
        grads_vars = zip(grads, self.set_encoder.trainable_variables + self.hyper_net.trainable_variables)

        self.optimizer.apply_gradients(grads_vars)

        # Note: `image_loss` is the *unscaled* image loss;
        # I.E. lambda_mse scaling is not applied to it. It is used only for logging.
        return {'loss': loss, 'image_loss': image_loss, 'embedding_loss': embedding_loss, 'reg_loss': reg_loss}

    @tf.function
    def predict_with_context(self, coords, embeddings):
        param_list = self.hyper_net(embeddings)
        decoded_images = self.hyper_net.inner_call(coords, param_list)
        return decoded_images

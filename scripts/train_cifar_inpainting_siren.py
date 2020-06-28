import os
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds

from tf_siren.hypernet import NeuralProcessHyperNet

SAMPLING_RATIO = 200. / 1024.  # use K / 32*32 pixels per batch during training
BATCH_SIZE = 200
EPOCHS = 600
LATENT_DIM = 256

ds, ds_info = tfds.load('cifar10', split='train', with_info=True)  # type: tf.data.Dataset

input_shape = ds_info.features['image'].shape
dataset_len = ds_info.splits['train'].num_examples

rows, cols, channels = input_shape
pixel_count = rows * cols
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)

print("Total pixel samples : ", (sampled_pixel_count))


@tf.function
def build_train_tensors(ds):
    # Build mask and mask idx
    img_mask_x = tf.random.uniform([sampled_pixel_count], maxval=rows, dtype=tf.int32)
    img_mask_y = tf.random.uniform([sampled_pixel_count], maxval=cols, dtype=tf.int32)

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y], axis=-1)  # used for gather_nd
    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)  # used as input vector
    img_mask = 2.0 * img_mask - 1.0  # scale to range [-1, 1]

    img = tf.cast(ds['image'], tf.float32) / 255.
    img = tf.gather_nd(img, img_mask_idx, batch_dims=0)
    return img_mask, img


ds = ds.map(build_train_tensors, num_parallel_calls=2 * os.cpu_count())
ds = ds.shuffle(dataset_len)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

# Build model
model = NeuralProcessHyperNet(
    input_units=2, output_units=channels,  # number of inputs and outputs
    siren_units=LATENT_DIM, hyper_units=LATENT_DIM, latent_dim=LATENT_DIM,  # number of units
    num_siren_layers=3, num_hyper_layers=1, num_encoder_layers=2,  # number of layers
    encoder_activation='sine', hyper_activation='relu', final_activation='sigmoid',  # activations
    lambda_embedding=0.1, lambda_hyper=100., lambda_mse=100.0,  # Loss scaling
)

# instantiate model
dummy_input = [tf.zeros([BATCH_SIZE, sampled_pixel_count, 2]), tf.zeros([BATCH_SIZE, sampled_pixel_count, 3])]
_ = model(dummy_input)

model.summary()

BATCH_SIZE = min(BATCH_SIZE, sampled_pixel_count)
num_steps = int(dataset_len * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.00005, decay_steps=num_steps, end_learning_rate=0.00002,
                                                              power=2.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


@tf.function
def loss_func(y_true, y_pred):
    # Note: This loss is slightly different from the paper.
    # Note: This loss is MSE * channels. To compute true MSE, divide the loss value by number of channels.
    diff = 1.0 / (sampled_pixel_count) * (tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2]))
    diff = tf.reduce_mean(diff)
    return diff


model.compile(optimizer, loss=loss_func, run_eagerly=False)

checkpoint_dir = 'checkpoints/cifar10/inpainting/'
checkpoint_path = checkpoint_dir + 'model'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


if os.path.exists(checkpoint_dir + 'checkpoint'):
    print("Loaded weights for continued training !")
    model.load_weights(checkpoint_path)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('../logs/cifar10/inpainting/', timestamp)

if not os.path.exists(logdir):
    os.makedirs(logdir)

callbacks = [
    # Select lowest pixel mse loss as checkpoint saver.
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + 'model', monitor='image_loss', verbose=0,
                                       save_best_only=True, save_weights_only=True, mode='min'),
    tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch='500,520')
]

model.fit(ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

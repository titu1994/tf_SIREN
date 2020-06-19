import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tf_siren.mixed_prec import siren_mlp

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


SAMPLING_RATIO = 0.1
BATCH_SIZE = 8192
EPOCHS = 5000

# Image Reference - http://earthsongtiles.com/celtic_tiles.html
img_filepath = 'data/celtic_spiral_knot.jpg'
img_raw = tf.io.read_file(img_filepath)
img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)

rows, cols, channels = img_ground_truth.shape
pixel_count = rows * cols
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)


def build_train_tensors():
    img_mask_x = tf.random.uniform([sampled_pixel_count], maxval=rows, seed=0, dtype=tf.int32)
    img_mask_y = tf.random.uniform([sampled_pixel_count], maxval=cols, seed=1, dtype=tf.int32)

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_train = tf.gather_nd(img_ground_truth, img_mask_idx, batch_dims=0)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)

    return img_mask, img_train


img_mask, img_train = build_train_tensors()
img_mask = tf.cast(img_mask, tf.float16)
img_train = tf.cast(img_train, tf.float16)

train_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Build model
model = siren_mlp.SIRENModel(units=256, final_units=3, final_activation='sigmoid', num_layers=5,
                             w0=1.0, w0_initial=30.0, dtype=policy)

# instantiate model
_ = model(tf.zeros([1, 2]))

model.summary()

BATCH_SIZE = min(BATCH_SIZE, len(img_mask))
num_steps = int(len(img_mask) * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.0002, decay_steps=num_steps, end_learning_rate=5e-5,
                                                              power=2.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer, loss='mse')

checkpoint_dir = 'checkpoints/siren_mixed_prec/inpainting/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('logs/siren_mixed_prec/inpainting/', timestamp)

if not os.path.exists(logdir):
    os.makedirs(logdir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + 'model', monitor='loss', verbose=0,
                                       save_best_only=True, save_weights_only=True, mode='min'),
    tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch=20)
]

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)

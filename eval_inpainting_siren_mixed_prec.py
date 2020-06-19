import glob
import numpy as np
import matplotlib.pyplot as plt
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

BATCH_SIZE = 8192

# Image Reference - http://earthsongtiles.com/celtic_tiles.html
img_filepath = 'data/celtic_spiral_knot.jpg'
img_raw = tf.io.read_file(img_filepath)
img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)

rows, cols, channels = img_ground_truth.shape
pixel_count = rows * cols


def build_eval_tensors():
    img_mask_x = tf.range(0, rows, dtype=tf.int32)
    img_mask_y = tf.range(0, cols, dtype=tf.int32)

    img_mask_x, img_mask_y = tf.meshgrid(img_mask_x, img_mask_y, indexing='ij')

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_mask = tf.reshape(img_mask, [-1, 2])

    img_train = tf.reshape(img_ground_truth, [-1, 3])

    return img_mask, img_train


img_mask, img_eval = build_eval_tensors()

test_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_eval))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Build model
model = siren_mlp.SIRENModel(units=256, final_units=3, final_activation='sigmoid', num_layers=5,
                             w0=1.0, w0_initial=30.0, dtype=policy)

# Restore model
checkpoint_path = 'checkpoints/siren_mixed_prec/inpainting/model'
if len(glob.glob(checkpoint_path + "*.index")) == 0:
    raise FileNotFoundError("Model checkpoint not found !")

# instantiate model
_ = model(tf.zeros([1, 2]))

# load checkpoint
model.load_weights(checkpoint_path).expect_partial()  # skip optimizer loading

predicted_image = model.predict(test_dataset, batch_size=BATCH_SIZE, verbose=1)
predicted_image = predicted_image.reshape((rows, cols, channels))  # type: np.ndarray
predicted_image = predicted_image.clip(0.0, 1.0)

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.imshow(img_ground_truth.numpy())
plt.title("Ground Truth Image")

plt.sca(axes[1])
plt.imshow(predicted_image)
plt.title("Predicted Image")

fig.tight_layout()
plt.show()

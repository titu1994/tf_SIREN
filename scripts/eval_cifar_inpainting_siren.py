import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tf_siren.hypernet import NeuralProcessHyperNet

SAMPLED_PIXELS = 1000  # number of pixels to use as context to predict of entire image
LATENT_DIM = 256  # latent dimension of the encoder, decoder and hyper networks

NUM_PLOT_IMAGES = 5  # Number of test images to plot

# Use test set for evaluation
ds, ds_info = tfds.load('cifar10', split='test', with_info=True)  # type: tf.data.Dataset

input_shape = ds_info.features['image'].shape
dataset_len = ds_info.splits['test'].num_examples

rows, cols, channels = input_shape
pixel_count = rows * cols

# Build global mask and mask idx once only
img_mask_x = tf.range(0, rows, dtype=tf.int32)
img_mask_y = tf.range(0, cols, dtype=tf.int32)

img_mask_x, img_mask_y = tf.meshgrid(img_mask_x, img_mask_y, indexing='ij')

img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)
img_mask = tf.reshape(img_mask, [-1, 2])

img_mask = 2.0 * img_mask - 1.0  # scale to range [-1, 1]

# Subsample pixels
np.random.seed(0)
indices = np.asarray(list(range(pixel_count)), dtype=np.int32)
pixel_ids = np.random.choice(indices, size=SAMPLED_PIXELS, replace=False)

pixel_ids = tf.expand_dims(pixel_ids, axis=-1)
subsample_mask = tf.gather_nd(img_mask, pixel_ids, batch_dims=0)  # the subsampled pixel mas
subsample_mask = tf.expand_dims(subsample_mask, axis=0)  # [add batch dimension]


@tf.function
def build_eval_tensors(ds):
    img = tf.cast(ds['image'], tf.float32) / 255.
    img = tf.reshape(img, [-1, channels])
    return img_mask, img


ds = ds.map(build_eval_tensors, num_parallel_calls=2 * os.cpu_count())
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
dummy_input = [tf.zeros([1, pixel_count, 2]), tf.zeros([1, pixel_count, 3])]
_ = model(dummy_input)

model.summary()

checkpoint_dir = 'checkpoints/cifar10/inpainting/'
checkpoint_path = checkpoint_dir + 'model'

if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError("Checkpoint directory not found !")

model.load_weights(checkpoint_path).assert_existing_objects_matched()


# Predict pixels of the different images
eval_images = [None] * NUM_PLOT_IMAGES
output_images = [None] * NUM_PLOT_IMAGES
img_save_idx = -1

mean_squared_errors = []

print()
print("Performing evaluation on test set ...")
with tqdm.tqdm(total=dataset_len) as progbar:
    for global_idx, img_gt in ds:
        # Get random selected pixels out of 1024 pixels
        subsampled_image = tf.gather_nd(img_gt, pixel_ids, batch_dims=0)

        # Add batch dimension to ground truth and subsampled images
        subsampled_image = tf.expand_dims(subsampled_image, axis=0)

        # Obtain the encoder context vector from the subsampled images
        _, embeddings = model.predict([subsample_mask, subsampled_image], batch_size=1, verbose=0)

        # Predict the full image using the subsampled image context vector
        global_idx = tf.expand_dims(global_idx, axis=0)  # [add batch dim]
        predicted_image = model.predict_with_context(global_idx, embeddings)

        # Convert to single image for mse calculation
        predicted_image = predicted_image.numpy()
        predicted_image = predicted_image.reshape((rows, cols, channels))  # type: np.ndarray
        predicted_image = predicted_image.clip(0.0, 1.0)

        # Convert ground truth to compare with above predicted image
        img_gt = tf.reshape(img_gt, [rows, cols, channels]).numpy()

        # Compute mse
        mse = np.mean((img_gt - predicted_image) ** 2)
        mean_squared_errors.append(mse)

        # Save last 5 images from test set to plot
        img_save_idx = (img_save_idx + 1) % NUM_PLOT_IMAGES
        eval_images[img_save_idx] = img_gt
        output_images[img_save_idx] = predicted_image

        progbar.update(1)

mse = np.asarray(mean_squared_errors).mean()
print("Dataset MSE : ", mse)  # should be close to 0.009 for trained model checkpoint

# Plot the last 5 images from the test set
fig, axes = plt.subplots(len(output_images), 2)

for ix, (ground_truth_img, predicted_img) in enumerate(zip(eval_images, output_images)):
    plt.sca(axes[ix, 0])
    gt_img = ground_truth_img
    gt_img = gt_img.clip(0.0, 1.0)
    plt.imshow(gt_img)
    plt.title("Ground Truth Image")

    plt.sca(axes[ix, 1])
    plt.imshow(predicted_img)
    plt.title("Predicted Image")

fig.tight_layout()
plt.show()

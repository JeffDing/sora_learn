import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Image preprocessing

def read_image(image_file="./SDXL.png", scale=True, image_dim=336):

    image = tf.keras.utils.load_img(
        image_file, grayscale=False, color_mode='rgb', target_size=None,
        interpolation='nearest'
    )
    image_arr_orig = tf.keras.preprocessing.image.img_to_array(image)
    if(scale):
        image_arr_orig = tf.image.resize(
            image_arr_orig, [image_dim, image_dim],
            method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False
        )
    image_arr = tf.image.crop_to_bounding_box(
        image_arr_orig, 0, 0, image_dim, image_dim
    )

    return image_arr

# Patching
def create_patches(image):
    im = tf.expand_dims(image, axis=0)
    patches = tf.image.extract_patches(
        images=im,
        sizes=[1, 32, 32, 1],
        strides=[1, 32, 32, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [1, -1, patch_dims])

    return patches

image_arr = read_image()
patches = create_patches(image_arr)

# Drawing
def render_image_and_patches(image, patches):
    plt.figure(figsize=(16, 16))
    plt.suptitle(f"Cropped Image", size=48)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis("off")
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(16, 16))
    plt.suptitle(f"Image Patches", size=24)
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (32, 32, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")
    plt.savefig('render_image_and_patches.png')

def render_flat(patches):
    plt.figure(figsize=(32, 2))
    plt.suptitle(f"Flattened Image Patches", size=24)
    n = int(np.sqrt(patches.shape[1]))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(1, 101, i+1)
        patch_img = tf.reshape(patch, (32, 32, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")
        if(i == 100):
            break
    plt.savefig('render_flat.png')


render_image_and_patches(image_arr, patches)
render_flat(patches)
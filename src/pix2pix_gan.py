import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import pickle
from constants import LAMBDA, EPOCHS, IM_CHANNEL, IM_SIZE, DATA_FOLDER, \
    ALPHA_LEACKY_RELU
import time

# READ IMAGES HERE
with open('../day_paths.pickle', 'rb') as f:
    DAY_PATH = pickle.load(f)
with open('../night_paths.pickle', 'rb') as f:
    NIGHT_PATH = pickle.load(f)

inp = cv2.imread(f'{DATA_FOLDER}00000850/day/20151101_142506.jpg')
inp = cv2.resize(inp, (IM_SIZE, IM_SIZE))
tar = cv2.imread(f'{DATA_FOLDER}00000850/night/20151101_072507.jpg')
tar = cv2.resize(tar, (IM_SIZE, IM_SIZE))


def Batch(day_paths, night_paths):
    N = len(day_paths)
    for i in range(N):
        day = cv2.imread(f'{DATA_FOLDER}{day_paths[i]}')
        night = cv2.imread(f'{DATA_FOLDER}{night_paths[i]}')
        day = cv2.resize(day, (IM_SIZE, IM_SIZE))
        night = cv2.resize(night, (IM_SIZE, IM_SIZE))
        yield day, night


def image_normalization_mapping(image, from_min, from_max, to_min, to_max):
    """
    Function to scale image to required range.
    Map data from any interval [from_min, from_max] --> [to_min, to_max]
    Used to normalize and denormalize images
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype='f')
    return to_min + (scaled * to_range)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(alpha=ALPHA_LEACKY_RELU))

    return result


def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(IM_CHANNEL, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, IM_CHANNEL],
                                name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, IM_CHANNEL],
                                name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(
        zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU(alpha=ALPHA_LEACKY_RELU)(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(
        zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


generator = Generator()
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                 disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output],
                                              training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for d, n in Batch(DAY_PATH, NIGHT_PATH):
            # day, night = d/255, n/255
            day = image_normalization_mapping(d, 0, 255, -1, 1)
            night = image_normalization_mapping(n, 0, 255, -1, 1)

            day = tf.convert_to_tensor(day, dtype='float32')
            night = tf.convert_to_tensor(night, dtype='float32')

            day = tf.expand_dims(day, 0)
            night = tf.expand_dims(night, 0)

            train_step(night, day)

        inp1 = image_normalization_mapping(inp, 0, 255, -1, 1)
        tar1 = image_normalization_mapping(tar, 0, 255, -1, 1)

        inp1 = tf.convert_to_tensor(inp1, dtype='float32')
        tar1 = tf.convert_to_tensor(tar1, dtype='float32')

        inp1 = tf.expand_dims(inp1, 0)
        tar1 = tf.expand_dims(tar1, 0)

        generate_images(generator, tar1, inp1)


train(EPOCHS)

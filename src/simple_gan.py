import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import pickle
from constants import LATENT_SIZE

tf.reset_default_graph()

# READ IMAGES HERE
DATA_FOLDER = '../data/Image/'
IM_SIZE = 256
IM_CHANNEL = 3
with open('../day_paths.pickle', 'rb') as f:
    DAY_PATH = pickle.load(f)
with open('../night_paths.pickle', 'rb') as f:
    NIGHT_PATH = pickle.load(f)


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


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
    Map data from any interval [from_min, from_max] --> [to_min, to_max]
    Used to normalize and denormalize images
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


tf.reset_default_graph()


def generator(Z, reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(Z, 512, [4, 4], strides=(2, 2),
                                           padding='same')
        bn1 = lrelu(tf.layers.batch_normalization(conv1), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(bn1, 512, [4, 4], strides=(2, 2),
                                           padding='same')
        bn2 = lrelu(tf.layers.batch_normalization(conv2), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(bn2, 512, [4, 4], strides=(2, 2),
                                           padding='same')
        bn3 = lrelu(tf.layers.batch_normalization(conv3), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(bn3, 512, [4, 4], strides=(2, 2),
                                           padding='same')
        bn4 = lrelu(tf.layers.batch_normalization(conv4), 0.2)

        conv5 = tf.layers.conv2d_transpose(bn4, 256, [4, 4], strides=(2, 2),
                                           padding='same')
        bn5 = lrelu(tf.layers.batch_normalization(conv5), 0.2)

        conv6 = tf.layers.conv2d_transpose(bn5, 128, [4, 4], strides=(2, 2),
                                           padding='same')
        bn6 = lrelu(tf.layers.batch_normalization(conv6), 0.2)

        conv7 = tf.layers.conv2d_transpose(bn6, 64, [4, 4], strides=(2, 2),
                                           padding='same',
                                           activation=tf.nn.leaky_relu)

        # output layer
        conv8 = tf.layers.conv2d_transpose(conv7, 3, [4, 4], strides=(2, 2),
                                           padding='same')
        o = tf.nn.tanh(conv8)

    return o


def discriminator(X, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        conv1 = tf.layers.conv2d(X, 64, [4, 4], strides=(2, 2))

        conv2 = tf.layers.conv2d(conv1, 128, [4, 4], strides=(2, 2))
        bn2 = lrelu(tf.layers.batch_normalization(conv2), 0.2)

        conv3 = tf.layers.conv2d(bn2, 256, [4, 4], strides=(2, 2))
        bn3 = lrelu(tf.layers.batch_normalization(conv3), 0.2)

        conv4 = tf.layers.conv2d(bn3, 512, [4, 4], strides=(2, 2))
        bn4 = lrelu(tf.layers.batch_normalization(conv4), 0.2)

        conv5 = tf.layers.conv2d(bn4, 512, [4, 4], strides=(2, 2))
        bn5 = lrelu(tf.layers.batch_normalization(conv5), 0.2)

        conv6 = tf.layers.conv2d(bn5, 512, [4, 4], strides=(2, 2))
        bn6 = lrelu(tf.layers.batch_normalization(conv6), 0.2)

        logits = tf.layers.conv2d(bn6, 1, [2, 2], strides=(1, 1))

        o = tf.nn.sigmoid(logits)

    return o, logits


X = tf.placeholder(tf.float32, [None, IM_SIZE, IM_SIZE, IM_CHANNEL])
Z = tf.placeholder(tf.float32, [None, 1, 1, LATENT_SIZE])

G_sample = generator(Z)
print(G_sample)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

disc_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,
                                            labels=tf.ones_like(
                                                r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(
        logits=f_logits, labels=tf.zeros_like(f_logits)))

gen_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,
                                            labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                             scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope="GAN/Discriminator")

gen_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_loss,
                                                                var_list=gen_vars)  # G Train step
disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(disc_loss,
                                                                 var_list=disc_vars)  # D Train step


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(11):
            for d, n in Batch(DAY_PATH, NIGHT_PATH):
                # day, night = d/255, n/255
                day = image_normalization_mapping(d, 0, 255, -1, 1)
                night = image_normalization_mapping(n, 0, 255, -1, 1)

                Z_batch = np.random.normal(size=LATENT_SIZE)
                Z_batch = np.expand_dims(Z_batch, axis=0)
                Z_batch = np.expand_dims(Z_batch, axis=0)
                Z_batch = np.expand_dims(Z_batch, axis=0)

                _, dloss = sess.run([disc_step, disc_loss],
                                    feed_dict={X: day[np.newaxis, :],
                                               Z: Z_batch})
                _, gloss = sess.run([gen_step, gen_loss],
                                    feed_dict={Z: Z_batch})

                Z_batch = np.random.normal(size=LATENT_SIZE)
                Z_batch = np.expand_dims(Z_batch, axis=0)
                Z_batch = np.expand_dims(Z_batch, axis=0)
                Z_batch = np.expand_dims(Z_batch, axis=0)

                generated = sess.run([G_sample], feed_dict={Z: Z_batch})

            if i % 5 == 0:
                generated = image_normalization_mapping(generated[0][0], -1, 1,
                                                        0, 255)
                plt.imshow(generated)
                plt.show()

            print(
                "Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (
                    i, dloss, gloss))


train()

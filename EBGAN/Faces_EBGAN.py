from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, inspect

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import TensorflowUtils as utils
import Dataset_Reader.read_celebADataset as celebA
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_EBGAN_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/CelebA_faces/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_bool("pt", "False", "Include pull away loss term")
tf.flags.DEFINE_integer("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("mode", "train", "train - visualize")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 800
MAX_ITERATIONS = int(1e5 + 1)
PT_LOSS_WEIGHT = 0.1
MODEL_IMAGE_SIZE = 108
IMAGE_SIZE = 64
NUM_OF_CHANNELS = 3
GEN_DIMENSION = 16
DISC_DIMENSION = 32


def _read_input(filename_queue):
    class DataRecord(object):
        pass

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    record = DataRecord()
    decoded_image = tf.image.decode_jpeg(value, channels=NUM_OF_CHANNELS)
    decoded_image_4d = tf.expand_dims(decoded_image, 0)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, [IMAGE_SIZE, IMAGE_SIZE])
    record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
    cropped_image = tf.cast(tf.image.crop_to_bounding_box(decoded_image, 55, 35, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
                            tf.float32)
    decoded_image_4d = tf.expand_dims(cropped_image, 0)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, [IMAGE_SIZE, IMAGE_SIZE])
    record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
    return record


def read_input_queue(filename_queue):
    read_input = _read_input(filename_queue)
    num_preprocess_threads = 4
    min_queue_examples = int(0.1 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print("Shuffling")
    input_image = tf.train.shuffle_batch([read_input.input_image],
                                         batch_size=FLAGS.batch_size,
                                         num_threads=num_preprocess_threads,
                                         capacity=min_queue_examples + 2 * FLAGS.batch_size,
                                         min_after_dequeue=min_queue_examples)
    input_image = input_image / 127.5 - 1
    return input_image


def generator(z, train_mode):
    """
    Conv Generator graph definition
    :param z: input latent variable
    :param train_mode: True/ False for batch normalization
    :return:
    """
    with tf.variable_scope("generator") as scope:
        W_1 = utils.weight_variable([FLAGS.z_dim, 64 * GEN_DIMENSION // 2 * IMAGE_SIZE // 16 * IMAGE_SIZE // 16],
                                    name="W_1")
        b_1 = utils.bias_variable([64 * GEN_DIMENSION // 2 * IMAGE_SIZE // 16 * IMAGE_SIZE // 16], name="b_0")
        z_1 = tf.matmul(z, W_1) + b_1
        h_1 = tf.reshape(z_1, [-1, IMAGE_SIZE // 16, IMAGE_SIZE // 16, 64 * GEN_DIMENSION // 2])
        h_bn1 = utils.batch_norm(h_1, 64 * GEN_DIMENSION // 2, train_mode, scope="gen_bn1")
        h_relu1 = tf.nn.relu(h_bn1, name='relu1')
        utils.add_activation_summary(h_relu1)

        W_2 = utils.weight_variable([5, 5, 64 * GEN_DIMENSION // 4, 64 * GEN_DIMENSION // 2],
                                    name="W_2")
        b_2 = utils.bias_variable([64 * GEN_DIMENSION // 4], name="b_2")
        deconv_shape = tf.pack([tf.shape(h_relu1)[0], IMAGE_SIZE // 8, IMAGE_SIZE // 8, 64 * GEN_DIMENSION // 4])
        h_conv_t2 = utils.conv2d_transpose_strided(h_relu1, W_2, b_2, output_shape=deconv_shape)
        h_bn2 = utils.batch_norm(h_conv_t2, 64 * GEN_DIMENSION // 4, train_mode, scope="gen_bn2")
        h_relu2 = tf.nn.relu(h_bn2, name='relu2')
        utils.add_activation_summary(h_relu2)

        W_3 = utils.weight_variable([5, 5, 64 * GEN_DIMENSION // 8, 64 * GEN_DIMENSION // 4],
                                    name="W_3")
        b_3 = utils.bias_variable([64 * GEN_DIMENSION // 8], name="b_3")
        deconv_shape = tf.pack([tf.shape(h_relu2)[0], IMAGE_SIZE // 4, IMAGE_SIZE // 4, 64 * GEN_DIMENSION // 8])
        h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3, output_shape=deconv_shape)
        h_bn3 = utils.batch_norm(h_conv_t3, 64 * GEN_DIMENSION // 8, train_mode, scope="gen_bn3")
        h_relu3 = tf.nn.relu(h_bn3, name='relu3')
        utils.add_activation_summary(h_relu3)

        W_4 = utils.weight_variable([5, 5, 64 * GEN_DIMENSION // 16, 64 * GEN_DIMENSION // 8],
                                    name="W_4")
        b_4 = utils.bias_variable([64 * GEN_DIMENSION // 16], name="b_4")
        deconv_shape = tf.pack([tf.shape(h_relu3)[0], IMAGE_SIZE // 2, IMAGE_SIZE // 2, 64 * GEN_DIMENSION // 16])
        h_conv_t4 = utils.conv2d_transpose_strided(h_relu3, W_4, b_4, output_shape=deconv_shape)
        h_bn4 = utils.batch_norm(h_conv_t4, 64 * GEN_DIMENSION // 16, train_mode, scope="gen_bn4")
        h_relu4 = tf.nn.relu(h_bn4, name='relu4')
        utils.add_activation_summary(h_relu4)

        W_5 = utils.weight_variable([5, 5, NUM_OF_CHANNELS, 64 * GEN_DIMENSION // 16], name="W_5")
        b_5 = utils.bias_variable([NUM_OF_CHANNELS], name="b_5")
        deconv_shape = tf.pack([tf.shape(h_relu4)[0], IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS])
        h_conv_t5 = utils.conv2d_transpose_strided(h_relu4, W_5, b_5, output_shape=deconv_shape)
        pred_image = tf.nn.tanh(h_conv_t5, name='pred_image')
        utils.add_activation_summary(pred_image)

    return pred_image


def encoder(input_images, train_mode):
    """
    Encoder graph definition
    :param train_mode:
    :param input_images: tensor of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS]
    :return: embedding
    """
    W_1 = utils.weight_variable([4, 4, NUM_OF_CHANNELS, DISC_DIMENSION * 1], stddev=0.002, name="enc_W_1")
    b_1 = utils.bias_variable([DISC_DIMENSION * 1], name="enc_b_1")
    h_conv1 = utils.conv2d_strided(input_images, W_1, b_1)
    h_bn1 = utils.batch_norm(h_conv1, DISC_DIMENSION * 1, train_mode, scope='enc_bn1')
    h_relu1 = utils.leaky_relu(h_bn1, name="enc_relu1")
    utils.add_activation_summary(h_relu1)

    W_2 = utils.weight_variable([4, 4, DISC_DIMENSION * 1, DISC_DIMENSION * 2], stddev=0.002, name="enc_W_2")
    b_2 = utils.bias_variable([DISC_DIMENSION * 2], name="enc_b_2")
    h_conv2 = utils.conv2d_strided(h_relu1, W_2, b_2)
    h_bn2 = utils.batch_norm(h_conv2, DISC_DIMENSION * 2, train_mode, scope='enc_bn2')
    h_relu2 = utils.leaky_relu(h_bn2, name="enc_relu2")
    utils.add_activation_summary(h_relu2)

    W_3 = utils.weight_variable([4, 4, DISC_DIMENSION * 2, DISC_DIMENSION * 4], stddev=0.002, name="enc_W_3")
    b_3 = utils.bias_variable([DISC_DIMENSION * 4], name="enc_b_3")
    h_conv3 = utils.conv2d_strided(h_relu2, W_3, b_3)
    h_bn3 = utils.batch_norm(h_conv3, DISC_DIMENSION * 4, train_mode, scope='enc_bn3')
    h_relu3 = utils.leaky_relu(h_bn3, name="enc_relu3")
    utils.add_activation_summary(h_relu3)
    embedding = tf.reshape(h_relu3, [-1, IMAGE_SIZE // 8 * IMAGE_SIZE // 8 * DISC_DIMENSION * 4])

    return embedding


def decoder(embedding, train_mode):
    """
    Decoder graph definition
    :param train_mode:
    :param embedding: 2d tensor of shape [batch_size, auto encoder bottleneck dimension]
    :return:
    """

    h_reshaped = tf.reshape(embedding, [-1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, DISC_DIMENSION * 4])

    W_1 = utils.weight_variable([4, 4, DISC_DIMENSION * 2, DISC_DIMENSION * 4], stddev=0.002, name="dec_W_1")
    b_1 = utils.bias_variable([DISC_DIMENSION * 2], name="dec_b_1")
    deconv_shape = tf.pack([tf.shape(h_reshaped)[0], IMAGE_SIZE // 4, IMAGE_SIZE // 4, DISC_DIMENSION * 2])
    h_conv_t1 = utils.conv2d_transpose_strided(h_reshaped, W_1, b_1, output_shape=deconv_shape)
    h_bn1 = utils.batch_norm(h_conv_t1, DISC_DIMENSION * 2, train_mode, scope="dec_bn1")
    h_relu1 = utils.leaky_relu(h_bn1, name="dec_relu1")
    utils.add_activation_summary(h_relu1)

    W_2 = utils.weight_variable([4, 4, DISC_DIMENSION * 1, DISC_DIMENSION * 2], stddev=0.002, name="dec_W_2")
    b_2 = utils.bias_variable([DISC_DIMENSION * 1], name="dec_b_2")
    deconv_shape = tf.pack([tf.shape(h_conv_t1)[0], IMAGE_SIZE // 2, IMAGE_SIZE // 2, DISC_DIMENSION * 1])
    h_conv_t2 = utils.conv2d_transpose_strided(h_relu1, W_2, b_2, output_shape=deconv_shape)
    h_bn2 = utils.batch_norm(h_conv_t2, DISC_DIMENSION * 1, train_mode, scope="dec_bn2")
    h_relu2 = utils.leaky_relu(h_bn2, name="dec_relu2")
    utils.add_activation_summary(h_relu2)

    W_3 = utils.weight_variable([4, 4, NUM_OF_CHANNELS, DISC_DIMENSION * 1], stddev=0.002, name="dec_W_3")
    b_3 = utils.bias_variable([NUM_OF_CHANNELS], name="dec_b_3")
    deconv_shape = tf.pack([tf.shape(h_conv_t2)[0], IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS])
    h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3, output_shape=deconv_shape)
    decoded_image = tf.nn.tanh(h_conv_t3, name="decoded_image")
    utils.add_activation_summary(decoded_image)

    return decoded_image


def mse_loss(pred, data):
    loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / FLAGS.batch_size
    return loss_val


def discriminator(input_images, train_mode):
    """
    Discriminator graph definition
    :param input_images: tensor of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS]
    :param train_mode:
    :return:
    """
    embeddings = encoder(input_images, train_mode)
    decoded_images = decoder(embeddings, train_mode)
    mse = mse_loss(decoded_images, input_images)
    return mse, embeddings, decoded_images


def pullaway_loss(embeddings):
    """
    Pull Away loss calculation
    :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
    :return: pull away term loss
    """
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True)
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return pt_loss


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    print("Setting up image reader...")
    train_images, valid_images, test_images = celebA.read_dataset(FLAGS.data_dir)

    filename_queue = tf.train.string_input_producer(train_images)
    images = read_input_queue(filename_queue)
    train_phase = tf.placeholder(tf.bool)
    z_vec = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name="z")

    print("Setting up network model...")
    tf.histogram_summary("z", z_vec)
    # tf.image_summary("image_real", images, max_images=1)
    gen_images = generator(z_vec, train_phase)
    tf.image_summary("image_generated", gen_images, max_images=2)

    with tf.variable_scope("discriminator") as scope:
        discriminator_loss_real, embeddings_real, decoded_real = discriminator(images, train_phase)
        tf.image_summary("decoded_real", decoded_real, max_images=1)
        scope.reuse_variables()
        discrimintator_loss_fake, embeddings_fake, decoded_fake = discriminator(gen_images, train_phase)
        tf.image_summary("decoded_fake", decoded_fake, max_images=1)

    discriminator_loss = FLAGS.margin - discrimintator_loss_fake + discriminator_loss_real
    pt_loss = 0
    if FLAGS.pt:
        print("Adding pull away loss term...")
        # Using all the embeddings for pull away loss - no mini batches
        pt_loss = pullaway_loss(embeddings_fake)
    gen_loss = discrimintator_loss_fake + PT_LOSS_WEIGHT * pt_loss

    tf.scalar_summary("Discriminator_loss_real", discriminator_loss_real)
    tf.scalar_summary("Discrimintator_loss_fake", discrimintator_loss_fake)
    tf.scalar_summary("Discriminator_loss", discriminator_loss)
    tf.scalar_summary("Generator_loss", gen_loss)

    train_variables = tf.trainable_variables()
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # print(map(lambda x: x.op.name, generator_variables))
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    # print(map(lambda x: x.op.name, discriminator_variables))

    generator_train_op = train(gen_loss, generator_variables)
    discriminator_train_op = train(discriminator_loss, discriminator_variables)

    for v in train_variables:
        utils.add_to_regularization_and_summary(var=v)

    def visualize():
        count = 20
        z_feed = np.random.uniform(-1.0, 1.0, size=(count, FLAGS.z_dim)).astype(np.float32)
        # z_feed = np.tile(np.random.uniform(-1.0, 1.0, size=(1, FLAGS.z_dim)).astype(np.float32), (count, 1))
        # z_feed[:, 25] = sorted(10.0 * np.random.randn(count))
        image = sess.run(gen_images, feed_dict={z_vec: z_feed, train_phase: False})

        for iii in xrange(count):
            print(image.shape)
            utils.save_image(image[iii, :, :, :], IMAGE_SIZE, FLAGS.logs_dir, name="gen" + str(iii))
            print("Saving image" + str(iii))

    sess = tf.Session()
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "visualize":
        visualize()
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for itr in xrange(MAX_ITERATIONS):
            batch_z = np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
            feed_dict = {z_vec: batch_z, train_phase: True}

            sess.run(discriminator_train_op, feed_dict=feed_dict)
            sess.run(generator_train_op, feed_dict=feed_dict)
            sess.run(generator_train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                g_loss_val, d_loss_val, summary_str = sess.run([gen_loss, discriminator_loss, summary_op],
                                                               feed_dict=feed_dict)
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=itr)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except KeyboardInterrupt:
        print("Ending Training...")
    finally:
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)


if __name__ == "__main__":
    tf.app.run()

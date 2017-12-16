#!/usr/bin/env python3

import os
import tensorflow as tf

from mult999_constants import (CHARS, DATASET_ID, NUM_IN_CHARS, 
    NUM_IN_FEATURES,  NUM_OUT_CHARS)
from mult999_1st_constants import NUM_OUT_FEATURES
from mult999_1st_mnist_1_constants import BATCH_SIZE, LEARNING_RATE, MODEL_ID

def dataset(filename):

    LOOKUP_TABLE = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(list(CHARS)))
    IN_ONE_HOT_SHAPE = tf.constant([BATCH_SIZE, -1])
    OUT_ONE_HOT_SHAPE = tf.constant([BATCH_SIZE, -1])

    def lines_to_pairs(lines):
        in_ = tf.substr(lines, 0, NUM_IN_CHARS)
        out = tf.substr(lines, NUM_IN_CHARS, NUM_OUT_CHARS)
        return (in_, out)

    def pairs_to_chars(in_, out):
        in_chars = tf.sparse_tensor_to_dense(tf.string_split(in_, ""), "?")
        out_chars = tf.sparse_tensor_to_dense(tf.string_split(out, ""), "?")
        return (in_chars, out_chars)

    def gather_1st(in_chars, out_chars):
        return (in_chars, tf.gather(out_chars, [0], axis = 1))

    def chars_to_one_hot(in_chars, out_chars):

        def one_hot(chars):
            indices = LOOKUP_TABLE.lookup(chars)
            depth = len(CHARS)
            return tf.one_hot(indices, depth)       

        in_one_hot = tf.reshape(tf.map_fn(one_hot, in_chars, tf.float32),
                                IN_ONE_HOT_SHAPE)
        out_one_hot = tf.reshape(tf.map_fn(one_hot, out_chars, tf.float32), 
                                OUT_ONE_HOT_SHAPE)
        return (in_one_hot, out_one_hot)

    print("reading {0}".format(filename))
    return (tf.data.TextLineDataset(filename)
            .batch(BATCH_SIZE)
            .map(lines_to_pairs)
            .map(pairs_to_chars)
            .map(gather_1st)
            .map(chars_to_one_hot)) 

def model():
    x = tf.placeholder(tf.float32, [None, NUM_IN_FEATURES])
    W = tf.Variable(tf.zeros([NUM_IN_FEATURES, NUM_OUT_FEATURES]))
    b = tf.Variable(tf.zeros([NUM_OUT_FEATURES]))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, NUM_OUT_FEATURES])
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y))
    tf.summary.scalar("cross entropy", cross_entropy)
    train_step = (tf.train.GradientDescentOptimizer(LEARNING_RATE)
        .minimize(cross_entropy))
    saver = tf.train.Saver()
    return x, y_, accuracy, train_step, saver

def train(x, y_, train_step, dataset, saver, save_path, writer, merged):
    print("training")
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with tf.Session() as session:
        session.run(iterator.initializer)
        session.run(tf.tables_initializer())
        session.run(tf.global_variables_initializer())
        global_step = 0
        while True:
            try:
                in_one_hot, out_one_hot = session.run(get_next)
                summary, _ = session.run([merged, train_step], 
                    feed_dict = {x: in_one_hot, y_: out_one_hot})
                writer.add_summary(summary, global_step)
                global_step += 1
            except tf.errors.OutOfRangeError:
                break
        print("writing {0}".format(save_path))
        save_path = saver.save(session, save_path)
        print("writing to {0}".format(writer.get_logdir()))

def test(x, y_, dataset, saver, save_path, writer, merged, accuracy):
    print("testing")
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with tf.Session() as session:
        session.run(iterator.initializer)
        session.run(tf.tables_initializer())
        print("reading {0}".format(save_path))
        saver.restore(session, save_path)
        acc_min = float("+inf")
        acc_max = 0
        global_step = 0
        while True:
            try:
                in_one_hot, out_one_hot = session.run(get_next)
                summary, acc = session.run([merged, accuracy],
                   feed_dict = {x: in_one_hot, y_: out_one_hot})
                writer.add_summary(summary, global_step)
                acc_min = min(acc, acc_min)
                acc_max = max(acc, acc_max)
                global_step += 1
            except tf.errors.OutOfRangeError:
                break
        print("accuracy between {0}, {1}".format(acc_min, acc_max))
        print("writing to {0}".format(writer.get_logdir()))

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x, y_, accuracy, train_step, saver = model()

    directory = "datasets"
    filename_train = os.path.join(directory, "{0}.train".format(DATASET_ID))
    filename_test = os.path.join(directory, "{0}.test".format(DATASET_ID))
    dataset_train = dataset(filename_train)
    dataset_test = dataset(filename_test)

    logdir = "summaries"
    logdir_train = os.path.join(logdir, "train")
    logdir_test = os.path.join(logdir, "test")
    writer_train = tf.summary.FileWriter(logdir_train)
    writer_test = tf.summary.FileWriter(logdir_test)
 
    save_path = os.path.join("models", "{0}.checkpoint".format(MODEL_ID))

    merged = tf.summary.merge_all()
    train(x, y_, train_step, dataset_train, saver, save_path, writer_train,
        merged)
    test(x, y_, dataset_test, saver, save_path, writer_test, merged, accuracy)

if __name__ == "__main__":
    main()

import argparse
import mult999.file_utils
import os
import sys
import tensorflow as tf

from mult999.constants import (CHARS, DATASET_ID, NUM_IN_CHARS, NUM_IN_FEATURES,
  NUM_OUT_CHARS)
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io

def parse_args():
    parser = argparse.ArgumentParser(description = "runs trainer")
    parser.add_argument("--job-dir",
        action = "store", metavar = "dir", required = True,
        help = "training output")
    parser.add_argument("--filename-train",
        action = "store", metavar = "filename", required = True,
        help = "training data")
    parser.add_argument("--filename-eval",
        action = "store", metavar = "filename", required = True,
        help = "evaluation data")
    parser.add_argument("--summaries",
        action = "store", metavar = "step", required = False, type = int,
        help = "summaries for TensorBoard")
    parser.add_argument("--profiling",
        action = "store", metavar = "step", required = False, type = int,
        help = "timeline for chrome://tracing")
    dict = vars(parser.parse_args())
    return (
        dict["job_dir"], 
        dict["filename_train"], 
        dict["filename_eval"],
        dict["summaries"],
        dict["profiling"]) 

def check_files(filenames):
    for filename in filenames:
        if not file_io.file_exists(filename):
            sys.exit("{0} does not exist or is not a file".format(filename))

def dataset(filename, batch_size, gather, repeat = 1, buffer_size = None):

    LOOKUP_TABLE = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(list(CHARS)))
    IN_ONE_HOT_SHAPE = tf.constant([batch_size, -1])
    OUT_ONE_HOT_SHAPE = tf.constant([batch_size, -1])

    def lines_to_pairs(lines):
        in_ = tf.substr(lines, 0, NUM_IN_CHARS)
        out = tf.substr(lines, NUM_IN_CHARS, NUM_OUT_CHARS)
        return (in_, out)

    def pairs_to_chars(in_, out):
        in_chars = tf.sparse_tensor_to_dense(tf.string_split(in_, ""), "?")
        out_chars = tf.sparse_tensor_to_dense(tf.string_split(out, ""), "?")
        return (in_chars, out_chars)

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
    dataset = (tf.data.TextLineDataset(filename)
        .batch(batch_size)
        .map(lines_to_pairs)
        .map(pairs_to_chars)
        .map(gather)
        .map(chars_to_one_hot))
    if repeat != 1:
        dataset = dataset.repeat(repeat)
    if buffer_size != None:
        dataset = dataset.shuffle(buffer_size)
    return dataset

def train(session, x, y_, train_step, dataset,
    summaries = None, writer = None, merged = None, 
    profiling = None, run_options = None, run_metadata = None):
    print("training")
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    session.run(iterator.initializer)
    global_step = 0
    while True:
        try:
            in_one_hot, out_one_hot = session.run(get_next)
            if profiling != None and global_step % profiling != 0:
                run_options = None
                run_metadata = None
            if summaries != None and global_step % summaries == 0:
                summary, _ = session.run([merged, train_step], 
                    feed_dict = {x: in_one_hot, y_: out_one_hot},
                    options = run_options, run_metadata = run_metadata)
                writer.add_summary(summary, global_step)
            else:
                session.run(train_step, 
                    feed_dict = {x: in_one_hot, y_: out_one_hot},
                    options = run_options, run_metadata = run_metadata)
            global_step += 1
        except tf.errors.OutOfRangeError:
            break
    if summaries != None:
        print("writing {0}/*".format(writer.get_logdir()))

def eval(session, x, y_, accuracy, dataset, 
    summaries = None, writer = None, merged = None, 
    profiling = None, run_options = None, run_metadata = None):
    print("evaluating")
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    session.run(iterator.initializer)
    acc_min = float("+inf")
    acc_max = 0
    global_step = 0
    while True:
        try:
            in_one_hot, out_one_hot = session.run(get_next)
            if profiling != None and global_step % profiling != 0:
                run_options = None
                run_metadata = None
            if summaries != None and global_step % summaries == 0:
                summary, acc = session.run([merged, accuracy],
                   feed_dict = {x: in_one_hot, y_: out_one_hot},
                   options = run_options, run_metadata = run_metadata)
                writer.add_summary(summary, global_step)
            else:
                acc = session.run(accuracy,
                   feed_dict = {x: in_one_hot, y_: out_one_hot},
                   options = run_options, run_metadata = run_metadata)
            acc_min = min(acc, acc_min)
            acc_max = max(acc, acc_max)
            global_step += 1
        except tf.errors.OutOfRangeError:
            break
    print("accuracy between {0}, {1}".format(acc_min, acc_max))
    if summaries != None:
        print("writing {0}/*".format(writer.get_logdir()))

def main(model_id, model, batch_size, gather, repeat = 1, buffer_size = None):
    print("runtime Tensorflow {0}, Python {1}.{2}".format(
        tf.__version__, sys.version_info[0], sys.version_info[1]));
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    job_dir, filename_train, filename_eval, summaries, profiling = parse_args()
    mult999.file_utils.check_dirs([os.path.join(job_dir, ".")])
    check_files([filename_train, filename_eval])

    x, y_, accuracy, train_step = model()

    dataset_train = dataset(filename_train, batch_size, gather, repeat, 
        buffer_size)
    dataset_eval = dataset(filename_eval, batch_size, gather)

    if summaries != None:
        logdir_train = os.path.join(job_dir, "train")
        logdir_eval = os.path.join(job_dir, "eval")
        writer_train = tf.summary.FileWriter(logdir_train)
        writer_eval = tf.summary.FileWriter(logdir_eval)
        merged = tf.summary.merge_all()
    else:
        writer_train = None
        writer_eval = None
        merged = None
    if profiling != None:
        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as session:
        session.run(tf.tables_initializer())
        session.run(tf.global_variables_initializer())
        train(session, x, y_, train_step, dataset_train, 
            summaries, writer_train, merged,
            profiling, run_options, run_metadata)
        eval(session, x, y_, accuracy, dataset_eval, 
            summaries, writer_eval, merged,
            profiling, run_options, run_metadata)

    if summaries != None:
        writer_train.close()
        writer_eval.close()
    if profiling != None:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        filename_timeline = os.path.join(job_dir, "timeline.json")
        print("writing {0}".format(filename_timeline))
        local_filename_timeline, filename_timeline = (
            mult999.file_utils.gs_download(filename_timeline, "w"))
        with open(local_filename_timeline, "w") as file:
            file.write(ctf)
        mult999.file_utils.gs_upload(local_filename_timeline, filename_timeline,
            "w", "application/json")

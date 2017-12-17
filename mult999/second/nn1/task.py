import mult999.trainer_utils
import tensorflow as tf

from mult999.constants import NUM_IN_FEATURES

BATCH_SIZE = 200
BUFFER_SIZE = 200
LEARNING_RATE = 0.5
MODEL_ID = "mult999_second_nn1"
NUM_HIDDEN_FEATURES = 100
NUM_OUT_FEATURES = 15
REPEAT = 10

def model():
    x = tf.placeholder(tf.float32, [None, NUM_IN_FEATURES])
    W1 = tf.Variable(tf.zeros([NUM_IN_FEATURES, NUM_HIDDEN_FEATURES]))
    b1 = tf.Variable(tf.zeros([NUM_HIDDEN_FEATURES]))
    y1 = tf.matmul(x, W1) + b1
    W2 = tf.Variable(tf.zeros([NUM_HIDDEN_FEATURES, NUM_OUT_FEATURES]))
    b2 = tf.Variable(tf.zeros([NUM_OUT_FEATURES]))
    y = tf.matmul(y1, W2) + b2
    y_ = tf.placeholder(tf.float32, [None, NUM_OUT_FEATURES])
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y))
    tf.summary.scalar("cross_entropy", cross_entropy)
    train_step = (tf.train.GradientDescentOptimizer(LEARNING_RATE)
        .minimize(cross_entropy))
    return x, y_, accuracy, train_step

def gather_second(in_chars, out_chars):
    return (in_chars, tf.gather(out_chars, [1], axis = 1))

if __name__ == "__main__":
    mult999.trainer_utils.main(MODEL_ID, model,
        BATCH_SIZE, gather_second, REPEAT, BUFFER_SIZE)

import mult999.first.nn0.model as model
import mult999.trainer_utils
import tensorflow as tf

BATCH_SIZE = 200
MODEL_ID = "mult999_first_nn0"

def gather_first(in_chars, out_chars):
    return (in_chars, tf.gather(out_chars, [0], axis = 1))

if __name__ == "__main__":
    mult999.trainer_utils.main(MODEL_ID, model.model, BATCH_SIZE, gather_first)

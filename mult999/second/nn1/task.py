import mult999.second.nn1.model as model
import mult999.trainer_utils
import tensorflow as tf

BATCH_SIZE = 200
BUFFER_SIZE = 200
MODEL_ID = "mult999_second_nn1"
REPEAT = 10

def gather_second(in_chars, out_chars):
    return (in_chars, tf.gather(out_chars, [1], axis = 1))

if __name__ == "__main__":
    mult999.trainer_utils.main(MODEL_ID, model.model,
        BATCH_SIZE, gather_second, REPEAT, BUFFER_SIZE)

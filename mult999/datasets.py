import argparse
import math
import mult999.file_utils
import os
import random
import sys
import tensorflow as tf

from mult999.constants import (DATASET_ID, MAX, NUM_EVAL_ELEMENTS,
    NUM_IN_CHARS, NUM_OUT_CHARS, NUM_TRAIN_ELEMENTS)

def parse_args():
    parser = argparse.ArgumentParser(description='Create datasets.')
    parser.add_argument('--filename-train',
        action = "store", metavar = "filename", required = True,
        help = "training data")
    parser.add_argument('--filename-eval',
        action = "store", metavar = "filename", required = True,
        help = "evaluation data")
    dict = vars(parser.parse_args())
    return (
        dict["filename_train"],
        dict["filename_eval"])

def write_dataset(filename, num_elements):

    def reverse(number):
        return str(number)[::-1]

    def pad(str, len):
        return str.ljust(len, "_")

    local_filename, filename = mult999.file_utils.gs_download(filename, "w")
    print("writing {0}".format(local_filename))
    with open(local_filename, "w") as file:

        for _ in range(0, num_elements):
            x1 = random.randint(0, MAX)
            x2 = random.randint(0, MAX)

            l1 = int(math.log10(x1)) if x1 > 0 else 0 
            l2 = int(math.log10(x2)) if x2 > 0 else 0

            yi = [] 
            k = 0
            for i in range(0, l2 + 1):
                d2 = x2 // 10**i % 10

                yi.append((x1 * d2) * 10**k)
                k += 1

            y = sum(yi)
            assert x1 * x2 == y

            in_ = "{0}*{1}|".format(x1, x2)

            if l2 > 0:
                out = "={0}={1}|".format("+".join(map(reverse, yi)), reverse(y))
            else:
                out = "={0}|".format(reverse(y))

            file.write("{0}{1}\n".format(
                pad(in_, NUM_IN_CHARS),
                pad(out, NUM_OUT_CHARS)))
    mult999.file_utils.gs_upload(local_filename, filename, "w", "text/plain")

def main():
    print("runtime Tensorflow {0}, Python {1}.{2}".format(
        tf.__version__, sys.version_info[0], sys.version_info[1]));
    (filename_train, filename_eval) = parse_args()
    mult999.file_utils.check_dirs([filename_train, filename_eval])

    write_dataset(filename_train, NUM_TRAIN_ELEMENTS)
    write_dataset(filename_eval, NUM_EVAL_ELEMENTS)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import math
import os
import random

from mult999_constants import (DATASET_ID, MAX, NUM_IN_CHARS, NUM_OUT_CHARS,
    NUM_TEST_ELEMENTS, NUM_TRAIN_ELEMENTS)

def write_dataset(filename, num_elements):

    def reverse(number):
        return str(number)[::-1]

    def pad(str, len):
        return str.ljust(len, "_")

    print("writing {0}".format(filename))

    with open(filename, "w") as file:

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

def main():
    directory = "datasets"
    filename_train = os.path.join(directory, "{0}.train".format(DATASET_ID))
    filename_test = os.path.join(directory, "{0}.test".format(DATASET_ID))

    write_dataset(filename_train, NUM_TRAIN_ELEMENTS)
    write_dataset(filename_test, NUM_TEST_ELEMENTS)

if __name__ == '__main__':
    main()

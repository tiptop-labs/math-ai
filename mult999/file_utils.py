import errno
import os

from tensorflow.python.lib.io import file_io

def check_dirs(filenames):
    dirnames = set(map(
        lambda filename: os.path.dirname(filename) or ".",
        filenames))
    for dirname in dirnames:
        if not file_io.file_exists(dirname):
            file_io.recursive_create_dir(dirname)
            print("create directory {0}".format(dirname))

        num_files = 0
        for _, _, filenames in os.walk(dirname):
            num_files += len(filenames)
        if num_files > 0:
            print("warning: {0} already contains {1} files".format(
                dirname, num_files))

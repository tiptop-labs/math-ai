import errno
import google.cloud.storage
import os
import tempfile

from tensorflow.python.lib.io import file_io
from urllib.parse import urlparse

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

def gs_download(filename, mode):
    assert mode == "w"
    scheme = urlparse(filename)[0]
    if scheme == "gs":
        _, suffix = os.path.splitext(filename)
        _, localFilename = tempfile.mkstemp(
            dir = "/tmp", suffix = suffix, text = True)
        return localFilename, filename
    else:
        return filename, None

def gs_upload(local_filename, filename, mode, content_type = None):
    assert mode == "w"
    if filename != None:
        print("uploading {0} to {1}".format(local_filename, filename))
        schema, bucket_name, path, _, _, _ = urlparse(filename)
        assert schema == "gs"
        client = google.cloud.storage.Client() 
        bucket = client.get_bucket(bucket_name) 
        blob = google.cloud.storage.Blob(path.lstrip("/"), bucket) 
        with open(local_filename, 'rb') as file: 
            blob.upload_from_file(file, content_type) 
        print("removing {0}".format(local_filename))
        os.remove(local_filename)

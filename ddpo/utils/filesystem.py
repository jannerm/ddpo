import os
import io
import re
import json
import pickle
import shutil
import gcsfs
import numpy as np
from PIL import Image

builtin_open = open


def open(path, bucket=None, mode="rb"):
    if bucket is not None:
        bucket = "gs://" + bucket if "gs://" not in bucket else bucket
        path = os.path.join(bucket, path)
    if "gs://" in path:
        fs = gcsfs.GCSFileSystem()
        return fs.open(path, mode=mode)
    else:
        return builtin_open(path, mode=mode)


def ls(path, bucket=None, strip=True):
    if bucket is not None:
        bucket = "gs://" + bucket if "gs://" not in bucket else bucket
        path = os.path.join(bucket, path)
    if "gs://" in path:
        fs = gcsfs.GCSFileSystem()
        paths = sorted(fs.ls(path))
        if strip:
            paths = ["/".join(p.split("/")[1:]) for p in paths]
        return paths
    else:
        return sorted(os.listdir(path))


def exists(path):
    if is_remote(path):
        fs = gcsfs.GCSFileSystem()
        return fs.exists(path)
    else:
        return os.path.exists(path)


def save(path, x):
    with open(path, mode="wb") as f:
        pickle.dump(x, f)


def save_img(path, x):
    # img_bytes = io.BytesIO()
    # np.save(img_bytes, img)
    img = Image.fromarray(np.uint8(x * 255.0))
    with open(path, mode="wb") as f:
        img.save(f, format="png")


def unpickle(path):
    with open(path, mode="rb") as f:
        return pickle.load(f)


def is_remote(path):
    return "gs://" in path


def get_bucket(path):
    matched = re.match("gs://.+?/", path)
    assert (
        matched is not None
    ), f"[ utils/filesystem ] Expected bucket in savepath, got {path}"
    start, end = matched.span()
    ## `gs://{bucket}/`
    bucket = path[start:end]
    ## `{bucket}`
    bucket = bucket.replace("gs://", "").replace("/", "")
    path = path[end:]
    print(f"[ utils/filesystem ] Found bucket in savepath: {bucket} | " f"path: {path}")
    return bucket, path


def rm(path):
    assert not is_remote(path)
    print(f"[ utils/filesystem ] Removing {path}")
    shutil.rmtree(path)


def save_json(path, x):
    with open(path, mode="w") as f:
        json.dump(x, f)


def read_json(path):
    with open(path, mode="r") as f:
        return json.load(f)


def join_and_create(*args):
    """Same as os.path.join, except it creates directories along the way."""
    path = os.path.join(*args)
    os.umask(0)
    os.makedirs(os.path.dirname(path), exist_ok=True, mode=0o777)
    return path

import os
import io
import string
import re
import numpy as np
import random
import jax
import dill
import glob
import h5py
import pytz
import gcsfs

from collections import defaultdict
from datetime import datetime
from google.cloud import storage
from PIL import Image
import pdb

from ddpo import utils
from .serialization import mkdir
from .timer import Timer


def encode_jpeg(x, quality=95):
    """
    x : np array
    """
    if issubclass(x.dtype.type, np.floating) or "float" in str(x.dtype.type):
        assert np.abs(x).max() <= 1.0
        x = (x * 255).astype(np.uint8)

    img = Image.fromarray(x)
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    jpeg = buffer.getvalue()
    return np.frombuffer(jpeg, dtype=np.uint8)


def decode_jpeg(jpeg):
    stream = io.BytesIO(jpeg)
    img = Image.open(stream)
    x = np.array(img) / 255.0
    return x


def encode_generic(x):
    buffer = dill.dumps(x)
    return np.frombuffer(buffer, dtype=np.uint8)


def decode_generic(x):
    return dill.loads(x)


def timestamp(timezone="US/Pacific"):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz=tz).strftime("%y-%m-%d_%H:%M:%S")


def randstr(n=10):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def slice2range(slice):
    start = slice.start or 0
    stop = slice.stop
    step = slice.step or 1
    return np.arange(start, stop, step)


class H5Writer:
    def __init__(self, savepath):
        mkdir(savepath, fname=True)
        self._make_file(savepath)

    def _make_file(self, savepath):
        self.savepath = savepath
        self._file = h5py.File(savepath, "w")
        self._sizes = {}
        self._max_sizes = {}
        self._vlens = {}
        self._encode_fns = {}
        self._decode_fns = {}
        self._attrs = ["_sizes", "_max_sizes", "_vlens", "_encode_fns", "_decode_fns"]

    def configure_from_reader(self, reader, max_size):
        file = reader._files[0] if hasattr(reader, "_files") else reader._file
        encode_fns = decode_generic(file.attrs["encode_fns"])
        decode_fns = decode_generic(file.attrs["decode_fns"])
        for field in file.keys():
            self.configure(
                field,
                max_size,
                encode_fn=encode_fns[field],
                decode_fn=decode_fns[field],
            )

    def configure(self, field, max_size, vlen=False, encode_fn=None, decode_fn=None):
        vlen = vlen or encode_fn is not None
        self._sizes[field] = 0
        self._max_sizes[field] = max_size
        self._vlens[field] = vlen
        self._encode_fns[field] = encode_fn
        self._decode_fns[field] = decode_fn

    def _create_dataset(self, field, x):
        dtype = x.dtype if hasattr(x, "dtype") else type(x)
        max_size = self._max_sizes[field]
        vlen = self._vlens[field]
        if vlen or not hasattr(x, "shape"):
            dtype = h5py.special_dtype(vlen=dtype)
            max_size = (max_size,)
        else:
            max_size = (max_size,) + x.shape
        print(
            f"[ utils/hdf5 ] Creating dataset {field} | "
            f"max size: {max_size} | dtype: {dtype} | "
            f"encode: {self._encode_fns[field]} | "
            f"decode: {self._decode_fns[field]}"
        )
        self._file.create_dataset(field, max_size, dtype=dtype, chunks=True)

    def add(self, field, x, skip_encoding=False):
        encode_fn = self._encode_fns[field]
        if encode_fn is not None and not skip_encoding:
            x = encode_fn(x)
        size = self._sizes[field]
        if size == 0:
            self._create_dataset(field, x)
        self._file[field][size] = x
        self._sizes[field] += 1

    def adds(self, field, xs, **kwargs):
        for x in xs:
            self.add(field, x, **kwargs)

    def add_batch(self, batch, mask=None, **kwargs):
        keys = batch.keys()
        sizes = [len(val) for val in batch.values()]
        assert len(set(sizes)) == 1, f"Batch sizes must be equal, got {sizes}"

        if mask is None:
            indices = range(sizes[0])
        else:
            indices = np.where(mask)[0]

        start = list(self._sizes.values())[0]
        end = start + len(indices)
        print(f"[ utils/hdf5 ] Adding {len(indices)} samples | [{start}, {end}]")

        for i in indices:
            for key, val in batch.items():
                # print(f'[ utils/hdf5 ] Adding {key} | {i}')
                self.add(key, val[i], **kwargs)
        return len(indices)

    def add_sliced(self, batch):
        keys = batch.keys()
        sizes = [len(val) for val in batch.values()]
        assert len(set(sizes)) == 1, f"Batch sizes must be equal, got {sizes}"
        size = sizes[0]

        for field, x in batch.items():
            start = self._sizes[field]
            end = start + size

            if start == 0:
                self._create_dataset(field, x[0])

            print(f"[ utils/hdf5 ] Adding {size} samples | [{start}, {end}]")

            try:
                self._file[field][start:end] = x[:]
                self._sizes[field] += size
            except:
                pdb.set_trace()

    def close(self):
        for field, size in self._sizes.items():
            old_shape = self._file[field].shape
            new_shape = (size,) + old_shape[1:]
            print(
                f"[ utils/hdf5 ] Resizing dataset {field} | {old_shape} -> {new_shape}"
            )
            self._file[field].resize(new_shape)

        self._file.attrs.update(
            {
                "encode_fns": encode_generic(self._encode_fns),
                "decode_fns": encode_generic(self._decode_fns),
            }
        )
        self._file.close()

    def write_images(self, savepath=None, start=0):
        savepath = savepath or self.savepath
        w = jax.process_index()
        i = 0
        for x in self._file["images"]:
            x = self._decode_fns["images"](x)
            utils.save_image(os.path.join(savepath, f"{w}_{i}.png"), x)
            i += 1
        return i


class H5Reader:
    def __init__(self, loadpath, mode="r"):
        self._file = h5py.File(loadpath, mode=mode)
        self._encode_fns = decode_generic(self._file.attrs["encode_fns"])
        self._decode_fns = decode_generic(self._file.attrs["decode_fns"])
        self._keys = list(self._file.keys())
        self._sizes = {key: self._file[key].shape[0] for key in self._keys}

    @property
    def sizes(self):
        return self._sizes

    def get(self, field, idx):
        x = self._file[field][idx]
        decode_fn = self._decode_fns[field]
        if decode_fn is not None:
            if isinstance(idx, slice):
                x = np.stack([decode_fn(xi) for xi in x])
            else:
                x = decode_fn(x)
        return x

    def load_all(self):
        batch = {key: list(self._file[key][:]) for key in self._file.keys()}
        return batch

    def __getitem__(self, idx):
        batch = {}
        for key in self._keys:
            batch[key] = self.get(key, idx)
        return batch


class H5Modifier(H5Reader, H5Writer):
    def __init__(self, loadpath):
        super().__init__(loadpath, mode="a")


class RemoteWriter(H5Writer):
    def __init__(
        self, savepath, split_size=1e3, bucket=None, tmpdir="/tmp", write_images=False
    ):
        if bucket is None:
            bucket, savepath = utils.fs.get_bucket(savepath)

        self._savepath = savepath
        self._split_size = split_size
        self._tmpdir = tmpdir
        self._write_images = write_images
        self._i = 0

        self._client = storage.Client()
        self._bucket = self._client.get_bucket(bucket)

        self._update_paths()
        super().__init__(self._local_path)

    def __len__(self):
        return max(self._sizes.values())

    def _test_permissions(self):
        fpath = os.path.abspath(__file__)
        fullpath = os.path.join(self._savepath, "_test.py")
        blob = self._bucket.blob(fullpath)
        blob.upload_from_filename(fpath)
        print("[ utils/hdf5 ] Successfully uploaded test file to bucket")

    def _get_local_attrs(self):
        fields = list(self._file.keys())
        attrs = {attr: getattr(self, attr) for attr in self._attrs}
        return attrs | {"fields": fields}

    def _set_local_attrs(self, attrs):
        fields = attrs["fields"]

        for field in fields:
            kwargs = {
                "max_size": attrs["_max_sizes"][field],
                "vlen": attrs["_vlens"][field],
                "encode_fn": attrs["_encode_fns"][field],
                "decode_fn": attrs["_decode_fns"][field],
            }
            super().configure(field, **kwargs)

    def _update_paths(self):
        worker = jax.process_index()
        fname = f"{timestamp()}-w{worker}-{randstr()}.hdf5"
        self._local_path = os.path.join(self._tmpdir, fname)
        self._remote_path = os.path.join(self._savepath, fname)
        print(f"[ utils/hdf5 ] New fname: {fname}")

    def configure(self, field, **kwargs):
        super().configure(field, max_size=self._split_size, **kwargs)

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

        if all([size >= self._split_size for size in self._sizes.values()]):
            if len(set(self._sizes.values())) > 1:
                print(f"[ utils/hdf5 ] Warning: sizes imbalanced | " f"{self._sizes}")

            local_attrs = self._close_local()

            ## initialize new local file
            self._update_paths()
            self._make_file(self._local_path)

            self._set_local_attrs(local_attrs)
        elif any([size >= 2 * self._split_size for size in self._sizes.values()]):
            print(
                f"[ utils/hdf5 ] Warning: local hdf5 size unexpectedly large | "
                f"{self._sizes}"
            )

    def close(self):
        if len(self):
            self._close_local()

    def _close_local(self):
        ## get attributes of local file
        local_attrs = self._get_local_attrs()

        if self._write_images:
            savedir = os.path.dirname(self._remote_path)
            writepath = os.path.join(f"gs://{self._bucket.name}", savedir)
            self._i = super().write_images(writepath, self._i)

        ## close local file
        super().close()

        ## copy local file to bucket
        print(
            f"[ utils/hdf5 ] Syncing entries to "
            f"gs://{self._bucket.name}/{self._remote_path} | "
            f"{self._sizes}"
        )
        blob = self._bucket.blob(self._remote_path)
        blob.upload_from_filename(self._local_path)

        ## delete local file
        os.remove(self._local_path)

        return local_attrs


class RemoteReader:
    def __init__(self, loadpath, project="rail-tpus", bucket=None):
        if bucket is None:
            bucket, loadpath = utils.fs.get_bucket(loadpath)

        self._client = storage.Client()
        self._bucket = self._client.get_bucket(bucket)
        self._remote_fs = gcsfs.GCSFileSystem(project=project)
        self._current_fid = None
        self._fetch_sizes(loadpath)
        self.weighted = False

    def _fetch_sizes(self, loadpath):
        fullpath = os.path.join(f"gs://{self._bucket.name}", loadpath, "sizes.pkl")
        sizes = utils.fs.unpickle(fullpath)
        self._remote_paths = sorted(sizes.keys())
        self._total_size = sum(sizes.values())

        print(
            f"[ utils/hdf5 ] Found {len(self._remote_paths)} files | "
            f"{self._total_size} entries"
        )

        self._idx2file = np.zeros(self._total_size, dtype=np.int64)
        self._idx2idx = np.zeros(self._total_size, dtype=np.int64)

        start = 0
        for fid, fname in enumerate(self._remote_paths):
            size = sizes[fname]
            end = start + size

            self._idx2file[start:end] = fid
            self._idx2idx[start:end] = np.arange(size)
            start = end

    def _fetch_file(self, fid):
        if fid == self._current_fid:
            return

        path = self._remote_paths[fid]
        file = self._load_remote(path)

        self._keys = list(file.keys())
        self._decode_fns = decode_generic(file.attrs["decode_fns"])

        self._current_file = file
        self._current_fid = fid

    def __len__(self):
        return self._total_size

    def get(self, remote_idx, field="images"):
        if isinstance(remote_idx, slice):
            return np.stack(
                [self.get(idx, field=field) for idx in slice2range(remote_idx)], axis=0
            )
        else:
            fid = self._idx2file[remote_idx]
            local_idx = self._idx2idx[remote_idx]
            self._fetch_file(fid)
            x = self._current_file[field][local_idx]

            decode_fn = self._decode_fns[field]
            if decode_fn is not None:
                x = decode_fn(x)
            return x

    def _load_remote(self, remote_path):
        # dtype = h5py.special_dtype(vlen=np.uint8)
        fullpath = os.path.join(self._bucket.name, remote_path)
        f = self._remote_fs.open(fullpath, cache_type="block")
        h5file = h5py.File(f, "r")
        return h5file

    def __getitem__(self, idx):
        fid = self._idx2file[idx]
        self._fetch_file(fid)

        batch = {}
        for key in self._keys:
            batch[key] = self.get(idx, field=key)
        if self.weighted:
            batch["weights"] = self.weights[idx]
        return batch

    def make_weights(self, field, temperature, by_prompt):
        labels = self.get(slice(0, len(self)), field).squeeze()
        if by_prompt:
            prompts = self.get(slice(0, len(self)), "inference_prompts").squeeze()
            self.weights = np.empty_like(labels)
            for prompt in np.unique(prompts):
                mask = prompts == prompt
                self.weights[mask] = (
                    utils.softmax_ref(labels[mask], temperature=temperature)
                    * mask.sum()
                )
        else:
            self.weights = utils.softmax_ref(labels, temperature=temperature) * len(
                self
            )
        self.weighted = True
        ## sanity check
        cumsum = np.cumsum(np.sort(self.weights)[::-1] / len(self))
        n = ((cumsum <= 0.9) * np.arange(len(cumsum))).max()
        print(
            "[ utils/hdf5 ] Weights sanity check: "
            f"{n} / {len(cumsum)} ({(n / len(cumsum)):.3}%) samples "
            "account for 90% of the weight | "
            f"temperature: {temperature}"
        )


class SlowRemoteReader:
    def __init__(self, bucket, loadpath, project="rail-tpus"):
        self._client = storage.Client()
        self._bucket = self._client.get_bucket(bucket)
        self._remote_fs = gcsfs.GCSFileSystem(project=project)

        remote_paths = [
            blob.name
            for blob in self._bucket.list_blobs(prefix=loadpath)
            if ".hdf5" in blob.name
        ]
        self._remote_paths = sorted(remote_paths)
        print(f"[ utils/hdf5 ] Found {len(remote_paths)} files")
        if len(self._remote_paths):
            self._fetch_files(self._remote_paths)
        else:
            self._idx2file = []

    def _fetch_files(self, remote_paths):
        timer = Timer()
        field = "images"
        self._files = [self._load_remote(path) for path in remote_paths]
        total_size = sum([len(file[field]) for file in self._files])
        print(
            f"[ utils/hdf5 ] Loaded {len(remote_paths)} files containing "
            f"{total_size} entries in {timer():2f} seconds"
        )

        ## decoding functions
        self._decode_fns = decode_generic(self._files[0].attrs["decode_fns"])

        self._idx2file = np.zeros(total_size, dtype=np.int64)
        self._idx2idx = np.zeros(total_size, dtype=np.int64)
        self._keys = list(self._files[0].keys())

        start = 0
        for fid, file in enumerate(self._files):
            size = len(file[field])
            end = start + size

            self._idx2file[start:end] = fid
            self._idx2idx[start:end] = np.arange(size)
            start = end

    def __len__(self):
        return len(self._idx2file)

    def get(self, remote_idx, field="images"):
        if isinstance(remote_idx, slice):
            return np.stack(
                [self.get(idx, field=field) for idx in slice2range(remote_idx)], axis=0
            )
        else:
            fid = self._idx2file[remote_idx]
            local_idx = self._idx2idx[remote_idx]
            x = self._files[fid][field][local_idx]

            decode_fn = self._decode_fns[field]
            if decode_fn is not None:
                x = decode_fn(x)
        return x

    def _load_remote(self, remote_path):
        fullpath = os.path.join(self._bucket.name, remote_path)
        f = self._remote_fs.open(fullpath, cache_type="block")
        h5file = h5py.File(f, "r")
        return h5file

    def load_all(self):
        batch = defaultdict(lambda: list())
        for file in self._files:
            keys = file.keys()
            for key in keys:
                data = file[key][:]

                batch[key].extend(data)
        return batch

    def __getitem__(self, idx):
        batch = {}
        for key in self._keys:
            batch[key] = self.get(idx, field=key)
        return batch


class LocalReader(SlowRemoteReader):
    def __init__(self, loadpath):
        remote_paths = glob.glob(os.path.join(loadpath, "*.hdf5"))
        self._remote_paths = sorted(remote_paths)
        print(f"[ utils/hdf5 ] Found {len(remote_paths)} files at {loadpath}")
        self._fetch_files(self._remote_paths)

    def _load_remote(self, remote_path):
        f = open(remote_path, mode="rb")
        h5file = h5py.File(f, "r")
        return h5file

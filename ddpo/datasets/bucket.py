from functools import partial
import random
import numpy as np
import jax
import torch

from ddpo import utils


class BucketDataset(torch.utils.data.Dataset):
    def __init__(self, reader):
        self.reader = reader
        self.transform_fn = lambda x: x
        self._max_size = None
        self._offset = 0
        self._shuffled = np.arange(len(self))

    def __len__(self):
        return self._max_size or len(self.reader)

    def __getitem__(self, idx):
        worker_idx = self._offset + idx
        shuffled_idx = self._shuffled[worker_idx]
        x = self.reader[shuffled_idx]
        info = {"idx": worker_idx, "shuffled_idx": shuffled_idx}
        return self.transform_fn(x) | info

    def shuffle(self):
        print("[ datasets/bucket ] Shuffling dataset")
        self._shuffled = np.random.permutation(self._shuffled)

    def shard(self):
        host_id = jax.process_index()
        n_hosts = jax.process_count()
        n_samples_per_host = len(self) // n_hosts
        self._max_size = n_samples_per_host
        self._offset = host_id * n_samples_per_host
        print(
            f"[ datasets/bucket ] Host: {host_id} | "
            f"Samples per host: {self._max_size} | "
            f"Offset: {self._offset}"
        )

    def make_weights(self, *args, **kwargs):
        self.reader.make_weights(*args, **kwargs)

    def with_transform(self, transform_fn):
        self.transform_fn = transform_fn

    def subsample(self, N):
        self._max_size = N


def preprocess_train(text_transforms, examples):
    examples["input_ids"] = text_transforms(examples)
    return examples


def select_caption(examples, field="training_prompts"):
    caption = examples[field]
    if isinstance(caption, (list, np.ndarray)):
        caption = random.choice(caption)
    examples["text"] = caption


def make_uncond_text(tokenizer, batch_size):
    uncond_prompt = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )
    return uncond_prompt.input_ids


def collate_fn(tokenizer, examples, image_field="vae", text_field="input_ids"):
    pixel_values = np.stack([example[image_field] for example in examples])
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.astype(np.float32)
    captions = [example["text"] for example in examples]

    ## @TODO: add `callback_` prefix to callback labels
    callback_labels = {
        key: np.stack([example[key] for example in examples])
        for key in ["aesthetic", "consistency", "jpeg", "labels", "weights"]
        if key in examples[0]
    }
    idxs = np.stack([example["idx"] for example in examples])
    shuffled_idxs = np.stack([example["shuffled_idx"] for example in examples])

    padded_tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )

    batch_size = len(pixel_values)
    uncond_prompt = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )

    batch = {
        image_field: pixel_values,
        text_field: padded_tokens.input_ids,
        "idxs": idxs,
        "shuffled_idxs": shuffled_idxs,
        "uncond_text": uncond_prompt.input_ids,
        **callback_labels,
    }

    return batch


def get_bucket_loader(
    loadpath,
    tokenizer,
    batch_size,
    resolution=None,
    max_train_samples=None,
    num_workers=0,
):
    if utils.fs.is_remote(loadpath):
        reader = utils.RemoteReader(loadpath)
    else:
        reader = utils.H5Reader(loadpath)
    train_dataset = BucketDataset(reader)

    ## subsample dataset
    if max_train_samples is not None:
        train_dataset.subsample(max_train_samples)

    ## training transforms
    train_dataset.with_transform(partial(preprocess_train, select_caption))

    train_dataset.shard()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_dataset, train_dataloader

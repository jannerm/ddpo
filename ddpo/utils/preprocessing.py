import numpy as np
import random
import jax


def tokenize_captions(
    tokenizer,
    examples,
    field="text",
    is_train=True,
    padding="do_not_pad",
    truncation=True,
):
    captions = []
    for caption in examples[field]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{field}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding=padding,
        truncation=truncation,
    )
    input_ids = inputs.input_ids
    return input_ids


def shard(xs, devices=None):
    """Helper for pmap to shard a pytree of arrays by local_device_count.
    Args:
        xs: a pytree of arrays.
    Returns:
        A matching pytree with arrays' leading dimensions sharded by the
        local device count.
    """
    if devices:
        local_device_count = len(devices)
    else:
        local_device_count = jax.local_device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs
    )


def unshard(xs):
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), xs)

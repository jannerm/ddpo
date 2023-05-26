from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
from flax.jax_utils import replicate, unreplicate


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_across_devices(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_map(select, state)


def n_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def worker_sum(n):
    @partial(jax.pmap, axis_name="batch")
    def _sum(mask):
        total = jax.lax.psum(mask, "batch")
        return total

    n_rep = replicate(n)
    return unreplicate(_sum(n_rep)).item() / len(n_rep)


def softmax_ref(x, temperature=1.0):
    """
    Used to test correctness of pmapped softmax function
    """
    assert x.ndim == 1
    x = x * temperature
    z = x - x.max()
    numer = np.exp(z)
    denom = numer.sum()
    return numer / denom


def softmax(x, temperature=1.0):
    x = x * temperature

    @partial(jax.pmap, axis_name="batch")
    def _softmax(x):
        z = x - jax.lax.pmax(x, "batch").max(keepdims=True)
        numer = jnp.exp(z)
        denom = jax.lax.psum(numer, "batch").sum(keepdims=True)
        # return numer / denom
        return numer / denom

    return _softmax(x)

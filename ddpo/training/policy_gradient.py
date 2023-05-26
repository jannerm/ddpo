import jax
import jax.numpy as jnp
from diffusers.models import vae_flax
import pdb
from functools import partial
import optax
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from typing import Any, Callable
from ddpo.diffusers_patch.scheduling_ddim_flax import FlaxDDIMScheduler


class AccumulatingTrainState(TrainState):
    """A TrainState that accumulates gradients over multiple steps before
    applying them. This is an alternative to `optax.MultiSteps` that uses less
    memory because it takes `do_update` as a static argument (rather than
    computing it internally).

    Since MultiSteps effectively computes `do_update` internally and uses
    `jax.lax.cond` to select the output, it must allocate an additional buffer
    with the size of `opt_state` because it doesn't "know" at compile time
    whether it will update `opt_state` or not. Instead, this version leads to
    two compiled functions, one for each value of `do_update`, meaning it only
    allocates one buffer for `opt_state` in each case.

    This still requires an additional buffer of size `params` to store the
    accumulated gradients, but that is unavoidable."""

    grad_acc: FrozenDict[str, Any]
    n_acc: int

    def apply_gradients(self, *, grads, do_update, **kwargs):
        if do_update:
            new_state = super().apply_gradients(
                grads=jax.tree_map(
                    lambda ga, g: (ga + g) / (self.n_acc + 1), self.grad_acc, grads
                ),
                **kwargs
            )
            new_state = new_state.replace(
                grad_acc=jax.tree_map(jnp.zeros_like, self.grad_acc), n_acc=0
            )
        else:
            new_state = self.replace(
                grad_acc=jax.tree_map(jnp.add, self.grad_acc, grads),
                n_acc=self.n_acc + 1,
            )
        return new_state

    @classmethod
    def create(cls, *, params, **kwargs):
        return super().create(
            params=params,
            grad_acc=jax.tree_map(jnp.zeros_like, params),
            n_acc=0,
            **kwargs
        )


ADV_CLIP_MAX = 10.0


def train_step(
    state,
    batch,
    noise_scheduler_state,
    # static arguments below
    noise_scheduler,
    train_cfg,
    guidance_scale,
    eta,
    clip_range,
    # whether to update the optimizer or accumulate gradients. this has to be
    # computed outside of a jitted function for memory reasons.
    do_opt_update,
):
    assert isinstance(state, AccumulatingTrainState)
    assert isinstance(noise_scheduler, FlaxDDIMScheduler)
    assert (
        batch["latents"].shape[0]
        == batch["ts"].shape[0]
        == batch["next_latents"].shape[0]
        == batch["log_probs"].shape[0]
    )

    def compute_loss(params):
        unet_outputs = state.apply_fn(
            {"params": params},
            batch["latents"],
            batch["ts"],
            batch["prompt_embeds"],
            train=True,
        )

        if train_cfg:
            uncond_outputs = state.apply_fn(
                {"params": params},
                batch["latents"],
                batch["ts"],
                batch["uncond_embeds"],
                train=True,
            )
            noise_pred = uncond_outputs.sample + guidance_scale * (
                unet_outputs.sample - uncond_outputs.sample
            )
        else:
            noise_pred = unet_outputs.sample

        # just computes the log prob of batch["next_latents"] given noise_pred
        _, _, log_prob = noise_scheduler.step(
            noise_scheduler_state,
            noise_pred,
            batch["ts"],
            batch["latents"],
            key=None,
            prev_sample=batch["next_latents"],
            eta=eta,
        )

        # ppo logic
        advantages = jnp.clip(batch["advantages"], -ADV_CLIP_MAX, ADV_CLIP_MAX)
        ratio = jnp.exp(log_prob - batch["log_probs"])
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        loss = jnp.mean(jnp.maximum(unclipped_loss, clipped_loss))

        # debugging values
        info = {}
        # John Schulman says that (ratio - 1) - log(ratio) is a better
        # estimator, but most existing code uses this so...
        # http://joschu.net/blog/kl-approx.html
        info["approx_kl"] = 0.5 * jnp.mean((log_prob - batch["log_probs"]) ** 2)
        info["clipfrac"] = jnp.mean(jnp.abs(ratio - 1.0) > clip_range)
        info["loss"] = loss

        return loss, info

    grad_fn = jax.grad(compute_loss, has_aux=True)
    grad, info = grad_fn(state.params)

    grad = jax.lax.pmean(grad, "batch")
    info = jax.lax.pmean(info, "batch")

    new_state = state.apply_gradients(grads=grad, do_update=do_opt_update)

    return new_state, info

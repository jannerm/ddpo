import jax

jax.distributed.initialize()

import math
import numpy as np
import jax.numpy as jnp
import optax
import transformers
import diffusers

from functools import partial
from flax import jax_utils
from flax.training import train_state
from jax.experimental.compilation_cache import compilation_cache

import ddpo
from ddpo import training, utils


class Parser(utils.Parser):
    config: str = "config.base"
    dataset: str = "consistent_imagenet"


DEVICES = jax.local_devices()

p_train_step = jax.pmap(
    training.diffusion.train_step,
    axis_name="batch",
    donate_argnums=(0,),
    static_broadcasted_argnums=(5,),
)

# ------------------------------------ debug -----------------------------------#


@partial(jax.pmap, axis_name="batch")
def verify_n_workers(y):
    x = jnp.ones(1)
    x = jax.lax.psum(x, "batch")
    return x


n_workers = verify_n_workers(jnp.ones(len(DEVICES)))
print("WORKERS", n_workers)
print("hosts", jax.process_count())
print("proc", jax.process_index())

# ---------------------------------- end debug ---------------------------------#


def main():
    args = Parser().parse_args("train")
    transformers.set_seed(args.seed)
    compilation_cache.initialize_cache(args.cache)
    logger = utils.init_logging("finetune", args.verbose)
    args.modelpath = None if args.iteration == 0 else args.modelpath

    # --------------------------------- models ---------------------------------#

    stable_models, stable_params = utils.load_finetuned_stable_diffusion(
        args.modelpath,
        epoch=args.load_epoch,
        pretrained_model=args.pretrained_model,
        dtype=args.dtype,
        cache=args.cache,
    )
    tokenizer, text_encoder, vae, unet = stable_models
    vae_params, unet_params = stable_params

    print(f"n clip params: {utils.n_params(text_encoder.params)/1e6:.3f}M")
    print(f"n vae  params: {utils.n_params(vae_params)/1e6:.3f}M")
    print(f"n unet params: {utils.n_params(unet_params)/1e6:.3f}M")

    # -------------------------------- dataset ---------------------------------#

    worker_batch_size = args.train_batch_size * len(jax.local_devices())
    pod_batch_size = worker_batch_size * jax.process_count()

    train_dataset, train_dataloader = ddpo.datasets.get_bucket_loader(
        args.loadpath,
        tokenizer,
        batch_size=worker_batch_size,
        resolution=args.resolution,
        max_train_samples=args.max_train_samples,
    )

    assert not (
        args.weighted_batch and args.weighted_dataset
    ), "Cannot weight over both batch and dataset"

    if args.weighted_dataset:
        train_dataset.make_weights(
            args.filter_field, args.temperature, args.per_prompt_weights
        )

    # ------------------------------- optimizer --------------------------------#

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.beta1,
        b2=args.beta2,
        eps=args.epsilon,
        weight_decay=args.weight_decay,
        mu_dtype=jnp.bfloat16,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    # optimizer = optax.MultiSteps(
    #     optimizer,
    #     every_k_schedule=2,
    # )

    state = train_state.TrainState.create(
        apply_fn=unet.apply,
        params=unet_params,
        tx=optimizer,
    )

    noise_scheduler = diffusers.FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    noise_scheduler_state = noise_scheduler.create_state()

    # ------------------------------ replication -------------------------------#

    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    # vae_params = jax_utils.replicate(vae_params, devices=DEVICES)
    noise_scheduler_state = jax_utils.replicate(noise_scheduler_state)

    # -------------------------- generic training setup ------------------------#

    rng = jax.random.PRNGKey(args.seed)
    train_rngs = jax.random.split(rng, len(jax.local_devices()))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    static_broadcasted = (
        noise_scheduler,
        text_encoder,
        args.train_cfg,
        args.guidance_scale,
    )

    # -------------------------------- main loop -------------------------------#

    logger.info(f"dataset size: {len(train_dataset)}")
    logger.info(f"batch size per device: {args.train_batch_size}")
    logger.info(f"batch size per worker: {worker_batch_size}")
    logger.info(f"total pod batch size: {pod_batch_size}")
    logger.info(f"n epochs: {args.num_train_epochs}")
    logger.info(f"n optimization steps: {args.max_train_steps}")

    global_step = 0

    for epoch in range(args.num_train_epochs):
        losses = []
        # train_dataloader.dataset.shuffle()
        steps_per_epoch = len(train_dataset) // worker_batch_size
        progress = utils.Progress(steps_per_epoch, name=f"Epoch {epoch}")

        for batch in train_dataloader:
            batch = utils.shard(batch)

            if args.weighted_batch:
                rewards = batch[args.filter_field].squeeze()
                weights = utils.softmax(rewards, temperature=args.temperature)
            elif args.weighted_dataset:
                # each item has an expected weight of 1, so divide by batch size
                # to get an expected batch sum of 1
                weights = batch["weights"].squeeze() / pod_batch_size
            else:
                weights = None

            state, loss, train_rngs = p_train_step(
                state,
                text_encoder_params,
                batch,
                train_rngs,
                noise_scheduler_state,
                static_broadcasted,
                weights=weights,
            )
            losses.append(loss)

            description = {
                "idx_min": batch["idxs"].min(),
                "idx_max": batch["idxs"].max(),
                "shx_min": batch["shuffled_idxs"].min(),
                "shx_max": batch["shuffled_idxs"].max(),
                "cfg": args.train_cfg,
                "scale": args.guidance_scale,
                "weights_sum": weights.sum() if weights is not None else 0,
                # 'labels': batch['labels'].mean(),
            }
            progress(description)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

        loss_avg = np.mean(jax_utils.unreplicate(losses))
        progress({"average": loss_avg} | description, n=0)
        progress.stamp()

        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_train_epochs - 1:
            localpath, proc = utils.save_unet(
                args.savepath,
                state.params,
                all_workers=True,
                epoch=(epoch + 1) // args.save_freq * args.save_freq,
            )


if __name__ == "__main__":
    main()

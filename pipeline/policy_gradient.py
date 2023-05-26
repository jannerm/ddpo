import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import transformers

from functools import partial
from concurrent import futures
from flax import jax_utils
from flax.training.common_utils import shard
from flax.training.checkpoints import save_checkpoint_multiprocess
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.profiler

import ddpo
from ddpo import training, utils
from ddpo.utils.stat_tracking import PerPromptStatTracker
from ddpo.diffusers_patch.pipeline_flax_stable_diffusion import (
    FlaxStableDiffusionPipeline,
)
from ddpo.diffusers_patch.scheduling_ddim_flax import FlaxDDIMScheduler
from ddpo.utils.serialization import is_dtype, get_dtype, to_dtype, load_flax_model
import tqdm
import matplotlib.pyplot as plt


class Parser(utils.Parser):
    config: str = "config.base"
    dataset: str = "consistent_imagenet"


p_train_step = jax.pmap(
    training.policy_gradient.train_step,
    axis_name="batch",
    donate_argnums=0,
    static_broadcasted_argnums=(3, 4, 5, 6, 7, 8),
)


def main():
    args = Parser().parse_args("pg")
    transformers.set_seed(args.seed)
    compilation_cache.initialize_cache(args.cache)  # only works on TPU
    utils.init_logging("policy_gradient", args.verbose)

    rng = jax.random.PRNGKey(args.seed)
    n_devices = jax.local_device_count()
    train_worker_batch_size = n_devices * args.train_batch_size
    train_pod_batch_size = train_worker_batch_size * jax.process_count()
    train_effective_batch_size = train_pod_batch_size * args.train_accumulation_steps
    sample_worker_batch_size = n_devices * args.sample_batch_size
    sample_pod_batch_size = sample_worker_batch_size * jax.process_count()
    total_samples_per_epoch = args.num_sample_batches_per_epoch * sample_pod_batch_size
    print(
        f"[ policy_gradient ] local devices: {n_devices} | "
        f"number of workers: {jax.process_count()}"
    )
    print(
        f"[ policy_gradient ] sample worker batch size: {sample_worker_batch_size} | "
        f"sample pod batch size: {sample_pod_batch_size}"
    )
    print(
        f"[ policy_gradient ] train worker batch size: {train_worker_batch_size} | "
        f"train pod batch size: {train_pod_batch_size} | "
        f"train accumulated batch size: {train_effective_batch_size}"
    )
    print(
        f"[ policy_gradient ] number of sample batches per epoch: {args.num_sample_batches_per_epoch}"
    )
    print(
        f"[ policy_gradient ] total number of samples per epoch: {total_samples_per_epoch}"
    )
    print(
        f"[ policy_gradient ] number of gradient updates per inner epoch: {total_samples_per_epoch // train_effective_batch_size}"
    )
    assert args.sample_batch_size >= args.train_batch_size
    assert args.sample_batch_size % args.train_batch_size == 0
    assert total_samples_per_epoch % train_effective_batch_size == 0

    worker_id = jax.process_index()

    localpath = "logs/" + args.savepath.replace("gs://", "")
    os.umask(0)
    os.makedirs(localpath, exist_ok=True, mode=0o777)
    with open(f"{localpath}/args.json", "w") as f:
        json.dump(args._dict, f, indent=4)

    # --------------------------------- models ---------------------------------#
    # load text encoder, vae, and unet (unet is optionally loaded from a local
    # fine-tuned path if args.loadpath is not None)
    print("loading models...")
    pipeline, params = utils.load_unet(
        None,
        epoch=args.load_epoch,
        pretrained_model=args.pretrained_model,
        dtype=args.dtype,
        cache=args.cache,
    )
    pipeline.safety_checker = None
    params = jax.device_get(params)

    pipeline.scheduler = FlaxDDIMScheduler(
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
        beta_start=pipeline.scheduler.config.beta_start,
        beta_end=pipeline.scheduler.config.beta_end,
        beta_schedule=pipeline.scheduler.config.beta_schedule,
        trained_betas=pipeline.scheduler.config.trained_betas,
        set_alpha_to_one=pipeline.scheduler.config.set_alpha_to_one,
        steps_offset=pipeline.scheduler.config.steps_offset,
        prediction_type=pipeline.scheduler.config.prediction_type,
    )
    noise_scheduler_state = pipeline.scheduler.set_timesteps(
        params["scheduler"],
        num_inference_steps=args.n_inference_steps,
        shape=(
            args.train_batch_size,
            pipeline.unet.in_channels,
            args.resolution // pipeline.vae_scale_factor,
            args.resolution // pipeline.vae_scale_factor,
        ),
    )

    # ------------------------------- optimizer --------------------------------#
    print("initializing train state...")
    constant_scheduler = optax.constant_schedule(args.learning_rate)

    optim = {
        "adamw": optax.adamw(
            learning_rate=constant_scheduler,
            b1=args.beta1,
            b2=args.beta2,
            eps=args.epsilon,
            weight_decay=args.weight_decay,
            mu_dtype=jnp.bfloat16,
        ),
        "adafactor": optax.adafactor(
            learning_rate=constant_scheduler,
            weight_decay_rate=args.weight_decay,
        ),
    }[args.optimizer]

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optim,
    )

    opt_state = jax.jit(optimizer.init, backend="cpu")(params["unet"])
    # NOTE(kvablack) optax.MultiSteps takes way more memory than necessary; this
    # class is a workaround that requires compiling twice. there may be a better
    # way but this works.
    state = training.policy_gradient.AccumulatingTrainState(
        step=0,
        apply_fn=pipeline.unet.apply,
        params=params["unet"],
        tx=optimizer,
        opt_state=opt_state,
        grad_acc=jax.tree_map(np.zeros_like, params["unet"]),
        n_acc=0,
    )

    # ------------------------------ replication -------------------------------#
    state = jax_utils.replicate(state)
    sampling_scheduler_params = jax_utils.replicate(params["scheduler"])
    noise_scheduler_state = jax_utils.replicate(noise_scheduler_state)

    # -------------------------- setup ------------------------#
    timer = utils.Timer()

    @partial(jax.pmap)
    def vae_decode(latents, vae_params):
        # expects latents in NCHW format (batch_size, 4, 64, 64)
        latents = latents / 0.18215
        images = pipeline.vae.apply(
            {"params": vae_params}, latents, method=pipeline.vae.decode
        ).sample
        images = (images / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return images

    # text encode on CPU to save memory
    @partial(jax.jit, backend="cpu")
    def text_encode(input_ids):
        return pipeline.text_encoder(input_ids, params=params["text_encoder"])[0]

    # make uncond prompts and embed them
    uncond_prompt_ids = ddpo.datasets.make_uncond_text(pipeline.tokenizer, 1)
    timer()
    uncond_prompt_embeds = text_encode(uncond_prompt_ids).squeeze()
    print(f"[ embed uncond prompts ] in {timer():.2f}s")

    sample_uncond_prompt_embeds = np.broadcast_to(
        uncond_prompt_embeds, (args.sample_batch_size, *uncond_prompt_embeds.shape)
    )
    sample_uncond_prompt_embeds = jax_utils.replicate(sample_uncond_prompt_embeds)
    train_uncond_prompt_embeds = sample_uncond_prompt_embeds[:, : args.train_batch_size]

    train_rng, sample_rng = jax.random.split(rng)

    # ------------------------------ callbacks -------------------------------#
    callback_fns = {
        args.filter_field: training.callback_fns[args.filter_field](),
    }

    # executor to perform callbacks asynchronously
    # this is beneficial for the llava callbacks which makes a request to a
    # server running llava inference
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # put vae params on device for sampling
    params["vae"] = jax_utils.replicate(params["vae"])

    if args.per_prompt_stats_bufsize is not None:
        per_prompt_stats = PerPromptStatTracker(
            args.per_prompt_stats_bufsize, args.per_prompt_stats_min_count
        )

    mean_rewards = []  # mean reward for each epoch
    std_rewards = []  # std reward for each epoch
    for epoch in range(args.num_train_epochs):
        # list of dicts with entires of shape:
        # (num_devices * sample_batch_size, ...)
        samples = []

        for i in tqdm.tqdm(
            range(args.num_sample_batches_per_epoch),
            desc=f"epoch {epoch}: sample",
            position=0,
        ):
            # ----------------------------- make prompts ----------------------------- #
            # shape (num_devices * sample_batch_size, ...)
            sample_prompts, training_prompts, prompt_metadata = training.make_prompts(
                args.prompt_fn,
                n_devices * args.sample_batch_size,
                args.identical_batch,
                evaluate=args.evaluate,
                **args.prompt_kwargs,
            )

            # ----------------------------- sample ----------------------------- #
            sample_rng, sample_seed = jax.random.split(sample_rng)
            sample_seeds = jax.random.split(sample_seed, n_devices)

            # encode prompts
            sample_prompt_ids = pipeline.prepare_inputs(sample_prompts)
            sample_prompt_embeds = text_encode(sample_prompt_ids)

            timer()
            sampling_params = {
                "unet": state.params,
                "scheduler": sampling_scheduler_params,
            }
            # shape (num_devices, sample_batch_size, ...)
            final_latents, latents, next_latents, log_probs, ts = pipeline(
                shard(sample_prompt_embeds),
                sample_uncond_prompt_embeds,
                sampling_params,
                sample_seeds,
                args.n_inference_steps,
                jit=True,
                height=args.resolution,
                width=args.resolution,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
            )

            # ----------------------------- decode latents ----------------------------- #
            images = vae_decode(final_latents, params["vae"])

            # ----------------------------- evaluate callbacks ----------------------------- #
            # fetch decoded images to device
            images = jax.device_get(utils.unshard(images))
            # evaluate callbacks
            callbacks = executor.submit(
                training.evaluate_callbacks,
                callback_fns,
                images,
                sample_prompts,
                prompt_metadata,
            )
            # yield to callback to make sure it has a chance to run
            time.sleep(0)

            # ----------------------------- save ----------------------------- #
            samples.append(
                {
                    "prompts": np.array(sample_prompts),
                    "embeds": np.array(sample_prompt_embeds),
                    "latents": jax.device_get(utils.unshard(latents)),
                    "next_latents": jax.device_get(utils.unshard(next_latents)),
                    "log_probs": jax.device_get(utils.unshard(log_probs)),
                    "ts": jax.device_get(utils.unshard(ts)),
                    "callbacks": callbacks,
                }
            )

            # save a sample
            pipeline.numpy_to_pil(images[0])[0].save(
                utils.fs.join_and_create(
                    localpath, f"samples/{worker_id}_{epoch}_{i}.png"
                )
            )

        # take vae params off device to save memory
        # NOTE(kvablack) this doesn't actually seem to do anything
        # params["vae"] = jax.device_get(jax_utils.unreplicate(params["vae"]))

        # wait for callbacks to finish
        for sample in samples:
            sample["rewards"], sample["callback_info"] = sample["callbacks"].result()[
                args.filter_field
            ]
            del sample["callbacks"]

        # collate samples into a single dict, flattening
        # shape: (num_sample_batches_per_epoch * num_devices * sample_batch_size, ...)
        samples = jax.tree_map(lambda *xs: np.concatenate(xs), *samples)

        # allgather rewards (for multi-host training)
        rewards = np.array(
            multihost_utils.process_allgather(samples["rewards"], tiled=True)
        )

        # per-prompt mean/std tracking
        if args.per_prompt_stats_bufsize is not None:
            prompt_ids = pipeline.tokenizer(
                samples["prompts"].tolist(), padding="max_length", return_tensors="np"
            ).input_ids
            prompt_ids = multihost_utils.process_allgather(prompt_ids, tiled=True)
            prompts = np.array(
                pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            )

            advantages = per_prompt_stats.update(prompts, rewards)

            if jax.process_index() == 0:
                np.save(
                    utils.fs.join_and_create(
                        localpath, f"per_prompt_stats/{worker_id}_{epoch}.npy"
                    ),
                    per_prompt_stats.get_stats(),
                )
        else:
            advantages = (rewards - np.mean(rewards)) / np.std(rewards)

        samples["advantages"] = advantages.reshape(jax.process_count(), -1)[worker_id]
        print(f"mean reward: {np.mean(rewards):.4f}")

        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

        # save data for future analysis
        np.save(
            utils.fs.join_and_create(localpath, f"rewards/{worker_id}_{epoch}.npy"),
            samples["rewards"],
        )
        np.save(
            utils.fs.join_and_create(localpath, f"prompts/{worker_id}_{epoch}.npy"),
            samples["prompts"],
        )
        np.save(
            utils.fs.join_and_create(
                localpath, f"callback_info/{worker_id}_{epoch}.npy"
            ),
            samples["callback_info"],
        )
        del samples["prompts"]
        del samples["callback_info"]
        del samples["rewards"]

        for inner_epoch in range(args.num_inner_epochs):
            total_batch_size, num_timesteps = samples["log_probs"].shape
            assert (
                total_batch_size
                == args.num_sample_batches_per_epoch
                * n_devices
                * args.sample_batch_size
            )
            assert num_timesteps == args.n_inference_steps

            # shuffle samples along batch dimension
            perm = np.random.permutation(total_batch_size)
            samples = jax.tree_map(lambda x: x[perm], samples)

            # shuffle along time dimension, independently for each sample
            perms = np.array(
                [np.random.permutation(num_timesteps) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents", "log_probs", "ts"]:
                samples[key] = samples[key][np.arange(total_batch_size)[:, None], perms]

            # rebatch (and shard) for training
            samples_train = jax.tree_map(
                lambda x: x.reshape(-1, n_devices, args.train_batch_size, *x.shape[1:]),
                samples,
            )

            # dict of lists to list of dicts for iteration
            samples_train = [
                dict(zip(samples_train, x)) for x in zip(*samples_train.values())
            ]

            # number of timesteps within each trajectory to train on
            num_train_ts = int(num_timesteps * args.train_timestep_ratio)

            all_infos = []
            for i, sample in tqdm.tqdm(
                list(enumerate(samples_train)),
                desc=f"outer epoch {epoch}, inner epoch {inner_epoch}: train",
            ):
                for j in range(num_train_ts):
                    batch = {
                        "prompt_embeds": sample["embeds"],
                        "uncond_embeds": train_uncond_prompt_embeds,
                        "advantages": sample["advantages"],
                        "latents": sample["latents"][:, :, j],
                        "next_latents": sample["next_latents"][:, :, j],
                        "log_probs": sample["log_probs"][:, :, j],
                        "ts": sample["ts"][:, :, j],
                    }
                    # update if we're at the last timestep (i.e. we've finished a sequence)
                    # and we've accumulated enough samples
                    do_opt_update = (j == num_train_ts - 1) and (
                        (i + 1) % args.train_accumulation_steps == 0
                    )
                    if do_opt_update:
                        print(f"opt update at {i}, {j}")
                    state, info = p_train_step(
                        state,
                        batch,
                        noise_scheduler_state,
                        pipeline.scheduler,
                        args.train_cfg,
                        args.guidance_scale,
                        args.eta,
                        args.ppo_clip_range,
                        do_opt_update,
                    )
                    multihost_utils.assert_equal(info, "infos equal")
                    all_infos.append(
                        jax.tree_map(lambda x: np.mean(jax.device_get(x)), info)
                    )
            assert do_opt_update
            all_infos = jax.tree_map(lambda *xs: np.stack(xs), *all_infos)
            print(f"mean info: {jax.tree_map(np.mean, all_infos)}")
            if jax.process_index() == 0:
                np.save(
                    utils.fs.join_and_create(
                        localpath, f"train_info/{worker_id}_{epoch}_{inner_epoch}.npy"
                    ),
                    all_infos,
                )

        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_train_epochs - 1:
            save_checkpoint_multiprocess(
                os.path.join(args.savepath, "checkpoints"),
                jax_utils.unreplicate(state.params),
                step=epoch,
                keep=1e6,
                overwrite=True,
            )

        # plot reward curve
        if jax.process_index() == 0:
            plt.clf()
            plt.plot(mean_rewards, color="black")
            plt.fill_between(
                range(len(mean_rewards)),
                np.array(mean_rewards) - np.array(std_rewards),
                np.array(mean_rewards) + np.array(std_rewards),
                alpha=0.4,
                color="blue",
            )
            plt.savefig(os.path.join(localpath, f"log_{worker_id}.png"))
            print()

            utils.async_to_bucket(localpath, args.savepath)


if __name__ == "__main__":
    main()

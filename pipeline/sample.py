import os
import jax

jax.distributed.initialize()

import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax.experimental.compilation_cache import compilation_cache
import pdb

from ddpo import datasets, training, utils


class Parser(utils.Parser):
    config: str = "config.base"
    dataset: str = "compressed_dogs"


args = Parser().parse_args("sample")
compilation_cache.initialize_cache(args.cache)

rng = jax.random.PRNGKey(args.seed)
n_devices = jax.local_device_count()
batch_size = n_devices * args.n_samples_per_device
pod_batch_size = batch_size * jax.process_count()
print(
    f"[ sample ] local devices: {n_devices} | "
    f"pod devices: {n_devices * jax.process_count()} | "
    f"worker batch_size: {batch_size} | "
    f"pod batch size: {pod_batch_size}"
)

# ----------------------------------- loading ----------------------------------#

loadpath = None if args.iteration == 0 else args.loadpath
pipeline, params = utils.load_unet(
    loadpath,
    epoch=args.load_epoch,
    pretrained_model=args.pretrained_model,
)
params = replicate(params)
pipeline.safety_checker = None

callback_keys = [args.filter_field, "vae"]
callback_fns = {key: training.callback_fns[key]() for key in callback_keys}

if args.guidance_scale == "auto":
    args.guidance_scale = utils.load_guidance_scale(args.loadpath)

vae_decode = jax.pmap(training.vae_decode, static_broadcasted_argnums=(2, 3))
text_encode = jax.pmap(training.text_encode, static_broadcasted_argnums=(2,))
training.patch_scheduler(pipeline)

# ----------------------------------- bucket -----------------------------------#

writer = utils.RemoteWriter(args.savepath, split_size=args.local_size)
writer.configure("images", encode_fn=utils.encode_jpeg, decode_fn=utils.decode_jpeg)
writer.configure("inference_prompts")
writer.configure(
    "training_prompts", encode_fn=utils.encode_generic, decode_fn=utils.decode_generic
)
for key in callback_fns:
    writer.configure(key)

# -------------------------------- uncond prompt --------------------------------#

uncond_prompt_ids = datasets.make_uncond_text(pipeline.tokenizer, batch_size)
uncond_prompt_embeds = text_encode(
    shard(uncond_prompt_ids), params["text_encoder"], pipeline.text_encoder
)
print(f"[ sample ] embed uncond prompts: {uncond_prompt_embeds.shape}")

# ---------------------------------- main loop ---------------------------------#

masker = utils.make_masker(args.mask_mode, args.mask_param)
avg = utils.StreamingAverage()
timer = utils.Timer()

print(
    f"[ sample ] max_samples: {args.max_samples} | "
    f"max_steps: {args.max_steps} | eval: {args.evaluate}"
)

n_steps = 0
n_samples = 0
all_rewards = []
while True:
    rng, prng_seed = jax.random.split(rng)
    prng_seeds = jax.random.split(prng_seed, n_devices)

    inference_prompts, training_prompts, prompt_metadata = training.make_prompts(
        args.prompt_fn,
        batch_size,
        args.identical_batch,
        evaluate=args.evaluate,
        **args.prompt_kwargs,
    )
    print(f"[ sample ] prompts: {inference_prompts[:2]}")

    prompt_ids = pipeline.prepare_inputs(inference_prompts)
    prompt_embeds = text_encode(
        shard(prompt_ids), params["text_encoder"], pipeline.text_encoder
    )

    final_latents, *_ = pipeline(
        prompt_embeds,
        uncond_prompt_embeds,
        params,
        prng_seeds,
        args.n_inference_steps,
        jit=True,
        height=args.resolution,
        width=args.resolution,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )
    images = utils.unshard(
        vae_decode(
            final_latents,
            params["vae"],
            pipeline.vae.apply,
            pipeline.vae.decode,
        )
    )
    images = np.array(images).astype(np.float32)

    print(
        f"[ sample ] {len(images)} samples in {timer():.2f} seconds | eval: {args.evaluate}"
    )

    infos = training.evaluate_callbacks(
        callback_fns, images, training_prompts, prompt_metadata
    )
    rewards, metadata = infos[args.filter_field]
    all_rewards.append(rewards.squeeze())
    avg(rewards.mean().item())
    mask = masker(rewards)

    print(rewards.squeeze())

    batch = {
        "inference_prompts": inference_prompts,
        "training_prompts": training_prompts,
        "images": images,
        **{key: rew for key, (rew, _) in infos.items()},
    }

    n_added = writer.add_batch(batch, mask=mask)

    n_steps += 1
    n_samples += utils.worker_sum(n_added)

    print(
        f"[ sample ] batch {n_steps} " f"/ {args.max_steps} | "
        if args.max_steps
        else "| "
        f" saved: {n_added} | "
        f" total: {int(n_samples)} / {args.max_samples} | "
        f"average: {avg.avg:.3f} | "
        f"mask: {masker} |"
        f"saving: {timer():.2f} seconds\n"
    )

    if args.max_steps is not None and n_steps >= args.max_steps:
        break
    if args.max_samples is not None and n_samples >= args.max_samples:
        break

writer.close()

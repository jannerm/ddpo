import jax
import jax.numpy as jnp
from diffusers.models import vae_flax


def train_step(
    state,
    text_encoder_params,
    batch,
    train_rng,
    noise_scheduler_state,
    static_broadcasted,
    weights=None,
):
    noise_scheduler, text_encoder, train_cfg, guidance_scale = static_broadcasted
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

    def compute_loss(params):
        latent_dist = vae_flax.FlaxDiagonalGaussianDistribution(batch["vae"])
        latents = latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        batch_size = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            shape=(batch_size,),
            minval=0,
            maxval=noise_scheduler.config.num_train_timesteps,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state,
            latents,
            noise,
            timesteps,
        )

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        # Predict the noise residual and compute loss
        unet_outputs = state.apply_fn(
            {"params": params},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            train=True,
        )

        if train_cfg:
            # Get the text embedding for null conditioning
            uncond_hidden_states = text_encoder(
                batch["uncond_text"],
                params=text_encoder_params,
                train=False,
            )[0]

            uncond_outputs = state.apply_fn(
                {"params": params},
                noisy_latents,
                timesteps,
                uncond_hidden_states,
                train=True,
            )
            noise_pred = uncond_outputs.sample + guidance_scale * (
                unet_outputs.sample - uncond_outputs.sample
            )
        else:
            noise_pred = unet_outputs.sample

        loss = ((noise - noise_pred) ** 2).mean(axis=range(1, noise.ndim))
        if weights is None:
            ## average over batch dimension
            loss = loss.mean()
        else:
            ## multiply loss by weights
            assert loss.size == weights.size
            loss = (loss * weights).sum()

        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)

    loss = jax.lax.pmean(loss, "batch")
    grad = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad)

    return new_state, loss, new_train_rng


def vae_decode(latents, vae_params, apply_fn, decode_fn):
    ## cannot pass in pipeline.vae directly with static_broadcasted_argnums
    ## because it is not hashable;
    ## expects latents in NCHW format (batch_size, 4, 64, 64)
    latents = latents / 0.18215
    images = apply_fn({"params": vae_params}, latents, method=decode_fn).sample
    images = (images / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return images


def text_encode(input_ids, params, text_encoder):
    return text_encoder(input_ids, params=params)[0]


def patch_scheduler(pipeline):
    from ddpo import patch

    pipeline.scheduler = patch.scheduling_ddim_flax.FlaxDDIMScheduler(
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
        beta_start=pipeline.scheduler.config.beta_start,
        beta_end=pipeline.scheduler.config.beta_end,
        beta_schedule=pipeline.scheduler.config.beta_schedule,
        trained_betas=pipeline.scheduler.config.trained_betas,
        set_alpha_to_one=pipeline.scheduler.config.set_alpha_to_one,
        steps_offset=pipeline.scheduler.config.steps_offset,
        prediction_type=pipeline.scheduler.config.prediction_type,
    )

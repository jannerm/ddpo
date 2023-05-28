from . import user

base = {
    "sample": {
        "loadpath": "f:models/{iteration}",
        "savepath": "f:samples/{iteration}",
        "load_epoch": "latest",
        "n_samples_per_device": 4,
        "pretrained_model": "duongna/stable-diffusion-v1-4-flax",
        "prompt_kwargs": {},
        "n_inference_steps": 50,
        "eta": 1.0,
        "resolution": 512,
        "max_samples": 50e3,
        "max_steps": None,
        "local_size": 320,
        "guidance_scale": 5.0,
        "filter_field": "labels",
        "mask_mode": "streaming_percentile",
        "mask_param": 95,
        "identical_batch": False,
        "iteration": 0,
        "evaluate": False,
        "cache": "cache",
        "seed": None,
    },
    "sizes": {
        "loadpath": "f:samples/{iteration}",
        "iteration": 0,
    },
    "train": {
        "modelpath": "f:models/{iteration}",
        "loadpath": "f:samples/{iteration}",
        "savepath": "f:models/{iteration+1}",
        "pretrained_model": "duongna/stable-diffusion-v1-4-flax",
        "finetuned_model": None,
        "load_epoch": "latest",
        "max_train_samples": None,
        "resolution": 512,
        "train_cfg": False,
        "guidance_scale": None,
        "train_batch_size": 2,
        "num_train_epochs": 40,
        "max_train_steps": None,
        "learning_rate": 1e-5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-4,
        "epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "iteration": 0,
        "weighted_batch": False,
        "weighted_dataset": False,
        "dtype": "float32",
        "cache": "cache",
        "verbose": False,
        "save_freq": 100,
        "per_prompt_weights": False,
        "seed": 0,
    },
    "pg": {
        # misc
        "loadpath": "",
        "load_epoch": "latest",
        "modelpath": "models/pg",
        "savepath": "f:models/pg",
        "pretrained_model": "duongna/stable-diffusion-v1-4-flax",
        "resolution": 512,
        "filter_field": None,
        "guidance_scale": 7.5,
        "dtype": "float32",
        "cache": "cache",
        "verbose": False,
        "seed": 0,
        "iteration": 0,
        # sampling
        "sample_batch_size": 2,  # per device
        "num_sample_batches_per_epoch": 8,
        "n_inference_steps": 50,
        "identical_batch": False,
        "evaluate": False,
        "eta": 1.0,
        # training
        "train_batch_size": 2,  # per device
        "train_accumulation_steps": 4,
        "num_train_epochs": 1000,
        "num_inner_epochs": 1,  # inner epochs of PPO training (# of times to loop over collected data)
        "ppo_clip_range": 1e-4,
        "train_cfg": True,
        "learning_rate": 1e-5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-4,
        "epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "save_freq": 100,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "prompt_kwargs": {},
    },
}


compressed_animals = {
    "common": {
        "logbase": f"{user.bucket}/logs/identical-compressed-animals-s1024-p90",
        "prompt_fn": "imagenet_animals",
        "filter_field": "jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "n_samples_per_device": 16,
        "max_samples": 1024,
        "mask_mode": "percentile",
        "mask_param": 90,
        "identical_batch": True,
    },
    "train": {
        "train_cfg": True,
        "train_batch_size": 4,
        "num_train_epochs": 50,
        "save_freq": 20,
        "dtype": "float32",
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

neg_compressed_animals = {
    "common": {
        "logbase": f"{user.bucket}/logs/identical-neg-compressed-animals-s1024-p90",
        "prompt_fn": "imagenet_animals",
        "filter_field": "neg_jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 1024,
        "mask_mode": "percentile",
        "mask_param": 90,
        "identical_batch": True,
    },
    "train": {
        "train_cfg": True,
        "train_batch_size": 1,
        "num_train_epochs": 50,
        "save_freq": 20,
        "dtype": "float32",
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

compressed_animals_rwr = {
    "common": {
        "logbase": f"{user.bucket}/logs/rwr-compressed-animals-s10k",
        "prompt_fn": "imagenet_animals",
        "filter_field": "jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 10240,
        "mask_mode": "streaming_percentile",
        "mask_param": 0,  ## save all samples
        "identical_batch": False,
        "local_size": 1600,
    },
    "calibrate": {},
    "train": {
        "train_cfg": True,
        "train_batch_size": 1,
        "num_train_epochs": 5,  ## same number of gradient steps as filter-finetune
        "save_freq": 20,
        "dtype": "float32",
        "weighted_dataset": True,
        "temperature": 1 / 5.0,
    },
    "pg": {},
}

neg_compressed_animals_rwr = {
    "common": {
        "logbase": f"{user.bucket}/logs/rwr-neg-compressed-animals-s10k",
        "prompt_fn": "imagenet_animals",
        "filter_field": "neg_jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 10240,
        "mask_mode": "streaming_percentile",
        "mask_param": 0,  ## save all samples
        "identical_batch": False,
        "local_size": 1600,
    },
    "calibrate": {},
    "train": {
        "train_cfg": True,
        "train_batch_size": 1,
        "num_train_epochs": 5,  ## same number of gradient steps as filter-finetune
        "save_freq": 20,
        "dtype": "float32",
        "weighted_dataset": True,
        "temperature": 1 / 5.0,
    },
    "pg": {},
}

vqa_v0 = {
    "common": {
        "logbase": f"{user.bucket}/logs/vqa-v0-n2k-s5.0-e50",
        "prompt_fn": "vqa_dataset",
        "prompt_kwargs": {"loadpath": "assets/vqa_v0.txt"},
        "filter_field": "vqa",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 2e3,
        "mask_mode": "threshold",
        "mask_param": 0.65,
        "identical_batch": False,
        # 'local_size': 16,
    },
    "calibrate": {},
    "train": {
        "train_cfg": True,
        "train_batch_size": 1,
        "num_train_epochs": 50,
        "save_freq": 20,
    },
}

# ------------------------------------------------------------------------------#

rotational_pg = {
    "common": {
        "logbase": f"{user.bucket}/logs/rotational-animal-400-pg",
        "prompt_fn": "imagenet_animals",
        "filter_field": "rotational",
        "guidance_scale": 5.0,
        # 'sample_batch_size': 8,
    },
}

rotational_corr_pg = {
    "common": {
        "logbase": f"{user.bucket}/logs/rotational-corr-400-pg",
        "prompt_fn": "imagenet_animals",
        "filter_field": "rotational_corr",
        "guidance_scale": 5.0,
    },
}

mirror_pg = {
    "common": {
        "logbase": f"{user.bucket}/logs/mirror-pg-400-pod",
        "prompt_fn": "imagenet_animals",
        "filter_field": "mirror",
        "guidance_scale": 5.0,
        # 'sample_batch_size': 8,
    },
}

thumbnail_pg = {
    "common": {
        "logbase": f"{user.bucket}/logs/thumbnail-animal-400-pg",
        "prompt_fn": "imagenet_animals",
        "filter_field": "thumbnail",
        "guidance_scale": 5.0,
    },
}

aesthetic_pg = {
    "common": {
        "logbase": f"{user.bucket}/logs/aesthetic-animal-400-pg",
        "prompt_fn": "imagenet_animals",
        "filter_field": "aesthetic",
        "guidance_scale": 5.0,
    },
}

llava_vqa = {
    "common": {
        "logbase": f"{user.bucket}/logs/llava-vqa-v2",
        "prompt_fn": "vqa_dataset",
        "prompt_kwargs": {"loadpath": "assets/vqa_v2.txt"},
        "filter_field": "llava_vqa",
        "guidance_scale": 5.0,
    },
    "pg": {
        "per_prompt_stats_bufsize": 128,
        "per_prompt_stats_min_count": 32,
        "num_train_epochs": 120,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 2,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

llava_counting = {
    "common": {
        "logbase": f"{user.bucket}/logs/llava-counting-v0-8",
        "prompt_fn": "counting",
        "prompt_kwargs": {
            "nouns_path": "assets/very_simple_animals.txt",
            "number_range": (2, 8),
        },
        "filter_field": "llava_vqa",
        "guidance_scale": 5.0,
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 2,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

llava_bertscore = {
    "common": {
        "logbase": f"{user.bucket}/logs/llava-bertscore-2-simple-animals",
        "prompt_fn": "nouns_activities",
        "prompt_kwargs": {
            "nouns_path": "assets/common_animals.txt",
            "activities_path": "assets/activities_v0.txt",
        },
        "filter_field": "llava_bertscore",
        "guidance_scale": 5.0,
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 2,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 20,
    },
}

a_dog_1 = {
    "common": {
        "logbase": f"{user.bucket}/logs/aesthetic_dogs_sweep/one",
        "prompt_fn": "manual",
        "prompt_kwargs": {"prompts": ["a dog"]},
        "filter_field": "aesthetic",
        "guidance_scale": 5.0,
    },
    "pg": {
        "per_prompt_stats_bufsize": None,
        "per_prompt_stats_min_count": None,
        "num_train_epochs": 100,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

a_dog_2 = {
    "common": {
        "logbase": f"{user.bucket}/logs/aesthetic_dogs_sweep/imagenet",
        "prompt_fn": "imagenet_dogs",
        "prompt_kwargs": {},
        "filter_field": "aesthetic",
        "guidance_scale": 5.0,
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 100,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

a_animals = {
    "common": {
        "logbase": f"{user.bucket}/logs/aesthetic_simple_animals",
        "prompt_fn": "from_file",
        "prompt_kwargs": {"loadpath": "assets/common_animals.txt"},
        "filter_field": "aesthetic",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 1024,
        "mask_mode": "percentile",
        "mask_param": 90,
        "identical_batch": True,
    },
    "train": {
        "train_cfg": True,
        "train_batch_size": 1,
        "num_train_epochs": 50,
        "save_freq": 20,
        "dtype": "float32",
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 2,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 20,
    },
}

a_animals_rwr = {
    "common": {
        "logbase": f"{user.bucket}/logs/aesthetic_simple_animals_rwr_ppb",
        "prompt_fn": "from_file",
        "prompt_kwargs": {"loadpath": "assets/common_animals.txt"},
        "filter_field": "aesthetic",
        "guidance_scale": 5.0,
    },
    "sample": {
        "n_samples_per_device": 16,
        "max_samples": 10240,
        "mask_mode": "streaming_percentile",
        "mask_param": 0,  ## save all samples
        "identical_batch": False,
        "local_size": 1600,
    },
    "train": {
        "train_cfg": True,
        "train_batch_size": 4,
        "num_train_epochs": 5,  ## same number of gradient steps as filter-finetune
        "save_freq": 10000000,
        "dtype": "float32",
        "weighted_dataset": True,
        "temperature": 1 / 5.0,
        "per_prompt_weights": True,
    },
    "pg": {
        "per_prompt_stats_bufsize": 32,
        "per_prompt_stats_min_count": 16,
        "num_train_epochs": 200,
        "sample_batch_size": 8,
        "num_sample_batches_per_epoch": 1,
        "train_batch_size": 1,
        "train_accumulation_steps": 4,
        "num_inner_epochs": 1,
        "ppo_clip_range": 1e-4,
        "learning_rate": 1e-5,
        "optimizer": "adamw",
        "train_timestep_ratio": 1.0,
        "save_freq": 10,
    },
}

compressed_animals_nocfg = {
    "common": {
        "logbase": f"{user.bucket}/logs/nocfg-compressed-animals-s1024-p90",
        "prompt_fn": "imagenet_animals",
        "filter_field": "jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 1024,
        "mask_mode": "percentile",
        "mask_param": 90,
        "identical_batch": True,
    },
    "calibrate": {},
    "train": {
        "train_cfg": False,
        "train_batch_size": 2,
        "num_train_epochs": 50,
        "save_freq": 20,
        "dtype": "float32",
    },
    "pg": {},
}

neg_compressed_animals_nocfg = {
    "common": {
        "logbase": f"{user.bucket}/logs/nocfg-neg-compressed-animals-s1024-p90",
        "prompt_fn": "imagenet_animals",
        "filter_field": "neg_jpeg",
        "guidance_scale": 5.0,
    },
    "sample": {
        "max_samples": 1024,
        "mask_mode": "percentile",
        "mask_param": 90,
        "identical_batch": True,
    },
    "calibrate": {},
    "train": {
        "train_cfg": False,
        "train_batch_size": 2,
        "num_train_epochs": 50,
        "save_freq": 20,
        "dtype": "float32",
    },
    "pg": {},
}

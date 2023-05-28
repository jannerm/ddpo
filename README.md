# Denoising Diffusion Policy Optimization

Training code for the paper [Training Diffusion Models with Reinforcement Learning](https://rl-diffusion.github.io/).
This codebase has been tested on [Google Cloud TPUs](https://cloud.google.com/tpu) (v3 for RWR and v4 for DDPO); it has not been tested on GPUs.

| `prompt_fn` | `filter_field` | Weights and Demo |
| --- | --- | --- |
| `imagenet_animals` | `jpeg` | [ddpo-compressibility](https://huggingface.co/kvablack/ddpo-compressibility) |
| `imagenet_animals` | `neg_jpeg` | [ddpo-incompressibility](https://huggingface.co/kvablack/ddpo-incompressibility) |
| `from_file(assets/common_animals.txt)` | `aesthetic` | [ddpo-aesthetic](https://huggingface.co/kvablack/ddpo-aesthetic) |
| `nouns_activities(assets/common_animals.txt, assets/activities_v0.txt)` | `llava_bertscore` | [ddpo-alignment](https://huggingface.co/kvablack/ddpo-alignment) |


## Installation

```
conda env create -f environment_tpu.yml
conda activate ddpo-tpu
pip install -e .
```

## Running DDPO
```
python pipeline/policy_gradient.py --dataset compressed-animals
```

The `--dataset` flag can be replaced by any of the configs defined in `config/base.py`.
The first config dict, `base`, defines common arguments that are overridden in specific configs further down.
Some arguments are shared between methods; DDPO-specific hyperparameters are in the `"pg"` field.

The most important arguments are `prompt_fn` and `filter_field`, which define the prompt distribution and reward function, respectively.
See `training/prompts.py` for prompt functions and `training/callbacks.py` for reward functions.

## Running RWR
For standard RWR, where the weights are a softmax of the rewards:
```
bash pipeline/run-rwr.sh
```

For RWR-sparse, where only samples above a certain percentile of the reward distribution are kept and trained on:
```
bash pipeline/run-sparse.sh
```

These methods run the outermost training loop in bash rather than Python. They run the `pipeline/sample.py` script to collect a dataset of samples and rewards, run `pipeline/finetune.py` to train the model on the most recent dataset, and repeat for some number of iterations. The sampling step and finetuning step have different configs, which are labeled `"sample"` and `"train"`, respectively, in `config/base.py`.

## Running LLaVA Inference
LLaVA inference was performed by making HTTP requests to a separate GPU server. See the `llava_bertscore` reward function in `training/callbacks.py` for the client-side code, and [this repo](https://github.com/kvablack/LLaVA-server/) for the server-side code.

## Reference
```
@inproceedings{black2023ddpo,
      title={Training Diffusion Models with Reinforcement Learning},
      author={Kevin Black and Michael Janner and Yilun Du and Ilya Kostrikov and Sergey Levine},
      year={2023},
      eprint={2305.13301},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

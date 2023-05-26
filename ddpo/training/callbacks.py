import os
from functools import partial
import random
import sys
import numpy as np
import jax
import jax.numpy as jnp
import torch
from PIL import Image, ImageOps
import diffusers
import transformers
import pdb

from ddpo.models import laion
from ddpo import utils

DEVICES = jax.local_devices()


def shard_unshard(fn):
    def _wrapper(images, prompts, metadata):
        del prompts, metadata
        sharded = utils.shard(images)
        output = fn(sharded)
        return utils.unshard(output)

    return _wrapper


def normalize(x, mean, std):
    """
    mirrors `torchvision.transforms.Normalize(mean, std)`
    """
    return (x - mean) / std


def vae_fn(devices=DEVICES, dtype="float32", jit=True):
    vae, vae_params = diffusers.FlaxAutoencoderKL.from_pretrained(
        "duongna/stable-diffusion-v1-4-flax", subfolder="vae", dtype=dtype
    )

    def _fn(images):
        images = images.transpose(0, 3, 1, 2)
        images = normalize(images, mean=0.5, std=0.5)
        vae_outputs = vae.apply(
            {"params": vae_params},
            images,
            deterministic=True,
            method=vae.encode,
        )
        gaussian = vae_outputs.latent_dist
        return jnp.concatenate([gaussian.mean, gaussian.logvar], axis=-1), {}

    if jit:
        _fn = jax.pmap(_fn, axis_name="batch", devices=devices)

    return shard_unshard(_fn)


def aesthetic_fn(devices=DEVICES, rng=0, cache="cache", jit=True):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = transformers.CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    rng = jax.random.PRNGKey(rng)
    embed_dim = 768
    classifier = laion.AestheticClassifier()
    params = classifier.init(rng, jnp.ones((1, embed_dim)))

    repo_path = utils.get_repo_path()
    weights = laion.load_weights(cache=os.path.join(repo_path, cache))
    params = laion.set_weights(params, weights)

    def _fn(images):
        image_features = model.get_image_features(pixel_values=images)

        # normalize image features
        norm = jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        image_features = image_features / norm

        score = classifier.apply(params, image_features)
        return score

    if jit:
        _fn = jax.pmap(_fn, axis_name="batch", devices=devices)

    def _wrapper(images, prompts, metadata):
        del metadata
        inputs = processor(images=list(images), return_tensors="np")
        inputs = utils.shard(inputs["pixel_values"])
        output = _fn(inputs)
        return utils.unshard(output), {}

    return _wrapper


def diversity_fn(devices=DEVICES, jit=False):
    assert not jit

    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = transformers.CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    def _fn(images, prompts):
        del prompts
        inputs = processor(images=list(images), return_tensors="np")
        features = model.get_image_features(pixel_values=inputs["pixel_values"])

        n_images = len(features)
        n_pairs = 10000
        idx1 = np.random.randint(0, n_images, (n_pairs,))
        idx2 = np.random.randint(0, n_images, (n_pairs,))
        dist = np.linalg.norm(features[idx1] - features[idx2], axis=-1)
        avg = dist.mean()
        return avg, {}

    return _fn


def consistency_fn(devices=DEVICES, jit=False):
    assert not jit

    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def _fn(images, prompts, metadata):
        del metadata
        inputs = processor(
            text=prompts, images=list(images), return_tensors="np", padding=True
        )
        outputs = model(**inputs)
        ## [ n_images, n_prompts ]
        logits = outputs.logits_per_image
        return logits.diagonal()[:, None], {}

    return _fn


def jpeg_fn(devices=DEVICES, jit=False):
    assert not jit

    def _fn(images, prompts, metadata):
        del prompts, metadata
        lengths = [len(utils.encode_jpeg(image)) for image in images]
        ## uint8 : 1 byte (8 bits) per dim
        sizes_kb = [length / 1000.0 for length in lengths]
        return -np.array(sizes_kb)[:, None], {}

    return _fn


def neg_jpeg_fn(*args, **kwargs):
    _jpeg = jpeg_fn(*args, **kwargs)

    def _fn(*args, **kwargs):
        scores, infos = _jpeg(*args, **kwargs)
        return -scores, infos

    return _fn


def rotational_symmetry_fn(devices=DEVICES, jit=True):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = transformers.CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    def _fn(images):
        image_features = model.get_image_features(pixel_values=images)
        return image_features

    if jit:
        _fn = jax.pmap(_fn, axis_name="batch", devices=devices)

    def _wrapper(images, prompts, metadata):
        del metadata
        pils = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        degrees = [0, 90, 180, 270]
        rotated = []
        for rot in degrees:
            rotated += [pil.rotate(rot) for pil in pils]

        inputs = processor(images=list(rotated), return_tensors="np")
        inputs = utils.shard(inputs["pixel_values"])
        output = _fn(inputs)
        output = utils.unshard(output)

        embed_dim = output.shape[-1]
        output = output.reshape(-1, len(images), embed_dim)

        scores = 0
        for i in range(1, len(output)):
            numer = (output[0] * output[i]).sum(axis=-1)
            denom = np.linalg.norm(output[0], axis=-1) * np.linalg.norm(
                output[i], axis=-1
            )
            theta = np.arccos(np.clip(numer / denom, a_min=0, a_max=1))
            theta = theta * 180 / np.pi
            scores += theta

        ## take average angle over all rotations
        scores /= len(output) - 1

        ## negate scores so that lower angles are better
        scores = -scores

        return scores, {}

    return _wrapper


def rotational_correlation_fn(devices=DEVICES, jit=False):
    def _wrapper(images, prompts, metadata):
        del metadata
        pils = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        degrees = [0, 180]
        rotated = []
        for rot in degrees:
            rotated += [pil.rotate(rot) for pil in pils]

        output = np.array([np.array(img) for img in rotated])
        output = output.reshape(len(degrees), *images.shape)

        scores = 0
        for i in range(1, len(output)):
            mse = ((output[0] - output[i]) ** 2).mean(axis=(1, 2, 3))
            scores += mse

        ## take average angle over all rotations
        scores /= len(output) - 1

        ## negate scores so that lower mse is better
        scores = -scores

        return scores, {}

    return _wrapper


def mirror_symmetry_fn(devices=DEVICES, jit=False):
    def _wrapper(images, prompts, metadata):
        del metadata
        pils = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        mirrored = [ImageOps.mirror(img) for img in pils]

        pils = np.array([np.array(img) for img in pils])
        mirrored = np.array([np.array(img) for img in mirrored])

        scores = ((pils - mirrored) ** 2).mean(axis=(1, 2, 3))

        ## negate scores so that lower mse is better
        scores = -scores

        return scores, {}

    return _wrapper


def cov(X, Y):
    assert X.ndim == Y.ndim == 2
    return ((X - X.mean(-1, keepdims=True)) * (Y - Y.mean(-1, keepdims=True))).sum(-1)


def mirror_correlation_fn(devices=DEVICES, jit=False):
    def _wrapper(images, prompts, metadata):
        del metadata
        pils = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        mirrored = [ImageOps.mirror(img) for img in pils]

        pils = np.array([np.array(img) for img in pils])
        mirrored = np.array([np.array(img) for img in mirrored])

        pils = (pils / 255.0).astype(np.float32).reshape(len(images), -1)
        mirrored = (mirrored / 255.0).astype(np.float32).reshape(len(images), -1)

        scores = cov(pils, mirrored) / np.sqrt(
            cov(pils, pils) * cov(mirrored, mirrored)
        )

        ## sanity check
        ## assert np.isclose(scores[0], np.corrcoef(pils[0], mirrored[0])[0][1])

        ## negate scores so that lower correlation is better
        scores = -scores

        return scores, {}

    return _wrapper


def thumbnail_fn(devices=DEVICES, jit=True):
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = transformers.CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    def _fn(images):
        image_features = model.get_image_features(pixel_values=images)
        return image_features

    if jit:
        _fn = jax.pmap(_fn, axis_name="batch", devices=devices)

    def _wrapper(images, prompts, metadata):
        del metadata
        pils = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        downsampled = list(pils)
        downsample = [4, 8, 16]

        for d in downsample:
            size = (pils[0].size[0] // d, pils[0].size[1] // d)
            downsampled += [pil.resize(size) for pil in pils]

        inputs = processor(images=list(downsampled), return_tensors="np")
        inputs = utils.shard(inputs["pixel_values"])
        output = _fn(inputs)
        output = utils.unshard(output)

        embed_dim = output.shape[-1]
        output = output.reshape(-1, len(images), embed_dim)

        scores = 0
        for i in range(1, len(output)):
            numer = (output[0] * output[i]).sum(axis=-1)
            denom = np.linalg.norm(output[0], axis=-1) * np.linalg.norm(
                output[i], axis=-1
            )
            theta = np.arccos(np.clip(numer / denom, a_min=0, a_max=1))
            theta = theta * 180 / np.pi
            scores += theta

        ## take average angle over all rotations
        scores /= len(output) - 1

        ## negate scores so that lower angles are better
        scores = -scores

        return scores, {}

    return _wrapper


def arange_fn(devices=DEVICES, jit=False):
    assert not jit

    def _fn(images, prompts, metadata):
        del prompts, metadata
        return np.arange(len(images))[:, None], {}

    return _fn


def single_satisfaction(outputs, answers):
    assert len(outputs) == len(answers)
    correct = [ans in output for ans, output in zip(answers, outputs)]
    return np.array(correct, dtype=int)


def vqa_satisfaction(devices=DEVICES, jit=False):
    device = "cpu"
    dtype = torch.float32
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    vlm = transformers.Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )

    def _fn(images, prompts, metadata):
        n_questions = len(metadata[0]["questions"])
        images = (images * 255).astype(np.uint8)

        questions = [
            f'Question: {m["questions"][i]} Answer:'
            for m in metadata
            for i in range(n_questions)
        ]
        answers = [m["answers"][i] for m in metadata for i in range(n_questions)]
        images_rep = [img for img in images for _ in range(n_questions)]

        inputs = processor(
            images_rep,
            text=questions,
            return_tensors="pt",
            padding="longest",
        ).to(device, dtype)
        generated_ids = vlm.generate(**inputs, max_new_tokens=8)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [text.strip() for text in generated_text]

        correct = single_satisfaction(generated_text, answers)
        scores = correct.reshape(len(images), n_questions).mean(-1, keepdims=True)

        return scores, {}

    return _fn


def llava_vqa_satisfaction(devices=DEVICES, jit=False):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle
    from tqdm import tqdm

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        images = (images * 255).astype(np.uint8)

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in tqdm(
            list(zip(images_batched, metadata_batched)), desc="LLaVA", position=1
        ):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = [
                single_satisfaction(ans, m["answers"])
                for ans, m in zip(response_data["outputs"], metadata_batch)
            ]
            scores = np.array(correct).mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore(devices=DEVICES, jit=False):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle
    from tqdm import tqdm

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        images = (images * 255).astype(np.uint8)

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in tqdm(
            list(zip(images_batched, prompts_batched)),
            desc="LLaVA",
            position=1,
            leave=False,
        ):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def evaluate_callbacks(fns, images, prompts, metadata):
    if type(prompts[0]) == list:
        prompts = [random.choice(p) for p in prompts]

    images = images.astype(jnp.float32)
    outputs = {key: fn(images, prompts, metadata) for key, fn in fns.items()}
    return outputs


callback_fns = {
    "vae": vae_fn,
    "aesthetic": aesthetic_fn,
    "consistency": consistency_fn,
    "jpeg": jpeg_fn,
    "neg_jpeg": neg_jpeg_fn,
    "rotational": rotational_symmetry_fn,
    "rotational_corr": rotational_correlation_fn,
    "mirror": mirror_symmetry_fn,
    "mirror_corr": mirror_correlation_fn,
    "thumbnail": thumbnail_fn,
    "arange": arange_fn,
    "vqa": vqa_satisfaction,
    "llava_vqa": llava_vqa_satisfaction,
    "llava_bertscore": llava_bertscore,
}

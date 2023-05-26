import random
import numpy as np
import pdb

from ddpo import utils
from ddpo.utils import imagenet
import inflect

inflect_engine = inflect.engine()

# --------------------------------- general api --------------------------------#


def batchify(prompt_fn, batch_size, **kwargs):
    inference_prompts, training_prompts, prompt_metadata = zip(
        *[prompt_fn(**kwargs) for _ in range(batch_size)]
    )
    return list(inference_prompts), training_prompts, prompt_metadata


def batchify_identical(prompt_fn, batch_size, **kwargs):
    inference_prompt, training_prompts, prompt_metadata = prompt_fn(**kwargs)
    inference_batch = [inference_prompt for _ in range(batch_size)]
    training_batch = [training_prompts for _ in range(batch_size)]
    metadata_batch = [prompt_metadata for _ in range(batch_size)]
    return inference_batch, training_batch, metadata_batch


def make_prompts(fn_name, batch_size, identical_batch=False, **kwargs):
    prompt_fn = globals()[fn_name]
    if identical_batch:
        return batchify_identical(prompt_fn, batch_size, **kwargs)
    else:
        return batchify(prompt_fn, batch_size, **kwargs)


# ---------------------------- specific experiments ----------------------------#


def person_pet(evaluate=False):
    training_prompts = ["a photo of a person with their pet"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def consistent_animals(evaluate=False):
    inference_prompt = "a husky and a shoebill stork on the beach in a single image"
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def get_random_class(idx=None, low=None, high=None):
    if idx is not None:
        label = imagenet.classes[idx]
    elif low is not None and high is not None:
        idx = random.randint(low, high)
        label = imagenet.classes[idx]
    else:
        label = random.choice(imagenet.classes)
    # if ',' in label:
    #     label = label.split(',')[0]
    return label


def consistent_imagenet_animals(colors=False):
    class1 = get_random_class()
    class2 = get_random_class()
    if colors:
        inference_prompt = (
            f"a realistic photo of a {random.choice(imagenet.colors)} {class1} and "
            f"a {random.choice(imagenet.colors)} {class2}"
        )
    else:
        inference_prompt = f"a realistic photo of a {class1} and a {class2}"
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def consistent_imagenet_animals_3(colors=False):
    class1 = get_random_class()
    class2 = get_random_class()
    class3 = get_random_class()
    if colors:
        inference_prompt = (
            f"a realistic photo of a {random.choice(imagenet.colors)} {class1}, "
            f"a {random.choice(imagenet.colors)} {class2}, and "
            f"a {random.choice(imagenet.colors)} {class3}"
        )
    else:
        inference_prompt = (
            f"a realistic photo of a {class1}, a {class2}, and a {class3}"
        )
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def n_fingers(evaluate=False):
    n = random.randint(1, 4)
    inference_prompt = f'a photo of a hand holding up {n} finger{"s" if n > 1 else ""}'
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def imagenet_single(evaluate=False, idx=None):
    class1 = get_random_class(idx=idx)
    inference_prompt = f"a realistic photo of a {class1}"
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def imagenet_aesthetic(evaluate=False):
    class1 = get_random_class()
    training_prompts = [f"a realistic photo of a {class1}"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def imagenet_simple(evaluate=False, idx=None):
    class1 = get_random_class(idx=idx)
    inference_prompt = f"a {class1}"
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def imagenet_dogs(evaluate=False, idx=None):
    class1 = get_random_class(idx=idx, low=151, high=268)
    training_prompts = [f"{class1}"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def animal_debug(evaluate=False, idx=None):
    training_prompts = ["a peacock"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def simple_dogs(evaluate=False, idx=None):
    class1 = get_random_class(idx=idx, low=151, high=268)
    training_prompts = [f"{class1}"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def imagenet_animals(evaluate=False, idx=None):
    class1 = get_random_class(idx=idx, low=0, high=397)
    training_prompts = [f"{class1}"]
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def from_file(loadpath, evaluate=False, idx=None):
    prompts = utils.load_lines(loadpath)
    if idx is not None:
        inference_prompt = prompts[idx]
    else:
        inference_prompt = random.choice(prompts)
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def vqa_dataset(loadpath, max_samples=None, evaluate=False):
    dataset = utils.load_general_prompts(loadpath)
    entry = random.choice(dataset)
    training_prompts = [entry["prompt"]]
    inference_prompt = entry["prompt"]
    metadata = entry
    return inference_prompt, training_prompts, metadata


def manual(prompts, evaluate=False):
    training_prompts = prompts
    inference_prompt = random.choice(training_prompts)
    return inference_prompt, training_prompts, {}


def nouns_activities(nouns_path, activities_path, evaluate=False):
    nouns = utils.load_lines(nouns_path)
    activities = utils.load_lines(activities_path)
    inference_prompt = (
        f"{inflect_engine.a(random.choice(nouns))} {random.choice(activities)}"
    )
    training_prompts = [inference_prompt]
    return inference_prompt, training_prompts, {}


def counting(nouns_path, number_range, evaluate=False):
    nouns = utils.load_lines(nouns_path)
    number = inflect_engine.number_to_words(random.randint(*number_range))
    noun = random.choice(nouns)
    plural_noun = inflect_engine.plural(noun)
    inference_prompt = f"{number} {plural_noun}"
    training_prompts = [inference_prompt]
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return inference_prompt, training_prompts, metadata

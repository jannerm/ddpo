import os
import importlib
import random
import numpy as np
import jax
import torch
from tap import Tap
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)
from .logger import init_logging
from .filesystem import is_remote


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = "_".join(f"{k}-{v}" for k, v in val.items())
            exp_name.append(f"{label}{val}")
        exp_name = "_".join(exp_name)
        exp_name = exp_name.replace("/_", "/")
        exp_name = exp_name.replace("(", "").replace(")", "")
        exp_name = exp_name.replace(", ", "-")
        return exp_name

    return _fn


def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")


class LazyFn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return self.fn(args)


class Parser(Tap):
    def save(self):
        fullpath = os.path.join(self.savepath, "args.json")
        print(f"[ utils/setup ] Saved args to {fullpath}")
        super().save(fullpath, skip_unpicklable=True)

    def report(self):
        tab = " " * 8
        string = f"[ utils/setup ] Parser [ {self.dataset} ]"
        for key, val in self._dict.items():
            string += f"\n{tab}{key}: {val}"
        print(string, "\n")

    def parse_args(self, experiment=None):
        args = super().parse_args(known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, "config"):
            return args
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.eval_fns(args)
        self.eval_fstrings(args)
        self.get_commit(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        self.init_logging(args)
        self.set_seed(args)
        self.save_diff(args)
        self.report()
        return args

    def read_config(self, args, experiment):
        """
        Load parameters from config file
        """
        dataset = args.dataset.replace("-", "_")
        print(f"[ utils/parser ] Reading config: {args.config}:{dataset}:{experiment}")
        module = importlib.import_module(args.config)
        params = getattr(module, "base")[experiment]

        if hasattr(module, dataset):
            print(
                f"[ utils/parser ] Using overrides | config: {args.config} | dataset: {dataset}"
            )
            dataset_dict = getattr(module, dataset)
            dataset_common = dataset_dict.get("common", {})
            dataset_overrides = dataset_dict.get(experiment, {})
            params.update(dataset_common)
            params.update(dataset_overrides)
        else:
            print(
                f"[ utils/parser ] Not using overrides | config: {args.config} | dataset: {dataset}"
            )

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        """
        Override config parameters with command-line arguments
        """
        extras = args.extra_args
        if not len(extras):
            return

        print(f"[ utils/setup ] Found extras: {extras}")
        assert (
            len(extras) % 2 == 0
        ), f"Found odd number ({len(extras)}) of extras: {extras}"
        for i in range(0, len(extras), 2):
            key = extras[i].replace("--", "")
            val = extras[i + 1]
            assert hasattr(
                args, key
            ), f"[ utils/setup ] {key} not found in config: {args.config}"
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f"[ utils/setup ] Overriding config | {key} : {old_val} --> {val}")
            if val == "None":
                val = None
            elif val == "latest":
                val = "latest"
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(
                        f"[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str"
                    )
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == "f:":
                val = old.replace("{", "{args.").replace("f:", "")
                new = lazy_fstring(val, args)
                print(f"[ utils/setup ] Lazy fstring | {key} : {old} --> {new}")
                setattr(self, key, new)
                self._dict[key] = new

    def eval_fns(self, args):
        for key, old in self._dict.items():
            if isinstance(old, LazyFn):
                new = old(args)
                print(f"[ utils/setup ] Lazy fn | {key} : {new}")
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, "seed") or args.seed is None:
            args.seed = np.random.randint(0, int(1e6))
        args.seed = args.seed + jax.process_index()
        print(f"[ utils/setup ] Setting seed: {args.seed}")
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, "loadbase") and args.loadbase is None:
            print(f"[ utils/setup ] Setting loadbase: {args.logbase}")
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not "exp_name" in dir(args):
            return
        exp_name = getattr(args, "exp_name")
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f"[ utils/setup ] Setting exp_name to: {exp_name_string}")
            setattr(args, "exp_name", exp_name_string)
            self._dict["exp_name"] = exp_name_string

    def mkdir(self, args):
        if "logbase" in dir(args) and "savepath" in dir(args):
            args.savepath = os.path.join(args.logbase, args.savepath)
            self._dict["savepath"] = args.savepath
            if "suffix" in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if not is_remote(args.savepath) and mkdir(args.savepath):
                print(f"[ utils/setup ] Made savepath: {args.savepath}")
            # self.save()
        for key in ["loadpath", "modelpath"]:
            if "logbase" in dir(args) and key in dir(args):
                if getattr(args, key).startswith("/") or getattr(args, key).startswith(
                    "gs://"
                ):
                    ## absolute path
                    continue
                fullpath = os.path.join(args.logbase, getattr(args, key))
                setattr(args, key, fullpath)
                self._dict[key] = fullpath

    def init_logging(self, args):
        if hasattr(args, "verbose"):
            verbose = args.verbose
        else:
            verbose = False
        init_logging("ddpo", verbose=verbose)

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, "diff.txt"))
        except:
            print("[ utils/setup ] WARNING: did not save git diff")

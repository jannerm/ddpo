import os
import subprocess
import shlex
import jax
import pdb

from ddpo import utils


class Parser(utils.Parser):
    config: str = "config.base"
    dataset: str = "compressed_imagenet"
    override: str = None


args = Parser().parse_args("sizes")

if args.override:
    args.loadpath = args.override

bucket, _ = utils.fs.get_bucket(args.loadpath)
print(f"[ sizes ] {args.loadpath}")

savepath = os.path.join(args.loadpath, "sizes.pkl")

python = f"""
import sys
from ddpo import utils
fname = sys.argv[1]
reader = utils.SlowRemoteReader('{bucket}', fname)
print(fname)
print(len(reader))
"""

wait_every = 100

fnames = utils.fs.ls(args.loadpath)
fnames = [fname for fname in fnames if fname.endswith(".hdf5")]
print(f"Found {len(fnames)} files | {args.loadpath}")

processes = []
for i, fname in enumerate(fnames):
    command = f'python -c "{python}" {fname}'
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(process)
    if i > 0 and i % wait_every == 0:
        print(f"waiting at {i}")
        [process.wait() for process in processes]
        print("done")
        # if i > 0: pdb.set_trace()

print("waiting")
[process.wait() for process in processes]
print("done")

stdouts = [process.communicate()[0].decode() for process in processes]

sizes = []
for stdout in stdouts:
    lines = [line for line in stdout.split("\n") if len(line)]
    try:
        size = int(lines[-1])
        sizes.append(size)
    except:
        pdb.set_trace()

print(f"Found {len(sizes)} sizes | Total: {sum(sizes)}")

assert len(fnames) == len(sizes)
sizes_d = {fname: size for fname, size in zip(fnames, sizes)}

if jax.process_index() == 0:
    utils.fs.save(savepath, sizes_d)

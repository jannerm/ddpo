import logging
import warnings
import numpy as np
import pdb


def init_logging(name, verbose=False):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[ %(name)s ] %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if verbose:
        logging.getLogger("jax").setLevel(logging.INFO)
        logging.getLogger("datasets").setLevel(logging.WARNING)
    else:
        print("[ utils/logger ] Suppressing most dependency logging")
        logging.getLogger("jax").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        logging.getLogger("tcmalloc").setLevel(logging.ERROR)
        warnings.simplefilter(action="ignore", category=FutureWarning)

    return logger


class Masker:
    def __repr__(self):
        return f"[ {self._name} | {self.p} ]"

    def mask(self, xs):
        return xs >= self.p


class StreamingAverage:
    def __init__(self):
        self.n = 0
        self.avg = 0
        self._name = "streaming_average"

    def __call__(self, x):
        self.n += 1
        self.avg = self.avg * (self.n - 1) / self.n + x / self.n


class StreamingPercentile(Masker):
    def __init__(self, q=90, maxsize=5e6):
        self.q = q
        self.xs = np.zeros(int(maxsize))
        self.size = 0
        self._name = f"streaming_percentile: {q}"

    def __call__(self, xs):
        if xs.ndim == 2:
            xs = xs.squeeze(axis=-1)
        n = len(xs)
        self.xs[self.size : self.size + n] = xs[:]
        self.size += n
        self.p = np.percentile(self.xs[: self.size], self.q)
        return super().mask(xs)


class Percentile(Masker):
    def __init__(self, q=90, maxsize=5e6):
        self.q = q
        self._name = f"percentile: {q}"

    def __call__(self, xs):
        if xs.ndim == 2:
            xs = xs.squeeze(axis=-1)
        self.p = np.percentile(xs, self.q)
        return super().mask(xs)


class Threshold(Masker):
    def __init__(self, threshold=0.95):
        self.p = threshold
        self._name = f"threshold: {threshold}"

    def __call__(self, xs):
        return super().mask(xs)


def make_masker(mode, param):
    return {
        "percentile": Percentile,
        "streaming_percentile": StreamingPercentile,
        "threshold": Threshold,
    }[mode](param)


if __name__ == "__main__":
    import numpy as np

    xs = np.random.randn(1000)
    avg = StreamingAverage()
    for i, x in enumerate(xs):
        avg(x)
        assert np.isclose(avg.avg, xs[: i + 1].mean())
    print(f"{avg.avg:.5f} | {xs.mean():.5f}")

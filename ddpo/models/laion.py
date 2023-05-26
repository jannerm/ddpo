import os
import requests
import torch
from flax import linen as nn


class AestheticClassifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1024)(x)
        x = nn.Dropout(0.2)(x, deterministic=True)
        x = nn.Dense(features=128)(x)
        x = nn.Dropout(0.2)(x, deterministic=True)
        x = nn.Dense(features=64)(x)
        x = nn.Dropout(0.1)(x, deterministic=True)
        x = nn.Dense(features=16)(x)
        x = nn.Dense(features=1)(x)
        return x


def load_weights(cache="cache"):
    weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
    loadpath = os.path.join(cache, weights_fname)

    if not os.path.exists(loadpath):
        url = (
            "https://github.com/christophschuhmann/"
            f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
        )
        r = requests.get(url)

        with open(loadpath, "wb") as f:
            f.write(r.content)

    weights = torch.load(loadpath, map_location=torch.device("cpu"))
    return weights


def set_weights(params, loaded_weights):
    params = params.unfreeze()
    layer_names = [0, 2, 4, 6, 7]
    for i in range(5):
        layer_name = layer_names[i]
        weights = loaded_weights["layers.{}.weight".format(layer_name)]
        bias = loaded_weights["layers.{}.bias".format(layer_name)]

        layer_i = params["params"]["Dense_{}".format(i)]
        layer_i["kernel"] = weights.numpy().transpose()
        layer_i["bias"] = bias.numpy()

    return params

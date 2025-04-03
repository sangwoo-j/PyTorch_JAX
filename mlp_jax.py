import jax
import jax.numpy as jnp
from jax import random

activation_functions = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
}

def init_mlp_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes) - 1)
    params = [
        {"w": random.normal(k, (m, n)) * jnp.sqrt(2/m), "b": jnp.zeros(n)}
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]
    return params

def mlp_forward(params, x, activation="tanh"):
    act_fn = activation_functions[activation]
    for layer in params[:-1]:
        x = act_fn(jnp.dot(x, layer["w"]) + layer["b"])
    output = jnp.dot(x, params[-1]["w"]) + params[-1]["b"]
    return output
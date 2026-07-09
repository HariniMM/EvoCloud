"""
model.py

A tiny feedforward neural network used as the 'genome' for evolution.
Instead of training weights with backprop, we treat the weights themselves
as a chromosome that gets mutated and recombined across generations.
"""

import numpy as np


class TinyMLP:
    """
    A minimal 2-layer MLP: input -> hidden -> output.
    Weights are stored as a single flat numpy array so that crossover
    and mutation can be applied generically without caring about layer shapes.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1_shape = (input_size, hidden_size)
        self.b1_shape = (hidden_size,)
        self.w2_shape = (hidden_size, output_size)
        self.b2_shape = (output_size,)

        self.total_params = (
            input_size * hidden_size
            + hidden_size
            + hidden_size * output_size
            + output_size
        )

        if weights is None:
            weights = np.random.randn(self.total_params) * 0.5
        assert weights.shape[0] == self.total_params, "weight vector size mismatch"
        self.weights = weights.astype(np.float32)

    def _unpack(self):
        idx = 0
        w1 = self.weights[idx: idx + np.prod(self.w1_shape)].reshape(self.w1_shape)
        idx += np.prod(self.w1_shape)
        b1 = self.weights[idx: idx + np.prod(self.b1_shape)].reshape(self.b1_shape)
        idx += np.prod(self.b1_shape)
        w2 = self.weights[idx: idx + np.prod(self.w2_shape)].reshape(self.w2_shape)
        idx += np.prod(self.w2_shape)
        b2 = self.weights[idx: idx + np.prod(self.b2_shape)].reshape(self.b2_shape)
        return w1, b1, w2, b2

    def forward(self, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._unpack()
        h = np.tanh(x @ w1 + b1)
        out = h @ w2 + b2
        return out

    def act(self, observation: np.ndarray) -> int:
        """Return a discrete action (argmax of output layer)."""
        logits = self.forward(observation)
        return int(np.argmax(logits))

    def clone_with_weights(self, new_weights: np.ndarray) -> "TinyMLP":
        return TinyMLP(self.input_size, self.hidden_size, self.output_size, new_weights)
